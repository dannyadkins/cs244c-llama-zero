#define _GNU_SOURCE

#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

typedef ssize_t (*send_fn_t)(int, const void *, size_t, int);
typedef ssize_t (*sendto_fn_t)(int, const void *, size_t, int, const struct sockaddr *, socklen_t);
typedef ssize_t (*sendmsg_fn_t)(int, const struct msghdr *, int);
typedef int (*sendmmsg_fn_t)(int, struct mmsghdr *, unsigned int, int);
typedef ssize_t (*write_fn_t)(int, const void *, size_t);
typedef ssize_t (*writev_fn_t)(int, const struct iovec *, int);

static send_fn_t real_send = NULL;
static sendto_fn_t real_sendto = NULL;
static sendmsg_fn_t real_sendmsg = NULL;
static sendmmsg_fn_t real_sendmmsg = NULL;
static write_fn_t real_write = NULL;
static writev_fn_t real_writev = NULL;

typedef struct {
    int enabled;
    double rate_bytes_per_s;
    double latency_s;
    double burst_bytes;
    double tokens;
    double last_refill_s;
    pthread_mutex_t mu;
} token_bucket_state_t;

typedef struct {
    uint32_t magic;
    uint32_t version;
    volatile uint32_t initialized;
    token_bucket_state_t state;
} shared_shaper_state_t;

#define SHAPER_MAGIC 0x53485052u
#define SHAPER_VERSION 1u

static token_bucket_state_t g_local_state = {
    .enabled = 0,
    .rate_bytes_per_s = 0.0,
    .latency_s = 0.0,
    .burst_bytes = 0.0,
    .tokens = 0.0,
    .last_refill_s = 0.0,
    .mu = PTHREAD_MUTEX_INITIALIZER,
};
static token_bucket_state_t *g_state = &g_local_state;
static shared_shaper_state_t *g_shared_state = NULL;

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ((double)ts.tv_nsec / 1e9);
}

static void sleep_seconds(double seconds) {
    if (seconds <= 0.0) {
        return;
    }

    while (seconds > 0.0) {
        struct timespec req;
        req.tv_sec = (time_t)seconds;
        req.tv_nsec = (long)((seconds - (double)req.tv_sec) * 1e9);
        if (req.tv_nsec < 0) {
            req.tv_nsec = 0;
        }

        if (nanosleep(&req, &req) == 0) {
            return;
        }
        if (errno != EINTR) {
            return;
        }
        seconds = (double)req.tv_sec + ((double)req.tv_nsec / 1e9);
    }
}

static double getenv_double(const char *name, double fallback) {
    const char *value = getenv(name);
    if (value == NULL || value[0] == '\0') {
        return fallback;
    }
    return strtod(value, NULL);
}

static int lock_token_bucket(token_bucket_state_t *state) {
    int rc = pthread_mutex_lock(&state->mu);
#ifdef PTHREAD_MUTEX_ROBUST
    if (rc == EOWNERDEAD) {
        pthread_mutex_consistent(&state->mu);
        rc = 0;
    }
#endif
    return rc;
}

static void configure_token_bucket(token_bucket_state_t *state, double bandwidth_gbps, double latency_ms, double burst_bytes) {
    state->enabled = (bandwidth_gbps > 0.0) || (latency_ms > 0.0);
    state->rate_bytes_per_s = bandwidth_gbps > 0.0 ? bandwidth_gbps * 1e9 : 0.0;
    state->latency_s = latency_ms > 0.0 ? latency_ms / 1000.0 : 0.0;
    state->burst_bytes = burst_bytes > 0.0 ? burst_bytes : 262144.0;
    state->tokens = state->burst_bytes;
    state->last_refill_s = monotonic_seconds();
}

static int build_shared_name(const char *raw_name, char *buffer, size_t size) {
    if (raw_name == NULL || raw_name[0] == '\0' || size < 2) {
        return 0;
    }
    if (raw_name[0] == '/') {
        if (snprintf(buffer, size, "%s", raw_name) >= (int)size) {
            return 0;
        }
        return 1;
    }
    if (snprintf(buffer, size, "/%s", raw_name) >= (int)size) {
        return 0;
    }
    return 1;
}

static shared_shaper_state_t *open_shared_state(
    const char *raw_name,
    double bandwidth_gbps,
    double latency_ms,
    double burst_bytes
) {
    char shared_name[256];
    if (!build_shared_name(raw_name, shared_name, sizeof(shared_name))) {
        return NULL;
    }

    int created = 0;
    int fd = shm_open(shared_name, O_RDWR | O_CREAT | O_EXCL, 0600);
    if (fd >= 0) {
        created = 1;
        if (ftruncate(fd, (off_t)sizeof(shared_shaper_state_t)) != 0) {
            close(fd);
            shm_unlink(shared_name);
            return NULL;
        }
    } else if (errno == EEXIST) {
        fd = shm_open(shared_name, O_RDWR, 0600);
    }

    if (fd < 0) {
        return NULL;
    }

    void *mapping = mmap(NULL, sizeof(shared_shaper_state_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED) {
        if (created) {
            shm_unlink(shared_name);
        }
        return NULL;
    }

    shared_shaper_state_t *shared = (shared_shaper_state_t *)mapping;
    if (created) {
        memset(shared, 0, sizeof(*shared));

        pthread_mutexattr_t attr;
        if (pthread_mutexattr_init(&attr) != 0) {
            munmap(mapping, sizeof(shared_shaper_state_t));
            shm_unlink(shared_name);
            return NULL;
        }
        pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
#ifdef PTHREAD_MUTEX_ROBUST
        pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST);
#endif
        if (pthread_mutex_init(&shared->state.mu, &attr) != 0) {
            pthread_mutexattr_destroy(&attr);
            munmap(mapping, sizeof(shared_shaper_state_t));
            shm_unlink(shared_name);
            return NULL;
        }
        pthread_mutexattr_destroy(&attr);

        shared->magic = SHAPER_MAGIC;
        shared->version = SHAPER_VERSION;
        configure_token_bucket(&shared->state, bandwidth_gbps, latency_ms, burst_bytes);
        __sync_synchronize();
        shared->initialized = 1;
    } else {
        for (int attempt = 0; attempt < 5000 && !shared->initialized; attempt++) {
            sleep_seconds(0.001);
        }
        if (!shared->initialized || shared->magic != SHAPER_MAGIC || shared->version != SHAPER_VERSION) {
            munmap(mapping, sizeof(shared_shaper_state_t));
            return NULL;
        }
    }

    g_shared_state = shared;
    return shared;
}

static void load_real_symbols(void) {
    if (real_send == NULL) {
        real_send = (send_fn_t)dlsym(RTLD_NEXT, "send");
    }
    if (real_sendto == NULL) {
        real_sendto = (sendto_fn_t)dlsym(RTLD_NEXT, "sendto");
    }
    if (real_sendmsg == NULL) {
        real_sendmsg = (sendmsg_fn_t)dlsym(RTLD_NEXT, "sendmsg");
    }
    if (real_sendmmsg == NULL) {
        real_sendmmsg = (sendmmsg_fn_t)dlsym(RTLD_NEXT, "sendmmsg");
    }
    if (real_write == NULL) {
        real_write = (write_fn_t)dlsym(RTLD_NEXT, "write");
    }
    if (real_writev == NULL) {
        real_writev = (writev_fn_t)dlsym(RTLD_NEXT, "writev");
    }
}

static int is_inet_socket_fd(int fd) {
    int socket_type = 0;
    socklen_t type_len = sizeof(socket_type);
    if (getsockopt(fd, SOL_SOCKET, SO_TYPE, &socket_type, &type_len) != 0) {
        return 0;
    }

    struct sockaddr_storage addr;
    socklen_t addr_len = sizeof(addr);
    if (getpeername(fd, (struct sockaddr *)&addr, &addr_len) == 0) {
        return addr.ss_family == AF_INET || addr.ss_family == AF_INET6;
    }

    addr_len = sizeof(addr);
    if (getsockname(fd, (struct sockaddr *)&addr, &addr_len) == 0) {
        return addr.ss_family == AF_INET || addr.ss_family == AF_INET6;
    }

    return 0;
}

static void maybe_shape_send(int fd, size_t num_bytes) {
    token_bucket_state_t *state = g_state;
    if (!state->enabled || num_bytes == 0) {
        return;
    }
    if (!is_inet_socket_fd(fd)) {
        return;
    }

    double bandwidth_sleep_s = 0.0;
    if (lock_token_bucket(state) != 0) {
        return;
    }

    if (state->rate_bytes_per_s > 0.0) {
        double now_s = monotonic_seconds();
        if (state->last_refill_s <= 0.0) {
            state->last_refill_s = now_s;
            state->tokens = state->burst_bytes;
        }

        if (now_s > state->last_refill_s) {
            state->tokens += (now_s - state->last_refill_s) * state->rate_bytes_per_s;
            if (state->tokens > state->burst_bytes) {
                state->tokens = state->burst_bytes;
            }
            state->last_refill_s = now_s;
        }

        if ((double)num_bytes <= state->tokens) {
            state->tokens -= (double)num_bytes;
        } else {
            double deficit_bytes = (double)num_bytes - state->tokens;
            state->tokens = 0.0;
            bandwidth_sleep_s = deficit_bytes / state->rate_bytes_per_s;
            state->last_refill_s = now_s + bandwidth_sleep_s;
        }
    }

    double latency_sleep_s = state->latency_s;
    pthread_mutex_unlock(&state->mu);

    if (latency_sleep_s > 0.0) {
        sleep_seconds(latency_sleep_s);
    }
    if (bandwidth_sleep_s > 0.0) {
        sleep_seconds(bandwidth_sleep_s);
    }
}

static size_t iovec_total_bytes(const struct iovec *iov, int iovcnt) {
    size_t total = 0;
    for (int idx = 0; idx < iovcnt; idx++) {
        total += iov[idx].iov_len;
    }
    return total;
}

__attribute__((constructor)) static void init_socket_shaper(void) {
    load_real_symbols();

    double bandwidth_gbps = getenv_double("ZERO_SOCKET_SHAPER_BW_GBPS", 0.0);
    double latency_ms = getenv_double("ZERO_SOCKET_SHAPER_LATENCY_MS", 0.0);
    double burst_bytes = getenv_double("ZERO_SOCKET_SHAPER_BURST_BYTES", 262144.0);
    const char *shared_name = getenv("ZERO_SOCKET_SHAPER_SHARED_NAME");

    if (shared_name != NULL && shared_name[0] != '\0') {
        shared_shaper_state_t *shared = open_shared_state(shared_name, bandwidth_gbps, latency_ms, burst_bytes);
        if (shared != NULL) {
            g_state = &shared->state;
            return;
        }
    }

    configure_token_bucket(&g_local_state, bandwidth_gbps, latency_ms, burst_bytes);
}

ssize_t send(int sockfd, const void *buf, size_t len, int flags) {
    load_real_symbols();
    maybe_shape_send(sockfd, len);
    return real_send(sockfd, buf, len, flags);
}

ssize_t sendto(int sockfd, const void *buf, size_t len, int flags, const struct sockaddr *dest_addr, socklen_t addrlen) {
    load_real_symbols();
    maybe_shape_send(sockfd, len);
    return real_sendto(sockfd, buf, len, flags, dest_addr, addrlen);
}

ssize_t sendmsg(int sockfd, const struct msghdr *msg, int flags) {
    load_real_symbols();
    maybe_shape_send(sockfd, iovec_total_bytes(msg->msg_iov, msg->msg_iovlen));
    return real_sendmsg(sockfd, msg, flags);
}

int sendmmsg(int sockfd, struct mmsghdr *msgvec, unsigned int vlen, int flags) {
    load_real_symbols();
    size_t total = 0;
    for (unsigned int idx = 0; idx < vlen; idx++) {
        total += iovec_total_bytes(msgvec[idx].msg_hdr.msg_iov, msgvec[idx].msg_hdr.msg_iovlen);
    }
    maybe_shape_send(sockfd, total);
    return real_sendmmsg(sockfd, msgvec, vlen, flags);
}

ssize_t write(int fd, const void *buf, size_t count) {
    load_real_symbols();
    maybe_shape_send(fd, count);
    return real_write(fd, buf, count);
}

ssize_t writev(int fd, const struct iovec *iov, int iovcnt) {
    load_real_symbols();
    maybe_shape_send(fd, iovec_total_bytes(iov, iovcnt));
    return real_writev(fd, iov, iovcnt);
}
