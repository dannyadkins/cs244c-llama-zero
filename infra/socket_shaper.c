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

typedef enum {
    SHAPER_OP_SEND = 0,
    SHAPER_OP_SENDTO = 1,
    SHAPER_OP_SENDMSG = 2,
    SHAPER_OP_SENDMMSG = 3,
    SHAPER_OP_WRITE = 4,
    SHAPER_OP_WRITEV = 5,
    SHAPER_OP_COUNT = 6,
} shaper_op_t;

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
static pthread_mutex_t g_stats_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_total_shaped_calls = 0;
static uint64_t g_total_shaped_bytes = 0;
static uint64_t g_total_inet_fd_calls = 0;
static uint64_t g_total_inet_fd_bytes = 0;
static double g_total_bandwidth_sleep_s = 0.0;
static double g_total_latency_sleep_s = 0.0;
static uint64_t g_op_calls[SHAPER_OP_COUNT] = {0};
static uint64_t g_op_bytes[SHAPER_OP_COUNT] = {0};
static char g_stats_dir[512] = {0};

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

static const char *op_name(shaper_op_t op) {
    switch (op) {
        case SHAPER_OP_SEND:
            return "send";
        case SHAPER_OP_SENDTO:
            return "sendto";
        case SHAPER_OP_SENDMSG:
            return "sendmsg";
        case SHAPER_OP_SENDMMSG:
            return "sendmmsg";
        case SHAPER_OP_WRITE:
            return "write";
        case SHAPER_OP_WRITEV:
            return "writev";
        default:
            return "unknown";
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
    state->rate_bytes_per_s = bandwidth_gbps > 0.0 ? (bandwidth_gbps * 1e9 / 8.0) : 0.0;
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

static void record_socket_stats(
    shaper_op_t op,
    size_t num_bytes,
    int inet_socket,
    double bandwidth_sleep_s,
    double latency_sleep_s
) {
    pthread_mutex_lock(&g_stats_mu);
    if ((int)op >= 0 && op < SHAPER_OP_COUNT) {
        g_op_calls[op] += 1;
        g_op_bytes[op] += (uint64_t)num_bytes;
    }
    if (inet_socket) {
        g_total_inet_fd_calls += 1;
        g_total_inet_fd_bytes += (uint64_t)num_bytes;
    }
    if (bandwidth_sleep_s > 0.0 || latency_sleep_s > 0.0) {
        g_total_shaped_calls += 1;
        g_total_shaped_bytes += (uint64_t)num_bytes;
        g_total_bandwidth_sleep_s += bandwidth_sleep_s;
        g_total_latency_sleep_s += latency_sleep_s;
    }
    pthread_mutex_unlock(&g_stats_mu);
}

static void write_stats_file(void) {
    if (g_stats_dir[0] == '\0') {
        return;
    }

    char path[768];
    int written = snprintf(path, sizeof(path), "%s/socket_shaper_%ld.json", g_stats_dir, (long)getpid());
    if (written <= 0 || written >= (int)sizeof(path)) {
        return;
    }

    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return;
    }

    const char *local_rank = getenv("LOCAL_RANK");
    const char *rank = getenv("RANK");
    const char *world_size = getenv("WORLD_SIZE");

    pthread_mutex_lock(&g_stats_mu);
    fprintf(fp, "{\n");
    fprintf(fp, "  \"pid\": %ld,\n", (long)getpid());
    if (local_rank != NULL && local_rank[0] != '\0') {
        fprintf(fp, "  \"local_rank\": %d,\n", atoi(local_rank));
    }
    if (rank != NULL && rank[0] != '\0') {
        fprintf(fp, "  \"rank\": %d,\n", atoi(rank));
    }
    if (world_size != NULL && world_size[0] != '\0') {
        fprintf(fp, "  \"world_size\": %d,\n", atoi(world_size));
    }
    fprintf(fp, "  \"total_shaped_calls\": %llu,\n", (unsigned long long)g_total_shaped_calls);
    fprintf(fp, "  \"total_shaped_bytes\": %llu,\n", (unsigned long long)g_total_shaped_bytes);
    fprintf(fp, "  \"total_inet_fd_calls\": %llu,\n", (unsigned long long)g_total_inet_fd_calls);
    fprintf(fp, "  \"total_inet_fd_bytes\": %llu,\n", (unsigned long long)g_total_inet_fd_bytes);
    fprintf(fp, "  \"total_bandwidth_sleep_s\": %.9f,\n", g_total_bandwidth_sleep_s);
    fprintf(fp, "  \"total_latency_sleep_s\": %.9f,\n", g_total_latency_sleep_s);
    fprintf(fp, "  \"ops\": [\n");
    for (int idx = 0; idx < SHAPER_OP_COUNT; idx++) {
        fprintf(
            fp,
            "    {\"op\": \"%s\", \"calls\": %llu, \"bytes\": %llu}%s\n",
            op_name((shaper_op_t)idx),
            (unsigned long long)g_op_calls[idx],
            (unsigned long long)g_op_bytes[idx],
            idx + 1 == SHAPER_OP_COUNT ? "" : ","
        );
    }
    fprintf(fp, "  ]\n");
    fprintf(fp, "}\n");
    pthread_mutex_unlock(&g_stats_mu);

    fclose(fp);
}

static void maybe_shape_send(int fd, size_t num_bytes, shaper_op_t op) {
    token_bucket_state_t *state = g_state;
    if (!state->enabled || num_bytes == 0) {
        record_socket_stats(op, num_bytes, 0, 0.0, 0.0);
        return;
    }
    int inet_socket = is_inet_socket_fd(fd);
    if (!inet_socket) {
        record_socket_stats(op, num_bytes, 0, 0.0, 0.0);
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

    record_socket_stats(op, num_bytes, inet_socket, bandwidth_sleep_s, latency_sleep_s);

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
    const char *stats_dir = getenv("ZERO_SOCKET_SHAPER_STATS_DIR");

    if (stats_dir != NULL && stats_dir[0] != '\0') {
        if (snprintf(g_stats_dir, sizeof(g_stats_dir), "%s", stats_dir) < (int)sizeof(g_stats_dir)) {
            atexit(write_stats_file);
        } else {
            g_stats_dir[0] = '\0';
        }
    }

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
    maybe_shape_send(sockfd, len, SHAPER_OP_SEND);
    return real_send(sockfd, buf, len, flags);
}

ssize_t sendto(int sockfd, const void *buf, size_t len, int flags, const struct sockaddr *dest_addr, socklen_t addrlen) {
    load_real_symbols();
    maybe_shape_send(sockfd, len, SHAPER_OP_SENDTO);
    return real_sendto(sockfd, buf, len, flags, dest_addr, addrlen);
}

ssize_t sendmsg(int sockfd, const struct msghdr *msg, int flags) {
    load_real_symbols();
    maybe_shape_send(sockfd, iovec_total_bytes(msg->msg_iov, msg->msg_iovlen), SHAPER_OP_SENDMSG);
    return real_sendmsg(sockfd, msg, flags);
}

int sendmmsg(int sockfd, struct mmsghdr *msgvec, unsigned int vlen, int flags) {
    load_real_symbols();
    size_t total = 0;
    for (unsigned int idx = 0; idx < vlen; idx++) {
        total += iovec_total_bytes(msgvec[idx].msg_hdr.msg_iov, msgvec[idx].msg_hdr.msg_iovlen);
    }
    maybe_shape_send(sockfd, total, SHAPER_OP_SENDMMSG);
    return real_sendmmsg(sockfd, msgvec, vlen, flags);
}

ssize_t write(int fd, const void *buf, size_t count) {
    load_real_symbols();
    maybe_shape_send(fd, count, SHAPER_OP_WRITE);
    return real_write(fd, buf, count);
}

ssize_t writev(int fd, const struct iovec *iov, int iovcnt) {
    load_real_symbols();
    maybe_shape_send(fd, iovec_total_bytes(iov, iovcnt), SHAPER_OP_WRITEV);
    return real_writev(fd, iov, iovcnt);
}
