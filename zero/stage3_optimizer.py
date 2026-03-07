from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from collectives import CollectiveOps, LocalCollectives, SendRecvCollectives
from .common import (
    FlatParamMetadata,
    ShardSpec,
    assign_flat_params,
    build_flat_param_metadata_from_params,
    compute_shard_spec,
    flatten_params_fp32,
    get_rank_world_size,
    unique_trainable_params,
)


def _flatten_optional_grads(meta: FlatParamMetadata, grads: Sequence[Optional[torch.Tensor]]) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    for param, grad in zip(meta.params, grads):
        if grad is None:
            parts.append(torch.zeros(param.numel(), dtype=torch.float32, device=meta.device))
        else:
            parts.append(grad.detach().view(-1).to(device=meta.device, dtype=torch.float32))
    return torch.cat(parts, dim=0)


@dataclass
class _Stage3ParamHandle:
    name: str
    meta: FlatParamMetadata
    shard: ShardSpec
    collectives: CollectiveOps
    module_names: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        full_params = flatten_params_fp32(self.meta)
        self.local_param_shard = full_params[self.shard.shard_start : self.shard.shard_end].clone()
        self.exp_avg = torch.zeros(self.shard.shard_numel, dtype=torch.float32, device=self.meta.device)
        self.exp_avg_sq = torch.zeros(self.shard.shard_numel, dtype=torch.float32, device=self.meta.device)
        self.grad_shard = torch.zeros(self.shard.shard_numel, dtype=torch.float32, device=self.meta.device)
        self.materialized = True

    @property
    def world_size(self) -> int:
        return self.shard.world_size

    def materialize(self) -> float:
        if self.materialized:
            return 0.0

        t0 = time.perf_counter()
        full_params = self.collectives.allgather(self.local_param_shard)
        assign_flat_params(self.meta, full_params[: self.meta.total_numel])
        self.materialized = True
        return (time.perf_counter() - t0) * 1000.0

    def reshard(self) -> None:
        if not self.materialized:
            return
        for param in self.meta.params:
            param.grad = None
            param.data = torch.empty(0, dtype=param.dtype, device=param.device)
        self.materialized = False

    def reset_grad(self) -> None:
        self.grad_shard.zero_()

    def accumulate_grad_shard(self, param_grads: Sequence[Optional[torch.Tensor]]) -> float:
        flat_grads = _flatten_optional_grads(self.meta, param_grads)

        t0 = time.perf_counter()
        reduced_shard = self.collectives.reduce_scatter(flat_grads)
        reduced_shard = reduced_shard / self.world_size
        comm_ms = (time.perf_counter() - t0) * 1000.0

        expected = self.shard.shard_numel
        if reduced_shard.numel() < expected:
            raise ValueError(f"reduce_scatter returned too few elements: {reduced_shard.numel()} < {expected}")

        self.grad_shard.add_(reduced_shard[:expected])
        return comm_ms


class _ShardedModuleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, runner, autograd_token: torch.Tensor, *args):
        del autograd_token
        ctx.runner = runner
        ctx.arg_is_tensor = [torch.is_tensor(arg) for arg in args]
        ctx.arg_requires_grad = []
        ctx.non_tensor_args = []

        tensor_args: List[torch.Tensor] = []
        for arg in args:
            if torch.is_tensor(arg):
                tensor_args.append(arg.detach())
                requires_grad = arg.requires_grad and (arg.is_floating_point() or arg.is_complex())
                ctx.arg_requires_grad.append(requires_grad)
                ctx.non_tensor_args.append(None)
            else:
                ctx.arg_requires_grad.append(False)
                ctx.non_tensor_args.append(arg)

        ctx.save_for_backward(*tensor_args)
        ctx.cpu_rng_state = torch.get_rng_state()
        if runner.device.type == "cuda" and torch.cuda.is_available():
            device_index = runner.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            ctx.cuda_rng_state = torch.cuda.get_rng_state(device_index)
        else:
            ctx.cuda_rng_state = None

        if runner.device.type == "cuda":
            ctx.autocast_enabled = torch.is_autocast_enabled()
            ctx.autocast_dtype = torch.get_autocast_gpu_dtype()
        elif runner.device.type == "cpu":
            ctx.autocast_enabled = torch.is_autocast_cpu_enabled()
            ctx.autocast_dtype = torch.get_autocast_cpu_dtype()
        else:
            ctx.autocast_enabled = False
            ctx.autocast_dtype = torch.float32

        runner.engine._record_forward_comm_ms(runner.handle.materialize())
        with torch.no_grad():
            output = runner.module(*args)
        runner.handle.reshard()
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        runner = ctx.runner
        saved_tensors = list(ctx.saved_tensors)
        saved_iter = iter(saved_tensors)
        rebuilt_args: List[object] = []
        grad_inputs: List[torch.Tensor] = []

        devices: List[int] = []
        if runner.device.type == "cuda" and torch.cuda.is_available():
            device_index = runner.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            devices.append(device_index)

        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.set_rng_state(ctx.cpu_rng_state)
            if ctx.cuda_rng_state is not None:
                torch.cuda.set_rng_state(ctx.cuda_rng_state, device=devices[0])

            comm_ms = runner.handle.materialize()

            with torch.enable_grad():
                for is_tensor, requires_grad, non_tensor in zip(
                    ctx.arg_is_tensor,
                    ctx.arg_requires_grad,
                    ctx.non_tensor_args,
                ):
                    if not is_tensor:
                        rebuilt_args.append(non_tensor)
                        continue

                    saved = next(saved_iter)
                    rebuilt = saved.detach()
                    if requires_grad:
                        rebuilt.requires_grad_(True)
                        grad_inputs.append(rebuilt)
                    rebuilt_args.append(rebuilt)

                autocast_enabled = ctx.autocast_enabled and runner.device.type in {"cpu", "cuda"}
                with torch.autocast(
                    device_type=runner.device.type,
                    dtype=ctx.autocast_dtype,
                    enabled=autocast_enabled,
                ):
                    output = runner.module(*rebuilt_args)

                param_inputs = list(runner.handle.meta.params)
                grads = torch.autograd.grad(
                    outputs=output,
                    inputs=grad_inputs + param_inputs,
                    grad_outputs=grad_output,
                    allow_unused=True,
                )

            input_grads = grads[: len(grad_inputs)]
            param_grads = grads[len(grad_inputs) :]
            comm_ms += runner.handle.accumulate_grad_shard(param_grads)
            runner.handle.reshard()

        runner.engine._record_backward_comm_ms(comm_ms)

        grad_args: List[Optional[torch.Tensor]] = []
        input_grad_iter = iter(input_grads)
        for is_tensor, requires_grad in zip(ctx.arg_is_tensor, ctx.arg_requires_grad):
            if is_tensor and requires_grad:
                grad_args.append(next(input_grad_iter))
            else:
                grad_args.append(None)
        return (None, None, *grad_args)


@dataclass
class _Stage3ModuleRunner:
    name: str
    module: nn.Module
    handle: _Stage3ParamHandle
    engine: "ZeROStage3Optimizer"

    def __post_init__(self) -> None:
        self.device = self.handle.meta.device
        self.autograd_token = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True)

    def call(self, *args):
        return _ShardedModuleFunction.apply(self, self.autograd_token, *args)


@dataclass
class ZeROStage3Optimizer:
    """Stage 3 with per-module parameter materialization and backward recomputation.

    This implementation keeps only rank-local parameter shards resident between module calls.
    Parameterized modules are executed through a custom autograd path that:
    - all-gathers one module's parameters immediately before forward
    - releases that module's full parameters after forward
    - all-gathers the same parameters again during backward recomputation
    - reduce-scatters that module's gradients into rank-local optimizer shards
    """

    model: nn.Module
    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    collectives: Optional[CollectiveOps] = None

    def __post_init__(self) -> None:
        rank, world_size = get_rank_world_size()
        self.rank = rank
        self.world_size = world_size

        if self.collectives is None:
            self.collectives = SendRecvCollectives() if world_size > 1 else LocalCollectives()

        self.step_count = 0
        self._forward_comm_ms = 0.0
        self._backward_comm_ms = 0.0
        self._summon_depth = 0

        self.handles: Dict[str, _Stage3ParamHandle] = {}
        self._handle_order: List[_Stage3ParamHandle] = []
        self._module_runners: Dict[str, _Stage3ModuleRunner] = {}

        self._build_module_runners()
        self.model._zero3_executor = self
        self._reshard_all_params()

    def _build_module_runners(self) -> None:
        handle_by_param_ids: Dict[Tuple[int, ...], _Stage3ParamHandle] = {}
        module_specs: List[Tuple[str, nn.Module]] = [("tok_embeddings", self.model.tok_embeddings)]
        module_specs.extend((f"layers.{idx}", layer) for idx, layer in enumerate(self.model.layers))
        module_specs.append(("norm", self.model.norm))
        module_specs.append(("lm_head", self.model.lm_head))

        for module_name, module in module_specs:
            params = unique_trainable_params(list(module.parameters(recurse=True)))
            if not params:
                continue

            key = tuple(sorted(id(param) for param in params))
            handle = handle_by_param_ids.get(key)
            if handle is None:
                meta = build_flat_param_metadata_from_params(params)
                shard = compute_shard_spec(meta.total_numel, rank=self.rank, world_size=self.world_size)
                handle = _Stage3ParamHandle(
                    name=module_name,
                    meta=meta,
                    shard=shard,
                    collectives=self.collectives,
                    module_names=[module_name],
                )
                handle_by_param_ids[key] = handle
                self.handles[handle.name] = handle
                self._handle_order.append(handle)
            elif module_name not in handle.module_names:
                handle.module_names.append(module_name)

            self._module_runners[module_name] = _Stage3ModuleRunner(
                name=module_name,
                module=module,
                handle=handle,
                engine=self,
            )

    def call_module(self, module_name: str, module: nn.Module, *args):
        runner = self._module_runners.get(module_name)
        if runner is None or runner.module is not module:
            return module(*args)
        return runner.call(*args)

    def _record_forward_comm_ms(self, value: float) -> None:
        self._forward_comm_ms += value

    def _record_backward_comm_ms(self, value: float) -> None:
        self._backward_comm_ms += value

    def _reshard_all_params(self) -> None:
        if self._summon_depth > 0:
            return
        for handle in self._handle_order:
            handle.reshard()

    @contextmanager
    def summon_full_params(self) -> Iterator[nn.Module]:
        self._summon_depth += 1
        if self._summon_depth == 1:
            for handle in self._handle_order:
                handle.materialize()
        try:
            yield self.model
        finally:
            self._summon_depth -= 1
            if self._summon_depth == 0:
                self._reshard_all_params()

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def prepare_forward(self) -> Dict[str, float]:
        self._reshard_all_params()
        return {"prepare_comm_ms": 0.0}

    def zero_grad(self) -> None:
        self._forward_comm_ms = 0.0
        self._backward_comm_ms = 0.0
        for handle in self._handle_order:
            handle.reset_grad()
        for param in self.model.parameters():
            param.grad = None
        self._reshard_all_params()

    def _global_grad_norm(self) -> float:
        sum_sq = torch.zeros(1, dtype=torch.float32, device=self._handle_order[0].meta.device)
        for handle in self._handle_order:
            sum_sq += torch.sum(handle.grad_shard * handle.grad_shard)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sum_sq)
        return float(torch.sqrt(sum_sq).item())

    def _clip_local_grads_inplace(self, max_grad_norm: float) -> float:
        grad_norm = self._global_grad_norm()
        if max_grad_norm > 0 and grad_norm > max_grad_norm:
            scale = max_grad_norm / (grad_norm + 1e-6)
            for handle in self._handle_order:
                handle.grad_shard.mul_(scale)
        return grad_norm

    def _adamw_update_handle(self, handle: _Stage3ParamHandle) -> None:
        beta1, beta2 = self.betas
        local_params = handle.local_param_shard
        local_grads = handle.grad_shard

        handle.exp_avg.mul_(beta1).add_(local_grads, alpha=1.0 - beta1)
        handle.exp_avg_sq.mul_(beta2).addcmul_(local_grads, local_grads, value=1.0 - beta2)

        bias_correction1 = 1.0 - (beta1**self.step_count)
        bias_correction2 = 1.0 - (beta2**self.step_count)

        if self.weight_decay != 0.0:
            local_params.mul_(1.0 - (self.lr * self.weight_decay))

        denom = handle.exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(self.eps)
        step_size = self.lr / bias_correction1
        local_params.addcdiv_(handle.exp_avg, denom, value=-step_size)

    def step_with_stats(self, max_grad_norm: float = 0.0) -> Dict[str, float]:
        t0 = time.perf_counter()
        grad_norm = self._clip_local_grads_inplace(max_grad_norm=max_grad_norm)

        self.step_count += 1
        t_opt0 = time.perf_counter()
        for handle in self._handle_order:
            self._adamw_update_handle(handle)
        optim_ms = (time.perf_counter() - t_opt0) * 1000.0

        self._reshard_all_params()
        total_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "grad_norm": grad_norm,
            "comm_ms": self._forward_comm_ms + self._backward_comm_ms,
            "optim_ms": optim_ms,
            "total_ms": total_ms,
        }

    def step(self, max_grad_norm: float = 0.0) -> float:
        return float(self.step_with_stats(max_grad_norm=max_grad_norm)["grad_norm"])

    def state_dict(self) -> Dict[str, object]:
        return {
            "step_count": self.step_count,
            "rank": self.rank,
            "world_size": self.world_size,
            "handles": {
                handle.name: {
                    "module_names": list(handle.module_names),
                    "shard_start": handle.shard.shard_start,
                    "shard_end": handle.shard.shard_end,
                    "local_param_shard": handle.local_param_shard.detach().cpu(),
                    "exp_avg": handle.exp_avg.detach().cpu(),
                    "exp_avg_sq": handle.exp_avg_sq.detach().cpu(),
                }
                for handle in self._handle_order
            },
            "hparams": {
                "lr": self.lr,
                "betas": self.betas,
                "eps": self.eps,
                "weight_decay": self.weight_decay,
            },
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.step_count = int(state["step_count"])
        handle_state = state["handles"]
        for handle in self._handle_order:
            if handle.name not in handle_state:
                raise KeyError(f"missing stage3 handle state for '{handle.name}'")
            saved = handle_state[handle.name]
            handle.local_param_shard = saved["local_param_shard"].to(device=handle.meta.device, dtype=torch.float32)
            handle.exp_avg = saved["exp_avg"].to(device=handle.meta.device, dtype=torch.float32)
            handle.exp_avg_sq = saved["exp_avg_sq"].to(device=handle.meta.device, dtype=torch.float32)
            handle.grad_shard.zero_()
        self._reshard_all_params()
