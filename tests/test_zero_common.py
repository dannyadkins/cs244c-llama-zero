import torch

from zero.common import assign_flat_params, build_flat_param_metadata, flatten_params_fp32


def test_assign_flat_params_reuses_existing_storage_when_shapes_match() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    meta = build_flat_param_metadata(model)

    original_ptrs = [param.data.data_ptr() for param in meta.params]
    updated = flatten_params_fp32(meta) + 1.0
    assign_flat_params(meta, updated)

    assert [param.data.data_ptr() for param in meta.params] == original_ptrs