"""The module grouping the restricted-write scripts and the per-module figure share."""

import numpy as np

from goalmisgen.analysis.layers import forward_order, module_groups, module_of, restrict


def test_drc_paths_group_into_named_modules_in_forward_order():
    paths = {
        "params/network_params/conv_list_0/kernel": "input conv",
        "params/network_params/conv_list_0/bias": "input conv",
        "params/network_params/conv_list_1/kernel": "conv 1",
        "params/network_params/cell_list_0/ih/kernel": "cell 0 input",
        "params/network_params/cell_list_0/fence/kernel": "cell 0 input",
        "params/network_params/cell_list_0/hh/kernel": "cell 0 recurrent",
        "params/network_params/cell_list_2/project": "cell 2 input",
        "params/network_params/dense_list_0/kernel": "dense",
        "params/actor_params/Output/kernel": "heads",
        "params/critic_params/Output/bias": "heads",
    }
    for path, module in paths.items():
        assert module_of(path) == module, path
    assert forward_order(set(paths.values())) == [
        "input conv",
        "conv 1",
        "cell 0 input",
        "cell 0 recurrent",
        "cell 2 input",
        "dense",
        "heads",
    ]


def test_resnet_vit_and_bc_paths_group_too():
    assert module_of("params/network_params/GuezResNet_0/GuezConvSequence_0/xXx_Input_xXx/kernel") == "input conv"
    assert module_of("params/network_params/GuezResNet_0/GuezConvSequence_3/GuezResidualBlock_1/Conv_0/kernel") == "stage 3"
    assert module_of("params/network_params/GuezResNet_0/Dense_0/kernel") == "dense"
    assert module_of("params/network_params/Dense_0/kernel") == "dense"
    assert module_of("params/network_params/final_norm/scale") == "dense"
    assert module_of("params/network_params/embed/kernel") == "patch embed"
    assert module_of("params/network_params/position") == "patch embed"
    assert module_of("params/network_params/block_2/attention/out/kernel") == "block 2 attention"
    assert module_of("params/network_params/block_2/mlp_in/kernel") == "block 2 mlp"
    assert module_of("params/block_1/attention/qkv/kernel") == "block 1 attention"
    assert module_of("params/block_3/mlp_out/kernel") == "block 3 mlp"
    assert module_of("params/cell_in/kernel") == "embed"
    assert module_of("params/action_position") == "embed"
    assert module_of("params/sep") == "embed"
    assert module_of("params/ln_final/scale") == "head"
    assert module_of("params/head/kernel") == "head"
    assert forward_order({"embed", "block 1 mlp", "block 0 attention", "block 0 mlp", "head"}) == [
        "embed",
        "block 0 attention",
        "block 0 mlp",
        "block 1 mlp",
        "head",
    ]


def test_module_groups_cover_a_pytree_once_and_restrict_keeps_only_one():
    params = {
        "params": {
            "network_params": {"conv_list_0": {"kernel": np.zeros((2, 3))}, "dense_list_0": {"kernel": np.zeros(4)}},
            "actor_params": {"Output": {"kernel": np.zeros(5)}},
        }
    }
    groups, order = module_groups(params)
    assert order == ["input conv", "dense", "heads"]
    assert {k: len(v) for k, v in groups.items()} == {"input conv": 6, "dense": 4, "heads": 5}
    assert sorted(np.concatenate(list(groups.values()))) == list(range(15))
    vector = np.arange(15, dtype=float)
    kept = restrict(vector, groups, "dense")
    assert kept[groups["dense"]].tolist() == vector[groups["dense"]].tolist()
    assert kept.sum() == vector[groups["dense"]].sum()
