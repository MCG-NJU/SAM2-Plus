# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import fnmatch
import inspect
import itertools
import logging
import types
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import hydra

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from .optimizer import Optimizer, set_default_parameters, map_scheduler_cfgs_to_param_groups, _unix_pattern_to_parameter_names, get_module_cls_to_param_names

"""
Add if parameter.requires_grad filter.
"""
def validate_param_group_params_plus(param_groups: List[Dict], model: nn.Module):
    """Check that the param groups are non-overlapping and cover all the parameters.

    Args:
        param_groups: List of all param groups
        model: Model to validate against. The check ensures that all the model
            parameters are part of param_groups
    """
    for pg in param_groups:
        # no param should be repeated within a group
        assert len(pg["params"]) == len(set(pg["params"]))
    parameters = [set(param_group["params"]) for param_group in param_groups]
    model_parameters = {parameter for _, parameter in model.named_parameters() if parameter.requires_grad}
    for p1, p2 in itertools.permutations(parameters, 2):
        assert p1.isdisjoint(p2), "Scheduler generated param_groups should be disjoint"
    assert set.union(*parameters) == model_parameters, (
        "Scheduler generated param_groups must include all parameters of the model."
        f" Found {len(set.union(*parameters))} params whereas model has"
        f" {len(model_parameters)} params"
    )

"""
Change validate_param_group_params to validate_param_group_params_plus.
"""
def construct_optimizer_plus(
    model: torch.nn.Module,
    optimizer_conf: Any,
    options_conf: Mapping[str, List] = None,
    param_group_modifiers_conf: List[Callable] = None,
    param_allowlist: Optional[Set[str]] = None,
    validate_param_groups=True,
) -> Optimizer:
    """
    Constructs a stochastic gradient descent or ADAM (or ADAMw) optimizer
    with momentum. i.e, constructs a torch.optim.Optimizer with zero-weight decay
    Batchnorm and/or no-update 1-D parameters support, based on the config.

    Supports wrapping the optimizer with Layer-wise Adaptive Rate Scaling
    (LARS): https://arxiv.org/abs/1708.03888

    Args:
        model: model to perform stochastic gradient descent
            optimization or ADAM optimization.
        optimizer_conf: Hydra config consisting a partial torch optimizer like SGD or
            ADAM, still missing the params argument which this function provides to
            produce the final optimizer
        param_group_modifiers_conf: Optional user specified functions which can modify
            the final scheduler configs before the optimizer's param groups are built
        param_allowlist: The parameters to optimize. Parameters which are not part of
            this allowlist will be skipped.
        validate_param_groups: If enabled, valides that the produced param_groups don't
            overlap and cover all the model parameters.
    """
    if param_allowlist is None:
        param_allowlist = {name for name, _ in model.named_parameters()}

    named_parameters = {
        name: param
        for name, param in model.named_parameters()
        if name in param_allowlist
    }

    if not options_conf:
        optimizer = hydra.utils.instantiate(optimizer_conf, named_parameters.values())
        return Optimizer(optimizer)

    all_parameter_names = {
        name for name, _ in model.named_parameters() if name in param_allowlist
    }
    module_cls_to_all_param_names = get_module_cls_to_param_names(
        model, param_allowlist
    )

    scheduler_cfgs_per_option = hydra.utils.instantiate(options_conf)
    all_scheduler_cfgs = []
    for option, scheduler_cfgs in scheduler_cfgs_per_option.items():
        for config in scheduler_cfgs:
            config.option = option
            config.parameter_names = _unix_pattern_to_parameter_names(
                config, all_parameter_names, module_cls_to_all_param_names
            )
        set_default_parameters(scheduler_cfgs, all_parameter_names)
        all_scheduler_cfgs.append(scheduler_cfgs)

    if param_group_modifiers_conf:
        for custom_param_modifier in param_group_modifiers_conf:
            custom_param_modifier = hydra.utils.instantiate(custom_param_modifier)
            all_scheduler_cfgs = custom_param_modifier(
                scheduler_cfgs=all_scheduler_cfgs, model=model
            )
    schedulers, param_groups = map_scheduler_cfgs_to_param_groups(
        all_scheduler_cfgs, named_parameters
    )
    if validate_param_groups:
        validate_param_group_params_plus(param_groups, model)
    optimizer = hydra.utils.instantiate(optimizer_conf, param_groups)
    return Optimizer(optimizer, schedulers)
