# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import logging
import os
import random
import sys
import traceback
from argparse import ArgumentParser

import submitit
import torch

from hydra import compose, initialize_config_module
from hydra.utils import instantiate

from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf

from training.utils.train_utils import makedir, register_omegaconf_resolvers

os.environ["HYDRA_FULL_ERROR"] = "1"


def single_proc_run(local_rank, main_port, cfg, world_size):
    """Single GPU process"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        register_omegaconf_resolvers()
    except Exception as e:
        logging.info(e)

    trainer = instantiate(cfg.trainer, _recursive_=False)
    trainer.run()


def single_node_runner(cfg, main_port: int):
    assert cfg.launcher.num_nodes == 1
    num_proc = cfg.launcher.gpus_per_node
    torch.multiprocessing.set_start_method(
        "spawn"
    )  # CUDA runtime does not support `fork`
    if num_proc == 1:
        # directly call single_proc so we can easily set breakpoints
        # mp.spawn does not let us set breakpoints
        single_proc_run(local_rank=0, main_port=main_port, cfg=cfg, world_size=num_proc)
    else:
        mp_runner = torch.multiprocessing.start_processes
        args = (main_port, cfg, num_proc)
        # Note: using "fork" below, "spawn" causes time and error regressions. Using
        # spawn changes the default multiprocessing context to spawn, which doesn't
        # interact well with the dataloaders (likely due to the use of OpenCV).
        mp_runner(single_proc_run, args=args, nprocs=num_proc, start_method="spawn")


def format_exception(e: Exception, limit=20):
    traceback_str = "".join(traceback.format_tb(e.__traceback__, limit=limit))
    return f"{type(e).__name__}: {e}\nTraceback:\n{traceback_str}"


def torch_run_ddp(cfg):

    # ---- Setup torch.distributed ----
    add_pythonpath_to_sys_path()
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "NNODES", "NPROC_PER_NODE", "NODE_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")

    register_omegaconf_resolvers()
    cfg_resolved = OmegaConf.to_container(cfg, resolve=False)
    cfg_resolved = OmegaConf.create(cfg_resolved)
    
    trainer = instantiate(cfg_resolved.trainer, _recursive_=False)
    trainer.run()



def add_pythonpath_to_sys_path():
    if "PYTHONPATH" not in os.environ or not os.environ["PYTHONPATH"]:
        return
    sys.path = os.environ["PYTHONPATH"].split(":") + sys.path


def main(args) -> None:
    cfg = compose(config_name=args.config)
    if cfg.launcher.experiment_log_dir is None:
        cfg.launcher.experiment_log_dir = os.path.join(
            os.getcwd(), "../sam2_logs", args.config
        )
    timestep = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    cfg.launcher.experiment_log_dir = os.path.join(
        os.path.dirname(cfg.launcher.experiment_log_dir), f"{timestep}_{args.git_version[:6]}_{os.path.basename(cfg.launcher.experiment_log_dir)}", 
    )
    print("###################### Train App Config ####################")
    print(OmegaConf.to_yaml(cfg))
    print("############################################################")

    add_pythonpath_to_sys_path()
    makedir(cfg.launcher.experiment_log_dir)
    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "config.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg))

    cfg_resolved = OmegaConf.to_container(cfg, resolve=False)
    cfg_resolved = OmegaConf.create(cfg_resolved)

    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "config_resolved.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg_resolved, resolve=True))

    submitit_conf = cfg.get("submitit", None)
    assert submitit_conf is not None, "Missing submitit config"

    submitit_dir = cfg.launcher.experiment_log_dir
    submitit_dir = os.path.join(submitit_dir, "submitit_logs")
    # Priotrize cmd line args
    cfg.launcher.gpus_per_node = (
        args.num_gpus if args.num_gpus is not None else cfg.launcher.gpus_per_node
    )
    cfg.launcher.num_nodes = (
        args.num_nodes if args.num_nodes is not None else cfg.launcher.num_nodes
    )
    submitit_conf.use_cluster = (
        args.use_cluster if args.use_cluster is not None else submitit_conf.use_cluster
    )
    if submitit_conf.use_cluster:
        raise NotImplementedError("Cluster not supported")
    elif args.torchrun_with_ddp:
        torch_run_ddp(cfg=cfg)
    else:
        cfg.launcher.num_nodes = 1
        main_port = random.randint(
            submitit_conf.port_range[0], submitit_conf.port_range[1]
        )
        single_node_runner(cfg, main_port)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--git-version",
        default="debug",
        type=str,
    )
    parser.add_argument(
        "--config-module",
        default='sam2',
        type=str,
        help="root path to config file (e.g. sam2 or sam2_plus)",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="path to config file (e.g. configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml)",
    )
    parser.add_argument(
        "--use-cluster",
        type=int,
        default=None,
        help="whether to launch on a cluster, 0: run locally, 1: run on a cluster",
    )
    parser.add_argument(
        "--torchrun_with_ddp",
        action="store_true",
        help="whether to run the training script directly using torch.multiprocessing.start_processes",
    )
    parser.add_argument("--partition", type=str, default=None, help="SLURM partition")
    parser.add_argument("--account", type=str, default=None, help="SLURM account")
    parser.add_argument("--qos", type=str, default=None, help="SLURM qos")
    parser.add_argument(
        "--num-gpus", type=int, default=None, help="number of GPUS per node"
    )
    parser.add_argument("--num-nodes", type=int, default=None, help="Number of nodes")
    args = parser.parse_args()
    args.use_cluster = bool(args.use_cluster) if args.use_cluster is not None else None

    initialize_config_module(args.config_module, version_base="1.2")
    register_omegaconf_resolvers()
    main(args)