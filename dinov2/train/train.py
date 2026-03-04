# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os
from functools import partial
from pathlib import Path

from fvcore.common.checkpoint import PeriodicCheckpointer
from omegaconf import OmegaConf
import torch
from torchvision.utils import save_image

from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data import (
    MorphologyNeighborhoodAugmentation,
    MorphologyNeighborhoodSameViewAugmentation,
    MorphologyNeighborhoodFourWayAugmentation,
    SingleViewAugmentation,
    RandomMixedViewAugmentation,
    collate_data_and_cast,
    DataAugmentationDINO,
    MaskingGenerator,
)
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler
from dinov2.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from dinov2.train.ssl_meta_arch import SSLMetaArch


torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def _unnormalize_images(images):
    mean = images.new_tensor(IMAGENET_DEFAULT_MEAN).view(1, -1, 1, 1)
    std = images.new_tensor(IMAGENET_DEFAULT_STD).view(1, -1, 1, 1)
    return images * std + mean


def maybe_dump_inputs(data, dump_dir, iteration, max_samples, tag):
    dump_path = Path(dump_dir)
    dump_path.mkdir(parents=True, exist_ok=True)

    def _save_grid(tensor, prefix):
        if tensor is None or tensor.numel() == 0:
            return
        subset = tensor[:max_samples].float()
        subset = torch.clamp(_unnormalize_images(subset), 0.0, 1.0).cpu()
        grid_path = dump_path / f"iter{iteration:06d}_{tag}_{prefix}.png"
        save_image(subset, str(grid_path), nrow=min(max_samples, 4))

    _save_grid(data.get("collated_global_crops"), "global")
    _save_grid(data.get("collated_local_crops"), "local")


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def _patch_wandb_working_set():
    if getattr(_patch_wandb_working_set, "_applied", False):
        return
    try:
        from wandb import util as wandb_util  # type: ignore
        from importlib import metadata as importlib_metadata
    except Exception:
        return

    def safe_working_set():
        for dist in importlib_metadata.distributions():
            metadata = getattr(dist, "metadata", None)
            if metadata is None:
                continue
            try:
                name = metadata["Name"]
            except (KeyError, TypeError):
                continue
            if not name:
                continue
            try:
                yield wandb_util.InstalledDistribution(key=name, version=getattr(dist, "version", ""))
            except Exception:
                continue

    wandb_util.working_set = safe_working_set
    _patch_wandb_working_set._applied = True


def init_wandb(cfg):
    wandb_cfg = getattr(cfg, "wandb", None)
    if not wandb_cfg or not wandb_cfg.enabled:
        return None
    if not distributed.is_main_process():
        os.environ.setdefault("WANDB_MODE", "offline")
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is enabled in the config, but the wandb package is not installed.") from exc

    _patch_wandb_working_set()

    init_kwargs = {
        "project": wandb_cfg.project,
        "dir": cfg.train.output_dir,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if wandb_cfg.entity:
        init_kwargs["entity"] = wandb_cfg.entity
    if wandb_cfg.group:
        init_kwargs["group"] = wandb_cfg.group
    if wandb_cfg.run_name:
        init_kwargs["name"] = wandb_cfg.run_name
    if wandb_cfg.tags:
        init_kwargs["tags"] = list(wandb_cfg.tags)

    run = wandb.init(**init_kwargs)
    return run


def do_test(cfg, model, iteration):
    new_state_dict = model.teacher.state_dict()

    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        # save teacher checkpoint
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)


def do_train(cfg, model, resume=False, wandb_run=None):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training

    dump_inputs_dir = os.environ.get("DINO_DUMP_INPUTS", "")
    dump_inputs_max = int(os.environ.get("DINO_DUMP_INPUTS_MAX", "16"))
    dump_inputs_done = False

    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)

    # checkpointer
    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)

    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=3 * OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=3,
    )

    # setup data preprocessing

    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    dataset_name = cfg.train.dataset_path.split(":")[0]
    single_view_source = getattr(cfg.train, "single_view_source", "")
    dual_view_mode = getattr(cfg.train, "dual_view_mode", "paired")

    normalize_mean = getattr(cfg.crops, "normalize_mean", None)
    normalize_std = getattr(cfg.crops, "normalize_std", None)
    norm_kw = {"normalize_mean": normalize_mean, "normalize_std": normalize_std}

    if dataset_name in {"MorphNeighborhood", "MorphNeighborhoodH5"} and dual_view_mode == "same_view":
        data_transform = MorphologyNeighborhoodSameViewAugmentation(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            1, 
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
            **norm_kw,
        )
    elif dataset_name in {"MorphNeighborhood", "MorphNeighborhoodH5"} and dual_view_mode == "four_way":
        data_transform = MorphologyNeighborhoodFourWayAugmentation(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            1,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
            **norm_kw,
        )
    elif dataset_name in {"MorphNeighborhood", "MorphNeighborhoodH5"} and single_view_source == "mixed":
        data_transform = RandomMixedViewAugmentation(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
            **norm_kw,
        )
    elif dataset_name in {"MorphNeighborhood", "MorphNeighborhoodH5"} and single_view_source:
        data_transform = SingleViewAugmentation(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
            source_key=single_view_source,
            **norm_kw,
        )
    elif dataset_name in {"MorphNeighborhood", "MorphNeighborhoodH5"}:
        data_transform = MorphologyNeighborhoodAugmentation(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
            **norm_kw,
        )
    else:
        data_transform = DataAugmentationDINO(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
            **norm_kw,
        )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    # setup data loader

    dataset = make_dataset(
        dataset_str=cfg.train.dataset_path,
        transform=data_transform,
        target_transform=lambda _: (),
    )
    # sampler_type = SamplerType.INFINITE
    sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=start_iter,  # TODO: Fix this -- cfg.train.seed
        sampler_type=sampler_type,
        sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # training loop

    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"

    for data in metric_logger.log_every(
        data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
        n_global_crops = data.get("n_global_crops", 2)
        current_batch_size = data["collated_global_crops"].shape[0] / n_global_crops
        if iteration > max_iter:
            return

        # apply schedules

        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # compute losses

        optimizer.zero_grad(set_to_none=True)
        if dump_inputs_dir and not dump_inputs_done:
            maybe_dump_inputs(data, dump_inputs_dir, iteration, max_samples=dump_inputs_max, tag="train")
            dump_inputs_done = True
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

        # clip gradients

        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            optimizer.step()

        # perform teacher EMA update

        model.update_teacher(mom)

        # logging

        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        losses_reduced = sum(loss_dict_reduced.values())
        non_finite_components = {k: v for k, v in loss_dict_reduced.items() if not math.isfinite(v)}
        if (not math.isfinite(losses_reduced)) or non_finite_components:
            logger.error(
                "NaN or Inf detected in reduced losses. iteration=%s losses=%s non_finite=%s",
                iteration,
                losses_reduced,
                non_finite_components,
            )
            for key, tensor_loss in loss_dict.items():
                if torch.isnan(tensor_loss).any() or torch.isinf(tensor_loss).any():
                    logger.error("Raw loss '%s' contains non-finite values: %s", key, tensor_loss)
            raise AssertionError("NaN detected during training")

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)
        loss_metrics_for_logging = {"total_loss": losses_reduced, **loss_dict_reduced}

        if wandb_run is not None and (iteration % cfg.wandb.log_every == 0):
            wandb_metrics = {
                "iteration": iteration,
                "lr": lr,
                "weight_decay": wd,
                "momentum_teacher": mom,
                "last_layer_lr": last_layer_lr,
                "teacher_temp": teacher_temp,
                "current_batch_size": current_batch_size,
                "total_loss": losses_reduced,
            }
            for key, raw_value in loss_metrics_for_logging.items():
                meter = metric_logger.meters.get(key)
                if meter is not None:
                    wandb_metrics[f"loss/{key}"] = meter.median
                else:
                    wandb_metrics[f"loss/{key}"] = raw_value
                wandb_metrics[f"loss_raw/{key}"] = raw_value
            wandb_run.log(wandb_metrics, step=iteration)

        # checkpointing and testing

        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)

        iteration = iteration + 1
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup(args)
    wandb_run = init_wandb(cfg)

    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        result = do_test(cfg, model, f"manual_{iteration}")
        if wandb_run is not None:
            wandb_run.finish()
        return result

    try:
        do_train(cfg, model, resume=not args.no_resume, wandb_run=wandb_run)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
