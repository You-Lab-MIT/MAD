#!/usr/bin/env python3
"""
Inference-only for ft_gene_regress.py checkpoints.

What it does:
  1) Loads checkpoint_best.pt / checkpoint_last.pt produced by your training script
  2) Rebuilds the model architecture using ckpt["args"] and ckpt["gene_names"]
  3) Loads labels (CSV/Parquet) to get the same gene column ordering
  4) Matches H5 cell_ids <-> labels index (vectorized)
  5) Runs inference ONLY (no training) and writes:
       - predictions parquet: index=cell_uid, columns=gene_names
       - optional embeddings parquet/pt

Usage example:
  python infer_gene_regress.py \
    --checkpoint /path/to/output_dir/checkpoint_best.pt \
    --h5-file /path/to/data.h5 \
    --labels-csv /path/to/labels.parquet \
    --output-preds /path/to/preds.parquet \
    --batch-size 64 --num-workers 8

Notes:
  - Requires your dinov2 package import path to work (same env as training).
  - If your labels parquet lost index, pass --id-column cell_uid (same as training).
"""

import argparse
import io
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, get_worker_info
from torchvision import transforms
from tqdm import tqdm

from dinov2.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from dinov2.hub import backbones as hub_backbones
from shapely import wkb
from typing import Dict
from shapely import affinity
import matplotlib.pyplot as plt
from shapely.geometry import box
from typing import Optional, Tuple
import numpy as np

LOGGER = logging.getLogger("dinov2.gene_regress.infer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

BACKBONES: Dict[str, Callable[..., nn.Module]] = {
    "dinov2_vits14": hub_backbones.dinov2_vits14,
    "dinov2_vitb14": hub_backbones.dinov2_vitb14,
    "dinov2_vitl14": hub_backbones.dinov2_vitl14,
    "dinov2_vitg14": hub_backbones.dinov2_vitg14,
    "dinov2_vits14_reg": hub_backbones.dinov2_vits14_reg,
    "dinov2_vitb14_reg": hub_backbones.dinov2_vitb14_reg,
    "dinov2_vitl14_reg": hub_backbones.dinov2_vitl14_reg,
    "dinov2_vitg14_reg": hub_backbones.dinov2_vitg14_reg,
}

DINOV2_PRETRAINED_SIZE = 518


# -----------------------------------------------------------------------------
# Small utils (same behavior as your training script)
# -----------------------------------------------------------------------------

def _decode(value) -> str:
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def _fix_duplicated_slide_prefix(cell_id: str, sep: str = "_") -> str:
    parts = cell_id.split(sep)
    if len(parts) >= 3 and parts[0] == parts[1]:
        return sep.join([parts[0]] + parts[2:])
    return cell_id


def get_patch_size(arch: str) -> int:
    if "14" in arch:
        return 14
    if "16" in arch:
        return 16
    return 14


def round_to_patch_size(size: int, patch_size: int = 14) -> int:
    return max(patch_size, (size // patch_size) * patch_size)


def maybe_resize_pos_embed(state_dict: Dict[str, torch.Tensor], backbone: nn.Module):
    key = "pos_embed"
    if key not in state_dict:
        return
    pretrained_pos_embed = state_dict[key]
    current_pos_embed = backbone.pos_embed
    if pretrained_pos_embed.shape == current_pos_embed.shape:
        return

    num_extra_tokens = backbone.num_tokens  # usually 1 (CLS)
    pretrained_token_count = pretrained_pos_embed.shape[1] - num_extra_tokens
    current_token_count = current_pos_embed.shape[1] - num_extra_tokens

    old_size = int(math.sqrt(pretrained_token_count))
    new_size = int(math.sqrt(current_token_count))
    if old_size * old_size != pretrained_token_count or new_size * new_size != current_token_count:
        LOGGER.warning(
            "Unable to reshape positional embeddings from %s to %s; skipping resize.",
            pretrained_pos_embed.shape,
            current_pos_embed.shape,
        )
        return

    cls_pos = pretrained_pos_embed[:, :num_extra_tokens]
    patch_pos = pretrained_pos_embed[:, num_extra_tokens:]
    dim = patch_pos.shape[-1]

    patch_pos = patch_pos.reshape(1, old_size, old_size, dim).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=(new_size, new_size), mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_size * new_size, dim)

    state_dict[key] = torch.cat((cls_pos, patch_pos), dim=1)


def create_backbone_with_interpolation(
    arch: str,
    img_size: int,
    pretrained: bool = True,
    backbone_checkpoint: Optional[str] = None,
) -> nn.Module:
    """
    Create DINOv2 backbone. If img_size != 518, loads pretrained weights and interpolates pos_embed once.
    """
    if pretrained and img_size != DINOV2_PRETRAINED_SIZE:
        backbone = BACKBONES[arch](img_size=img_size, pretrained=False)

        LOGGER.info("Loading pretrained weights + interpolating pos_embed %d -> %d", DINOV2_PRETRAINED_SIZE, img_size)
        pretrained_backbone = BACKBONES[arch](img_size=DINOV2_PRETRAINED_SIZE, pretrained=False)
        state_dict = pretrained_backbone.state_dict()
        del pretrained_backbone

        maybe_resize_pos_embed(state_dict, backbone)
        backbone.load_state_dict(state_dict, strict=False)
    else:
        backbone = BACKBONES[arch](img_size=img_size, pretrained=pretrained)
    if backbone_checkpoint:
        LOGGER.info("Loading custom backbone checkpoint: %s", backbone_checkpoint)
        state = torch.load(backbone_checkpoint, map_location="cpu")
        maybe_resize_pos_embed(state, backbone)
        backbone.load_state_dict(state, strict=False)

    return backbone


def build_eval_transform(image_size: int) -> Callable:
    normalize = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.05), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def detect_view_sizes(h5_path: Path, views: Sequence[str]) -> Dict[str, Tuple[int, int]]:
    sizes = {}
    with h5py.File(h5_path, "r") as handle:
        for view in views:
            if view not in handle:
                raise KeyError(f"View '{view}' not found in {h5_path}")
            shape = handle[view].shape  # (N, H, W, 3)
            if len(shape) != 4:
                raise ValueError(f"View '{view}' has unexpected shape {shape}, expected (N, H, W, 3)")
            sizes[view] = (shape[1], shape[2])
    return sizes


def load_labels(path: Path, id_column: Optional[str]) -> Tuple[pd.Index, np.ndarray, List[str]]:
    """
    Returns:
      labels_index: pd.Index of cell_uids
      labels_values: (N, G) float32 numpy matrix
      gene_names: list of gene columns (length G)
    """
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if id_column:
        if df.index.name == id_column:
            pass
        elif id_column in df.columns:
            df = df.set_index(id_column)
        else:
            raise ValueError(f"--id-column '{id_column}' not found in {path}. Columns: {list(df.columns)[:20]}")
    else:
        if isinstance(df.index, pd.RangeIndex):
            for cand in ("cell_uid", "cell_id", "cellids", "id"):
                if cand in df.columns:
                    df = df.set_index(cand)
                    break
            else:
                obj_cols = [c for c in df.columns if df[c].dtype == object]
                if obj_cols:
                    df = df.set_index(obj_cols[0])
                else:
                    raise ValueError(
                        f"Labels file {path} has RangeIndex and no string id column. "
                        f"Re-save parquet with index OR add a cell_uid column and pass --id-column cell_uid."
                    )

    df.index = df.index.astype(str)

    META_COLS = {"cell_uid", "cell_id", "slide_id", "split"}
    gene_columns = [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]
    if not gene_columns:
        raise ValueError(f"No numeric gene columns found in {path}. First columns: {list(df.columns)[:20]}")

    if not df.index.is_unique:
        dup = df.index[df.index.duplicated()].unique()[:5].tolist()
        raise ValueError(f"Labels index is not unique (examples: {dup}). Fix before inference.")

    gene_columns = list(gene_columns)
    labels_values = df[gene_columns].to_numpy(dtype=np.float32, copy=True)
    labels_index = pd.Index(df.index)

    LOGGER.info("Loaded labels: %s | cells=%d genes=%d", path.name, labels_values.shape[0], labels_values.shape[1])
    return labels_index, labels_values, gene_columns


@dataclass
class MatchedRecords:
    data_idx: np.ndarray   # indices into H5 datasets, int64
    label_idx: np.ndarray  # indices into labels_values, int32

    def __len__(self) -> int:
        return int(self.data_idx.shape[0])


def discover_h5_entries(
    h5_path: Path,
    labels_index: pd.Index,
    view: str = "morphology",
    slide_sep: str = "_",
) -> Tuple[MatchedRecords, List[str]]:
    """
    Matches H5 entries to labels_index. Returns (records, matched_cell_ids).
    matched_cell_ids is only for writing outputs / debugging.
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as handle:
        if "cell_ids" in handle:
            raw = handle["cell_ids"][:]
            raw_cell_ids = [_decode(x) for x in raw]
            if view not in handle:
                available = [k for k in handle.keys() if k not in ("cell_ids", "centroid_xy")]
                raise KeyError(f"View '{view}' not found in {h5_path}. Available: {available}")
        elif "filenames" in handle:
            LOGGER.warning("Detected legacy H5 format (filenames/images) in %s", h5_path)
            raw = handle["filenames"][:]
            raw_cell_ids = [_decode(x).split(".")[0] for x in raw]
        else:
            raise KeyError(f"H5 {h5_path} has neither 'cell_ids' nor 'filenames'.")

    pos = labels_index.get_indexer(raw_cell_ids)
    mask = pos >= 0
    n_match = int(mask.sum())

    if n_match == 0 and len(raw_cell_ids) > 0:
        sample = raw_cell_ids[0]
        fixed_sample = _fix_duplicated_slide_prefix(sample, sep=slide_sep)
        if fixed_sample != sample:
            LOGGER.warning("No matches found; trying duplicated slide prefix fix...")
            fixed_ids = [_fix_duplicated_slide_prefix(cid, sep=slide_sep) for cid in raw_cell_ids]
            pos = labels_index.get_indexer(fixed_ids)
            mask = pos >= 0
            n_match = int(mask.sum())
            raw_cell_ids = fixed_ids

    if n_match == 0:
        raise RuntimeError(
            f"No overlap between H5 cell_ids and labels.\n"
            f"H5 sample: {raw_cell_ids[:5]}\n"
            f"Labels sample: {labels_index[:5].tolist()}"
        )

    data_idx = np.nonzero(mask)[0].astype(np.int64, copy=False)
    label_idx = pos[mask].astype(np.int32, copy=False)

    matched_ids = [raw_cell_ids[i] for i in data_idx]
    LOGGER.info("Matched %d / %d entries in %s", n_match, len(raw_cell_ids), h5_path.name)
    return MatchedRecords(data_idx=data_idx, label_idx=label_idx), matched_ids


def save_h5_subset(
    src_h5: Path,
    dst_h5: Path,
    data_indices: np.ndarray,
    views: Optional[List[str]] = None,
) -> None:
    """
    Copy only the rows at *data_indices* from *src_h5* into a new, compact
    *dst_h5*.  All view datasets listed in *views* (or auto-detected) plus
    ``cell_ids`` and ``centroid_xy`` (if present) are copied.
    """
    src_h5, dst_h5 = Path(src_h5), Path(dst_h5)
    dst_h5.parent.mkdir(parents=True, exist_ok=True)
    idx = np.sort(data_indices)

    with h5py.File(src_h5, "r") as src, h5py.File(dst_h5, "w") as dst:
        meta_keys = {"cell_ids", "centroid_xy", "filenames"}
        if views is None:
            views = [k for k in src.keys() if k not in meta_keys]

        for mk in meta_keys:
            if mk in src:
                dst.create_dataset(mk, data=src[mk][idx])

        for vname in views:
            if vname not in src:
                continue
            src_ds = src[vname]
            subset = src_ds[idx]
            dst.create_dataset(
                vname, data=subset,
                chunks=src_ds.chunks, compression=src_ds.compression,
                compression_opts=src_ds.compression_opts,
            )

        n_src = len(src["cell_ids"]) if "cell_ids" in src else -1

    LOGGER.info(
        "Saved H5 subset: %s -> %s  (%d / %d rows, views=%s)",
        src_h5.name, dst_h5.name, len(idx), n_src, views,
    )


# -----------------------------------------------------------------------------
# Datasets (inference)
# -----------------------------------------------------------------------------

class H5InferenceDataset(Dataset):
    """
    Inference dataset:
      returns (image_tensor, cell_id)
    """
    def __init__(
        self,
        h5_path: Path,
        records: MatchedRecords,
        cell_ids: List[str],
        transform: Callable,
        view: str = "morphology",
        slide_sep: str = "_",
    ):
        self.h5_path = Path(h5_path)
        self.records = records
        self.cell_ids = cell_ids  # aligned with records.data_idx (same order)
        self.transform = transform
        self.view = view
        self.slide_sep = slide_sep
        self._handles: Dict[int, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _get_handle(self) -> h5py.File:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else -1
        handle = self._handles.get(worker_id)
        if handle is None:
            handle = h5py.File(self.h5_path, "r")
            self._handles[worker_id] = handle
        return handle

    def __getitem__(self, i: int):
        data_idx = int(self.records.data_idx[i])
        handle = self._get_handle()

        # load image
        if "cell_ids" in handle:
            arr = handle[self.view][data_idx]  # (H, W, 3) uint8
            img = Image.fromarray(arr, mode="RGB")
        else:
            raw = handle["images"][data_idx]
            blob = raw.tobytes() if hasattr(raw, "tobytes") else bytes(raw)
            with Image.open(io.BytesIO(blob)) as im:
                img = im.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        cid = self.cell_ids[i]
        cid = _fix_duplicated_slide_prefix(cid, sep=self.slide_sep)
        return img, cid

    def close(self):
        for h in self._handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._handles.clear()

    def __del__(self):
        self.close()


class MultiViewH5InferenceDataset(Dataset):
    """
    Multi-view inference dataset:
      returns ({view: tensor}, cell_id)
    """
    def __init__(
        self,
        h5_path: Path,
        records: MatchedRecords,
        cell_ids: List[str],
        views: Sequence[str],
        transforms_by_view: Dict[str, Callable],
        slide_sep: str = "_",
    ):
        self.h5_path = Path(h5_path)
        self.records = records
        self.cell_ids = cell_ids
        self.views = list(views)
        self.transforms_by_view = transforms_by_view
        self.slide_sep = slide_sep
        self._handles: Dict[int, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _get_handle(self) -> h5py.File:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else -1
        handle = self._handles.get(worker_id)
        if handle is None:
            handle = h5py.File(self.h5_path, "r")
            self._handles[worker_id] = handle
        return handle

    def __getitem__(self, i: int):
        data_idx = int(self.records.data_idx[i])
        handle = self._get_handle()

        images: Dict[str, torch.Tensor] = {}
        for view in self.views:
            arr = handle[view][data_idx]  # (H, W, 3)
            img = Image.fromarray(arr, mode="RGB")
            tf = self.transforms_by_view.get(view, None)
            if tf is not None:
                img = tf(img)
            images[view] = img

        cid = _fix_duplicated_slide_prefix(self.cell_ids[i], sep=self.slide_sep)
        return images, cid

    def close(self):
        for h in self._handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._handles.clear()

    def __del__(self):
        self.close()


def multiview_infer_collate_fn(batch):
    images_list, cell_ids = zip(*batch)
    views = images_list[0].keys()
    images = {v: torch.stack([x[v] for x in images_list], dim=0) for v in views}
    return images, list(cell_ids)


def _move_images_to_device(images, device):
    if isinstance(images, dict):
        return {k: v.to(device, non_blocking=True) for k, v in images.items()}
    return images.to(device, non_blocking=True)


# -----------------------------------------------------------------------------
# Model (same modules as training script)
# -----------------------------------------------------------------------------

class AttentionPooling(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.size(0)
        query = self.query.expand(B, -1, -1)
        pooled, _ = self.attn(query, tokens, tokens)
        return self.norm(pooled.squeeze(1))


def pool_backbone_features(backbone_out, pooling: str, attention_pooler: Optional[nn.Module] = None) -> torch.Tensor:
    if isinstance(backbone_out, dict):
        cls_token = backbone_out.get("x_norm_clstoken", backbone_out.get("cls_token"))
        patch_tokens = backbone_out.get("x_norm_patchtokens", backbone_out.get("tokens"))
    else:
        cls_token = backbone_out[:, 0]
        patch_tokens = backbone_out[:, 1:]

    if pooling == "cls":
        return cls_token
    if pooling == "mean":
        return patch_tokens.mean(dim=1)
    if pooling == "cls+mean":
        return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)
    if pooling == "attention_pool":
        if attention_pooler is None:
            raise ValueError("attention_pooler required for pooling='attention_pool'")
        all_tokens = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
        return attention_pooler(all_tokens)

    raise ValueError(f"Unknown pooling: {pooling}")


def get_pooled_dim(embed_dim: int, pooling: str) -> int:
    return embed_dim * 2 if pooling == "cls+mean" else embed_dim


class DinoRegressor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        pooling: str = "cls",
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling

        self.attn_pooler = None
        if pooling == "attention_pool":
            self.attn_pooler = AttentionPooling(embed_dim, num_heads=attention_heads, dropout=attention_dropout)

        pooled_dim = get_pooled_dim(embed_dim, pooling)

        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Dropout(dropout),
            nn.Linear(pooled_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.backbone.forward_features(x)
        pooled = pool_backbone_features(out, self.pooling, self.attn_pooler)
        preds = self.head(pooled)
        return preds, pooled


class FusionAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(feats, feats, feats)
        return self.norm(attn_out.mean(dim=1))


class MultiViewDinoRegressor(nn.Module):
    def __init__(
        self,
        views: Sequence[str],
        view_sizes: Dict[str, int],
        arch: str,
        embed_dim: int,
        out_dim: int,
        share_backbone: bool = False,
        fusion: str = "concat",
        pooling: str = "cls",
        dropout: float = 0.1,
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
        pretrained: bool = True,
        backbone_checkpoint: Optional[str] = None,
    ):
        super().__init__()
        self.views = list(views)
        self.share_backbone = share_backbone
        self.fusion = fusion
        self.pooling = pooling

        if share_backbone:
            shared_size = max(view_sizes.values())
            self.backbone = create_backbone_with_interpolation(
                arch, shared_size, pretrained=False, backbone_checkpoint=None
            )
            self.backbones = None
        else:
            self.backbone = None
            self.backbones = nn.ModuleDict()
            for v in self.views:
                self.backbones[v] = create_backbone_with_interpolation(
                    arch, view_sizes[v], pretrained=False, backbone_checkpoint=None
                )

        self.attn_pooler = None
        self.attn_poolers = None
        if pooling == "attention_pool":
            if share_backbone:
                self.attn_pooler = AttentionPooling(embed_dim, num_heads=attention_heads, dropout=attention_dropout)
            else:
                self.attn_poolers = nn.ModuleDict({
                    v: AttentionPooling(embed_dim, num_heads=attention_heads, dropout=attention_dropout) for v in self.views
                })

        pooled_dim = get_pooled_dim(embed_dim, pooling)
        V = len(self.views)

        if fusion == "concat":
            fused_dim = pooled_dim * V
        elif fusion in ("mean", "attention"):
            fused_dim = pooled_dim
            if fusion == "attention":
                self.fusion_attn = FusionAttention(pooled_dim, num_heads=attention_heads, dropout=attention_dropout)
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, out_dim),
        )

        LOGGER.info("MultiView: views=%s pooling=%s fused_dim=%d share_backbone=%s", self.views, pooling, fused_dim, share_backbone)

    def forward(self, images: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled_list = []
        for v in self.views:
            x = images[v]
            if self.share_backbone:
                out = self.backbone.forward_features(x)
                pooler = self.attn_pooler
            else:
                out = self.backbones[v].forward_features(x)
                pooler = self.attn_poolers[v] if self.attn_poolers is not None else None

            pooled = pool_backbone_features(out, self.pooling, pooler)
            pooled_list.append(pooled)

        if self.fusion == "concat":
            fused = torch.cat(pooled_list, dim=-1)
        else:
            stacked = torch.stack(pooled_list, dim=1)  # (B, V, D)
            if self.fusion == "mean":
                fused = stacked.mean(dim=1)
            else:
                fused = self.fusion_attn(stacked)

        preds = self.head(fused)
        return preds, fused


def freeze_all_params(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


# -----------------------------------------------------------------------------
# Inference loop
# -----------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    gene_names: List[str],
    save_preds: Path,
    save_embeds: Optional[Path] = None,
    fp16: bool = True,
):
    model.eval()
    preds_store = []
    embeds_store = [] if save_embeds is not None else None
    ids: List[str] = []

    pbar = tqdm(loader, desc="Infer", leave=False)
    for images, cell_ids in pbar:
        images = _move_images_to_device(images, device)

        if fp16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                preds, feats = model(images)
        else:
            preds, feats = model(images)

        preds_store.append(preds.float().cpu())
        if embeds_store is not None:
            embeds_store.append(feats.float().cpu())
        ids.extend(cell_ids)

    pred_mat = torch.cat(preds_store, dim=0).numpy()
    df_pred = pd.DataFrame(pred_mat, index=pd.Index(ids, name="cell_uid"), columns=gene_names)
    save_preds = Path(save_preds)
    save_preds.parent.mkdir(parents=True, exist_ok=True)
    df_pred.to_parquet(save_preds)
    LOGGER.info("Wrote predictions: %s | shape=%s", save_preds, df_pred.shape)

    if save_embeds is not None:
        save_embeds = Path(save_embeds)
        save_embeds.parent.mkdir(parents=True, exist_ok=True)
        feats = torch.cat(embeds_store, dim=0)

        if save_embeds.suffix == ".pt":
            torch.save({"cell_ids": ids, "features": feats}, save_embeds)
        else:
            df_f = pd.DataFrame(feats.numpy(), index=pd.Index(ids, name="cell_uid"))
            if save_embeds.suffix == ".parquet":
                df_f.to_parquet(save_embeds)
            else:
                df_f.to_csv(save_embeds)
        LOGGER.info("Wrote embeddings: %s | n=%d dim=%d", save_embeds, feats.shape[0], feats.shape[1])


# -----------------------------------------------------------------------------
# CLI / Main
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Inference-only for DINOv2 gene regression checkpoints")
    p.add_argument("--checkpoint", default=None, help="Path to checkpoint_best.pt / checkpoint_last.pt")
    p.add_argument("--h5-file", default = '/data/datasets/MAD/HEST_h5_new/ALL_NCBI_combined_with_niche.h5', help="Path to H5 with cell_ids and view arrays")
    p.add_argument("--labels-csv", default = './data/cellxgene_top100_counts.parquet', help="Labels parquet/csv used to define gene order + ID mapping")
    p.add_argument("--id-column", default=None, help="If labels has explicit id column, set it here (e.g. cell_uid)")
    p.add_argument("--output-preds", default = './neig_predictions.parquet', help="Output predictions parquet path")
    p.add_argument("--output-embeds", default=None, help="Optional: output embeddings (.pt/.parquet/.csv)")

    # Optional overrides (normally inferred from checkpoint args)
    p.add_argument("--device", default='cuda', help="cuda / cpu (default: auto)")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size (else use ckpt args)")
    p.add_argument("--num-workers", type=int, default=None, help="Override num_workers (else use ckpt args)")

    # If you want to run on a specific subset of slides at inference time
    p.add_argument("--val-slides", nargs="+", default=None, help="If set, only infer on these slides (slide prefix before sep)")
    p.add_argument("--slide-sep", default=None, help="Override slide_sep (else use ckpt args)")

    return p.parse_args()



def main():
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) or {}
    ckpt_gene_names = ckpt.get("gene_names", None)
    if ckpt_gene_names is None:
        raise ValueError("Checkpoint missing 'gene_names'. Use checkpoints produced by your ft_gene_regress.py.")

    # Load labels: we use it primarily to define gene order and to match cell IDs.
    # If labels and ckpt gene_names disagree, we reorder labels to ckpt gene_names (strict).
    labels_index, labels_values, labels_gene_cols = load_labels(Path(args.labels_csv), args.id_column)

    # Ensure gene ordering matches checkpoint
    if list(labels_gene_cols) != list(ckpt_gene_names):
        missing = [g for g in ckpt_gene_names if g not in set(labels_gene_cols)]
        if missing:
            raise ValueError(f"Labels file is missing genes present in checkpoint (examples: {missing[:10]})")
        # reorder
        df_tmp = pd.DataFrame(labels_values, index=labels_index, columns=labels_gene_cols)
        df_tmp = df_tmp.loc[:, ckpt_gene_names]
        labels_values = df_tmp.to_numpy(dtype=np.float32, copy=True)
        labels_gene_cols = list(ckpt_gene_names)
        LOGGER.warning("Reordered labels gene columns to match checkpoint gene_names")

    out_dim = len(labels_gene_cols)

    # Determine mode + views + sizes from checkpoint args
    views = ckpt_args.get("views", None)
    multi_view_mode = views is not None and isinstance(views, (list, tuple)) and len(views) > 0
    arch = "dinov2_vitl14"
    pooling = "cls"
    fusion = "concat"
    share_backbone = False
    dropout = 0.1
    attention_heads = 4
    attention_dropout = 0.1
    backbone_checkpoint = None
    LOGGER.info('767: backbone_checkpoint', backbone_checkpoint)
    h5_view = "morphology"
    slide_sep = '_'

    # device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loaders
    batch_size = int(args.batch_size)
    num_workers = int(args.num_workers)

    h5_path = Path(args.h5_file)

    if multi_view_mode:
        views = list(views)
        raw_sizes = detect_view_sizes(h5_path, views)
        patch_size = get_patch_size(arch)
        view_sizes = {v: round_to_patch_size(min(h, w), patch_size) for v, (h, w) in raw_sizes.items()}
        if share_backbone:
            shared = max(view_sizes.values())
            view_sizes = {v: shared for v in views}

        # match using first view (same as training)
        records, matched_ids = discover_h5_entries(h5_path, labels_index, view=views[0], slide_sep=slide_sep)

        # optional filter by slide at inference time
        if args.val_slides:
            keep = []
            keep_data = []
            keep_label = []
            val_set = set(args.val_slides)
            for i, cid in enumerate(matched_ids):
                slide = cid.split(slide_sep, 1)[0] if slide_sep in cid else cid
                if slide in val_set:
                    keep.append(cid)
                    keep_data.append(records.data_idx[i])
                    keep_label.append(records.label_idx[i])
            if len(keep) == 0:
                raise RuntimeError(f"No matched records found for val_slides={args.val_slides}")
            matched_ids = keep
            records = MatchedRecords(np.asarray(keep_data, dtype=np.int64), np.asarray(keep_label, dtype=np.int32))
            LOGGER.info("Filtered to slides=%s -> n=%d", args.val_slides, len(records))

        eval_tfs = {v: build_eval_transform(view_sizes[v]) for v in views}
        ds = MultiViewH5InferenceDataset(
            h5_path=h5_path,
            records=records,
            cell_ids=matched_ids,
            views=views,
            transforms_by_view=eval_tfs,
            slide_sep=slide_sep,
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=multiview_infer_collate_fn,
        )

        # embed dim (same trick as training)
        dummy = BACKBONES[arch](img_size=DINOV2_PRETRAINED_SIZE, pretrained=False)
        embed_dim = getattr(dummy, "embed_dim", getattr(dummy, "num_features"))
        del dummy

        model = MultiViewDinoRegressor(
            views=views,
            view_sizes=view_sizes,
            arch=arch,
            embed_dim=embed_dim,
            out_dim=out_dim,
            share_backbone=share_backbone,
            fusion=fusion,
            pooling=pooling,
            dropout=dropout,
            attention_heads=attention_heads,
            attention_dropout=attention_dropout,
            pretrained=True,
            backbone_checkpoint=backbone_checkpoint,
        )
    else:
        # single view
        patch_size = get_patch_size(arch)
        view = h5_view

        records, matched_ids = discover_h5_entries(h5_path, labels_index, view=view, slide_sep=slide_sep)

        if args.val_slides:
            val_set = set(args.val_slides)
            keep = []
            keep_data = []
            keep_label = []
            for i, cid in enumerate(matched_ids):
                slide = cid.split(slide_sep, 1)[0] if slide_sep in cid else cid
                if slide in val_set:
                    keep.append(cid)
                    keep_data.append(records.data_idx[i])
                    keep_label.append(records.label_idx[i])
            if len(keep) == 0:
                raise RuntimeError(f"No matched records found for val_slides={args.val_slides}")
            matched_ids = keep
            records = MatchedRecords(np.asarray(keep_data, dtype=np.int64), np.asarray(keep_label, dtype=np.int32))
            LOGGER.info("Filtered to slides=%s -> n=%d", args.val_slides, len(records))

        # determine image size (prefer checkpoint arg if present, else detect from H5)
        image_size = None
        if image_size is None:
            raw_sizes = detect_view_sizes(h5_path, [view])
            h, w = raw_sizes[view]
            image_size = round_to_patch_size(min(h, w), patch_size)
            LOGGER.info("Auto-detected image_size=%d from view shape %dx%d", image_size, h, w)
        else:
            image_size = int(image_size)

        eval_tf = build_eval_transform(image_size)

        ds = H5InferenceDataset(
            h5_path=h5_path,
            records=records,
            cell_ids=matched_ids,
            transform=eval_tf,
            view=view,
            slide_sep=slide_sep,
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        LOGGER.info('903: backbone_checkpoint', backbone_checkpoint)
        backbone = create_backbone_with_interpolation(
            arch=arch,
            img_size=image_size,
            pretrained=False,
            backbone_checkpoint=None,
        )
        embed_dim = getattr(backbone, "embed_dim", getattr(backbone, "num_features"))

        model = DinoRegressor(
            backbone=backbone,
            embed_dim=embed_dim,
            out_dim=out_dim,
            dropout=dropout,
            pooling=pooling,
            attention_heads=attention_heads,
            attention_dropout=attention_dropout,
        )

    # Load weights (strict=True like your training loader)
    model.load_state_dict(ckpt["model"], strict=True)
    freeze_all_params(model)
    model.to(device)

    LOGGER.info(
        "Loaded model from %s | mode=%s | n=%d | genes=%d | device=%s | bs=%d workers=%d",
        ckpt_path.name,
        "multiview" if multi_view_mode else "singleview",
        len(loader.dataset),
        out_dim,
        device,
        batch_size,
        num_workers,
    )

    run_inference(
        model=model,
        loader=loader,
        device=device,
        gene_names=labels_gene_cols,
        save_preds=Path(args.output_preds),
        save_embeds=Path(args.output_embeds) if args.output_embeds else None,
        fp16=True,
    )

def infer_gene_regress_notebook(
    checkpoint: str = "",
    h5_file: str = "",
    labels_csv: str = "",
    output_preds: str = "./neig_predictions.parquet",
    output_embeds: Optional[str] = None,
    device: str = "cuda",
    batch_size: Optional[int] = 32,
    num_workers: Optional[int] = 8,
    id_column: Optional[str] = None,
    val_slides: Optional[List[str]] = None,
    slide_sep_override: Optional[str] = None,
    fp16: bool = True,
    return_embeddings: bool = False,  # if True, returns (df_pred, embeds_tensor_or_df)
    save_subset_h5: Optional[str] = None,  # path to write a smaller H5 containing only the matched rows
):
    """
    Notebook entrypoint:
      - Uses fixed defaults (your paths) unless you override them in the call
      - Runs inference only and saves parquet(s)
      - Returns df_pred (and optionally embeddings)
    """
    torch.backends.cudnn.benchmark = True

    ckpt_path = Path(checkpoint)
    h5_path = Path(h5_file)
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) or {}
    ckpt_gene_names = ckpt.get("gene_names", None)
    if ckpt_gene_names is None:
        raise ValueError("Checkpoint missing 'gene_names'. Use checkpoints produced by your ft_gene_regress.py.")

    # Load labels: define gene order + ID matching
    labels_index, labels_values, labels_gene_cols = load_labels(Path(labels_csv), id_column)

    # Ensure gene ordering matches checkpoint
    if list(labels_gene_cols) != list(ckpt_gene_names):
        missing = [g for g in ckpt_gene_names if g not in set(labels_gene_cols)]
        if missing:
            raise ValueError(f"Labels file is missing genes present in checkpoint (examples: {missing[:10]})")
        df_tmp = pd.DataFrame(labels_values, index=labels_index, columns=labels_gene_cols)
        df_tmp = df_tmp.loc[:, ckpt_gene_names]
        labels_values = df_tmp.to_numpy(dtype=np.float32, copy=True)
        labels_gene_cols = list(ckpt_gene_names)
        LOGGER.warning("Reordered labels gene columns to match checkpoint gene_names")

    out_dim = len(labels_gene_cols)

    # Read settings from checkpoint
    views = ckpt_args.get("views", None)
    multi_view_mode = views is not None and isinstance(views, (list, tuple)) and len(views) > 0

    arch =  "dinov2_vitl14"
    pooling = "cls"
    fusion = "concat"
    share_backbone =  False
    dropout = 0.1
    attention_heads = 4
    attention_dropout = 0.1
    backbone_checkpoint = None
    h5_view = "morphology"
    slide_sep = '_'

    # device
    if device is not None:
        device_t = torch.device(device)
    else:
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loader params
    batch_size = int(batch_size)
    num_workers = int(num_workers)

    # --- Build dataset/loader + model ---
    if multi_view_mode:
        views = list(views)
        
        raw_sizes = detect_view_sizes(h5_path, views)
        patch_size = get_patch_size(arch)
        view_sizes = {v: round_to_patch_size(min(h, w), patch_size) for v, (h, w) in raw_sizes.items()}
        if share_backbone:
            shared = max(view_sizes.values())
            view_sizes = {v: shared for v in views}

        records, matched_ids = discover_h5_entries(h5_path, labels_index, view=views[0], slide_sep=slide_sep)

        if val_slides:
            val_set = set(val_slides)
            keep_ids, keep_data, keep_label = [], [], []
            for i, cid in enumerate(matched_ids):
                slide = cid.split(slide_sep, 1)[0] if slide_sep in cid else cid
                if slide in val_set:
                    keep_ids.append(cid)
                    keep_data.append(records.data_idx[i])
                    keep_label.append(records.label_idx[i])
            if len(keep_ids) == 0:
                raise RuntimeError(f"No matched records found for val_slides={val_slides}")
            matched_ids = keep_ids
            records = MatchedRecords(np.asarray(keep_data, dtype=np.int64), np.asarray(keep_label, dtype=np.int32))
            LOGGER.info("Filtered to slides=%s -> n=%d", val_slides, len(records))

        eval_tfs = {v: build_eval_transform(view_sizes[v]) for v in views}
        ds = MultiViewH5InferenceDataset(
            h5_path=h5_path,
            records=records,
            cell_ids=matched_ids,
            views=views,
            transforms_by_view=eval_tfs,
            slide_sep=slide_sep,
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=multiview_infer_collate_fn,
        )

        dummy = BACKBONES[arch](img_size=DINOV2_PRETRAINED_SIZE, pretrained=False)
        embed_dim = getattr(dummy, "embed_dim", getattr(dummy, "num_features"))
        del dummy

        model = MultiViewDinoRegressor(
            views=views,
            view_sizes=view_sizes,
            arch=arch,
            embed_dim=embed_dim,
            out_dim=out_dim,
            share_backbone=share_backbone,
            fusion=fusion,
            pooling=pooling,
            dropout=dropout,
            attention_heads=attention_heads,
            attention_dropout=attention_dropout,
            pretrained=True,
            backbone_checkpoint=backbone_checkpoint,
        )
    else:
        patch_size = get_patch_size(arch)
        view = h5_view

        records, matched_ids = discover_h5_entries(h5_path, labels_index, view=view, slide_sep=slide_sep)

        if val_slides:
            val_set = set(val_slides)
            keep_ids, keep_data, keep_label = [], [], []
            for i, cid in enumerate(matched_ids):
                slide = cid.split(slide_sep, 1)[0] if slide_sep in cid else cid
                if slide in val_set:
                    keep_ids.append(cid)
                    keep_data.append(records.data_idx[i])
                    keep_label.append(records.label_idx[i])
            if len(keep_ids) == 0:
                raise RuntimeError(f"No matched records found for val_slides={val_slides}")
            matched_ids = keep_ids
            records = MatchedRecords(np.asarray(keep_data, dtype=np.int64), np.asarray(keep_label, dtype=np.int32))
            LOGGER.info("Filtered to slides=%s -> n=%d", val_slides, len(records))

        image_size = None
        if image_size is None:
            raw_sizes = detect_view_sizes(h5_path, [view])
            h, w = raw_sizes[view]
            image_size = round_to_patch_size(min(h, w), patch_size)
            LOGGER.info("Auto-detected image_size=%d from view shape %dx%d", image_size, h, w)
        else:
            image_size = int(image_size)

        eval_tf = build_eval_transform(image_size)

        ds = H5InferenceDataset(
            h5_path=h5_path,
            records=records,
            cell_ids=matched_ids,
            transform=eval_tf,
            view=view,
            slide_sep=slide_sep,
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        LOGGER.info(f'1137: backbone_checkpoint: {backbone_checkpoint} !!')
        backbone = create_backbone_with_interpolation(
            arch=arch,
            img_size=image_size,
            pretrained=False,
            backbone_checkpoint=None,
        )
        embed_dim = getattr(backbone, "embed_dim", getattr(backbone, "num_features"))

        model = DinoRegressor(
            backbone=backbone,
            embed_dim=embed_dim,
            out_dim=out_dim,
            dropout=dropout,
            pooling=pooling,
            attention_heads=attention_heads,
            attention_dropout=attention_dropout,
        )
    if save_subset_h5 is not None:
        subset_views = views if multi_view_mode else [h5_view]
        save_h5_subset(h5_path, Path(save_subset_h5), records.data_idx, views=subset_views)

    model.load_state_dict(ckpt["model"], strict=True)
    freeze_all_params(model)
    model.to(device_t)

    LOGGER.info(
        "Loaded model from %s | mode=%s | n=%d | genes=%d | device=%s | bs=%d workers=%d",
        ckpt_path.name,
        "multiview" if multi_view_mode else "singleview",
        len(loader.dataset),
        out_dim,
        device_t,
        batch_size,
        num_workers,
    )

    # --- Inference (also return df to notebook) ---
    model.eval()
    preds_store = []
    embeds_store = [] if (output_embeds is not None or return_embeddings) else None
    ids: List[str] = []

    with torch.inference_mode():
        pbar = tqdm(loader, desc="Infer", leave=False)
        for images, cell_ids in pbar:
            images = _move_images_to_device(images, device_t)

            if fp16 and device_t.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    preds, feats = model(images)
            else:
                preds, feats = model(images)

            preds_store.append(preds.float().cpu())
            if embeds_store is not None:
                embeds_store.append(feats.float().cpu())
            ids.extend(cell_ids)

    pred_mat = torch.cat(preds_store, dim=0).numpy()
    df_pred = pd.DataFrame(pred_mat, index=pd.Index(ids, name="cell_uid"), columns=labels_gene_cols)

    out_path = Path(output_preds)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # parquet save (fallback to csv if parquet engine missing)
    try:
        df_pred.to_parquet(out_path)
        LOGGER.info("Wrote predictions parquet: %s | shape=%s", out_path, df_pred.shape)
    except Exception as e:
        fallback = out_path.with_suffix(".csv")
        df_pred.to_csv(fallback)
        LOGGER.warning("to_parquet failed (%s). Wrote CSV instead: %s", repr(e), fallback)

    df_emb = None
    if embeds_store is not None:
        feats = torch.cat(embeds_store, dim=0)
        if output_embeds is not None:
            emb_path = Path(output_embeds)
            emb_path.parent.mkdir(parents=True, exist_ok=True)
            if emb_path.suffix == ".pt":
                torch.save({"cell_ids": ids, "features": feats}, emb_path)
            else:
                df_emb = pd.DataFrame(feats.numpy(), index=pd.Index(ids, name="cell_uid"))
                try:
                    if emb_path.suffix == ".parquet":
                        df_emb.to_parquet(emb_path)
                    else:
                        df_emb.to_csv(emb_path)
                except Exception:
                    df_emb.to_csv(emb_path.with_suffix(".csv"))
            LOGGER.info("Wrote embeddings: %s | n=%d dim=%d", output_embeds, feats.shape[0], feats.shape[1])

        if return_embeddings and df_emb is None:
            # if user asked to return embeddings but didn't force a df format, just return tensor
            df_emb = feats

    return (df_pred, df_emb) if return_embeddings else df_pred


def boxplot_per_gene_pearson(
    model_to_pearson: Dict[str, pd.Series],
    *,
    gene_set: str = "intersection",
    dropna: bool = True,
    title: Optional[str] = "Per-gene Pearson across models",
    ylabel: str = "Pearson r (per gene)",
    show_points: bool = True,
    point_alpha: float = 0.35,
    rotate_xticks: int = 25,
    show_x_axis_label: bool = True,
    show_y_axis_label: bool = True,
    min_y: Optional[float] = None,
    max_y: Optional[float] = None,
    x_axis_label: str = "",
    ticks = False,
    box_colors: Optional[list] = None,     # 1. Hex color for each box (facecolor of boxes)
    box_alphas: Optional[list] = None,     # <--- Optional: Transparency per box (add param)
    point_colors: Optional[list] = None,   # 2. Hex color for each set of points
    box_edge_colors: Optional[list] = None,# 3. Hex color for edge of each box
    point_edge_colors: Optional[list] = None, # 4. Hex color for edge of each set of points
    point_size = 30
    ) -> pd.DataFrame:
    if not model_to_pearson:
        raise ValueError("model_to_pearson is empty.")

    # Ensure all are Series with string index
    cleaned = {}
    for name, s in model_to_pearson.items():
        if not isinstance(s, pd.Series):
            raise TypeError(f"{name} must be a pd.Series, got {type(s)}")
        ss = s.copy()
        ss.index = ss.index.astype(str)
        cleaned[name] = ss.astype(float)

    # Build aligned DataFrame
    if gene_set == "intersection":
        genes = None
        for s in cleaned.values():
            genes = s.index if genes is None else genes.intersection(s.index)
        genes = genes.sort_values()
        df = pd.DataFrame({name: s.reindex(genes) for name, s in cleaned.items()})
    elif gene_set == "union":
        genes = pd.Index([])
        for s in cleaned.values():
            genes = genes.union(s.index)
        genes = genes.sort_values()
        df = pd.DataFrame({name: s.reindex(genes) for name, s in cleaned.items()})
    else:
        raise ValueError("gene_set must be 'intersection' or 'union'.")

    # Prepare data for boxplot
    data = []
    labels = []
    for col in df.columns:
        vals = df[col].to_numpy()
        if dropna:
            vals = vals[~np.isnan(vals)]
        data.append(vals)
        labels.append(col)

    fig, ax = plt.subplots(figsize=(10, 10))
    # Remove outlier/fliers: showfliers=False
    bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)

    # Default color filling assignments if not provided
    num_boxes = len(data)
    def _expand_or_default(lst, default, n):
        if lst is None:
            return [default] * n
        elif len(lst) < n:
            return lst + [default] * (n - len(lst))
        return lst

    # Define default colors
    default_box_face = "#bbbbbb"
    default_box_edge = "#333333"
    default_box_alpha = 0.2  # transparent default
    default_point_color = "#3366cc"
    default_point_edge = "#222222"

    box_colors = _expand_or_default(box_colors, default_box_face, num_boxes)
    box_alphas = _expand_or_default(box_alphas, default_box_alpha, num_boxes)  # expand/assign alpha
    box_edge_colors = _expand_or_default(box_edge_colors, default_box_edge, num_boxes)
    point_colors = _expand_or_default(point_colors, default_point_color, num_boxes)
    point_edge_colors = _expand_or_default(point_edge_colors, default_point_edge, num_boxes)

    # Set hex colors for each box and edge, with transparency (alpha)
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(box_colors[i])
        box.set_edgecolor(box_edge_colors[i])
        box.set_alpha(box_alphas[i])  # This line sets the transparency of the boxes
        box.set_linewidth(2)
    for i, median in enumerate(bp['medians']):
        median.set_color(box_edge_colors[i])
        median.set_linewidth(2)
    for i, cap in enumerate(bp['caps']):
        cap.set_color(box_edge_colors[i//2])
        cap.set_linewidth(2)
    for i, whisker in enumerate(bp['whiskers']):
        whisker.set_color(box_edge_colors[i//2])
        whisker.set_linewidth(2)
    # No longer need to customize fliers, as they are not shown

    # Plot points *after* the boxplot so they appear on top
    if show_points:
        for i, vals in enumerate(data, start=1):
            x = np.random.normal(loc=i, scale=0.06, size=len(vals))
            colors = point_colors[i-1]
            edgecolors = point_edge_colors[i-1]
            # zorder=10 ensures these are drawn above the boxes
            ax.scatter(
                x, vals,
                alpha=point_alpha, s=point_size,
                color=colors,
                edgecolors=edgecolors,
                linewidths=1.2,
                zorder=10
            )

    if min_y is not None or max_y is not None:
        current_ylim = ax.get_ylim()
        y_min = min_y if min_y is not None else current_ylim[0]
        y_max = max_y if max_y is not None else current_ylim[1]
        ax.set_ylim(y_min, y_max)

    # Remove all y-ticks and y-tick labels
    if not ticks:
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticks([])
    # Remove all x-ticks (the numbers/positions) but keep the category labels if any
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
    # Remove x-tick labels if you want to hide the model names also:
    # ax.set_xticklabels([])
    ax.xaxis.set_tick_params(length=0)  # Remove tick lines

    # Remove the black boundary (spines) of the plot
    for spine in ax.spines.values():
        spine.set_visible(False)

    if show_y_axis_label and ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(None)
    if show_x_axis_label and x_axis_label:
        ax.set_xlabel(x_axis_label)
    else:
        ax.set_xlabel(None)
    if title:
        ax.set_title(title)
    if rotate_xticks:
        plt.setp(ax.get_xticklabels(), rotation=rotate_xticks, ha="right")

    plt.tight_layout()
    plt.show()
    return df

def compute_per_gene_pearson_and_loss(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    slide_id: Optional[str] = None,
    loss: str = "smoothl1",
    beta: float = 1.0,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Compute per-gene Pearson correlation and per-gene loss between GT and predictions.

    Args:
        gt, pred: DataFrames with index=cell_id and columns=genes (numeric).
        slide_id: If provided and pred indices don't start with '{slide_id}_', prepend it.
        loss: 'mse' or 'smoothl1'
        beta: SmoothL1 transition point (matches PyTorch SmoothL1Loss beta).

    Returns:
        per_gene_pearson: pd.Series indexed by gene
        per_gene_loss: pd.Series indexed by gene (mean over cells)
        gt_matched: pd.DataFrame with cells/genes used in metrics
    """
    if slide_id is not None and len(pred.index) > 0:
        sample_pred_idx = str(pred.index[0])
        if not sample_pred_idx.startswith(f"{slide_id}_"):
            pred = pred.copy()
            pred.index = [f"{slide_id}_{idx}" for idx in pred.index]
            print(f"Added slide prefix '{slide_id}_' to {len(pred)} predictions")

    common_cells = gt.index.intersection(pred.index)
    print(f"Common cells: {len(common_cells)}")

    if len(common_cells) == 0:
        print("WARNING: No common cells found!")
        print(f"Sample GT index: {list(gt.index[:3])}")
        print(f"Sample Pred index: {list(pred.index[:3])}")
        return pd.Series(dtype=float), pd.Series(dtype=float)
        return pd.Series(dtype=float), pd.Series(dtype=float), gt.loc[[]]
    common_genes = gt.columns.intersection(pred.columns)
    gt_matched = gt.loc[common_cells, common_genes].astype(float)
    pred_matched = pred.loc[common_cells, common_genes].astype(float)

    # Pearson per gene
    per_gene_pearson = gt_matched.corrwith(pred_matched, axis=0)

    # Loss per gene
    diff = gt_matched - pred_matched  # (cells, genes)

    loss = loss.lower()
    if loss == "mse":
        per_gene_loss = (diff ** 2).mean(axis=0)
    elif loss in ("smoothl1", "huber"):
        abs_diff = diff.abs()
        # SmoothL1 (PyTorch style): 0.5 * x^2 / beta if |x| < beta else |x| - 0.5*beta
        per_elem = np.where(
            abs_diff.to_numpy() < beta,
            0.5 * (diff.to_numpy() ** 2) / beta,
            abs_diff.to_numpy() - 0.5 * beta,
        )
        per_gene_loss = pd.Series(per_elem.mean(axis=0), index=common_genes, dtype=float)
    else:
        raise ValueError(f"Unknown loss='{loss}'. Use 'mse' or 'smoothl1'.")

    # Ensure Series index matches genes
    per_gene_loss = per_gene_loss.reindex(common_genes)

    # return per_gene_pearson, per_gene_loss
    return per_gene_pearson, per_gene_loss , gt_matched
def plot_gene_predictions(
    pred_df: pd.DataFrame,
    gene_name: str,
    location_df: pd.DataFrame,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    low_rgb=(255, 255, 255),
    high_rgb=(0, 0, 0),
    figsize=(8, 8),
    title_prefix="Predicted",
    show_axis: bool = True,
    show_title: bool = True,
    enlargement_factor: float = 2.0,    # New: control how much to enlarge cells
    prefix = 'NCBI856_'
):
    if "geom" not in location_df.columns:
        loc = location_df.copy()
        loc["geom"] = loc["geometry"].apply(wkb.loads)
    else:
        loc = location_df
    if gene_name not in pred_df.columns:
        raise ValueError(f"Gene '{gene_name}' not found in prediction DataFrame.")
    loc.index = prefix + loc.index.astype(str)
    merged = loc.join(pred_df[[gene_name]], how="inner")
    merged = merged.rename(columns={gene_name: "value"})
    print(f"Cells with both location and prediction: {len(merged)}")
    bbox_geom = box(x_min, y_min, x_max, y_max)
    mask = merged["geom"].apply(lambda g: g.intersects(bbox_geom))
    subset = merged[mask].copy()

    print(f"Cells inside box: {len(subset)}")
    if subset.empty:
        raise ValueError("No cells found inside the specified bounding box. Adjust x/y limits.")
    low_rgb_arr = np.array(low_rgb, dtype=float) / 255.0
    high_rgb_arr = np.array(high_rgb, dtype=float) / 255.0

    # Exp-transform the log predictions to get real-values for color mapping
    vals = subset["value"].to_numpy()
    # vals = np.exp(vals_log)
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    if vmax == vmin:
        norm_vals = np.zeros_like(vals)
    else:
        norm_vals = (vals - vmin) / (vmax - vmin)

    colors = low_rgb_arr + norm_vals[:, None] * (high_rgb_arr - low_rgb_arr)
    fig, ax = plt.subplots(figsize=figsize)

    for geom, color in zip(subset["geom"], colors):
        if geom.is_empty:
            continue
        # ENLARGE the geometry for better visibility at large scales
        try:
            # Support Polygon/MultiPolygon
            if geom.geom_type == "Polygon":
                geoms = [geom]
            else:
                geoms = list(geom.geoms)
            enlarged_geoms = []
            for g in geoms:
                if enlargement_factor != 1.0:
                    centroid = g.centroid
                    g_scaled = affinity.scale(g, xfact=enlargement_factor, yfact=enlargement_factor, origin=centroid)
                else:
                    g_scaled = g
                enlarged_geoms.append(g_scaled)
            for g in enlarged_geoms:
                xs, ys = g.exterior.xy
                ax.fill(xs, ys, color=color, edgecolor="none")
        except Exception as e:
            # If anything fails, just plot original shape
            try:
                xs, ys = geom.exterior.xy
                ax.fill(xs, ys, color=color, edgecolor="none")
            except Exception:
                continue

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")

    if not show_axis:
        ax.set_axis_off()
    if show_title:
        ax.set_title(f"{title_prefix} {gene_name} (darker = higher, exp-transformed)")
    plt.show()
    return subset


    try:
        from scipy.spatial import cKDTree
        from scipy import sparse
    except Exception as exc:
        raise ImportError("scipy is required for Moran's I/spatial correlation.") from exc

    xy = coords[["x", "y"]].to_numpy()
    n = xy.shape[0]
    if n <= k:
        raise ValueError(f"Need n > k (got n={n}, k={k}).")

    tree = cKDTree(xy)
    dists, idxs = tree.query(xy, k=k + 1)
    dists = dists[:, 1:]
    idxs = idxs[:, 1:]

    rows = np.repeat(np.arange(n), k)
    cols = idxs.reshape(-1)
    if weight == "binary":
        data = np.ones_like(cols, dtype=float)
    elif weight == "inverse_distance":
        data = 1.0 / (dists.reshape(-1) + 1e-12)
    else:
        raise ValueError("weight must be 'binary' or 'inverse_distance'.")

    W = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    if row_normalize:
        row_sums = np.asarray(W.sum(axis=1)).reshape(-1)
        row_sums[row_sums == 0.0] = 1.0
        W = sparse.diags(1.0 / row_sums).dot(W)
    return W

import matplotlib.pyplot as plt
import numpy as np

def plot_per_gene_pearson(
    per_gene_pearson_dict,
    sort_by=None,
    bar_methods=None,
    bar_colors='tab:blue',
    bar_alpha=0.8,
    bar_edge_color=None,
    bar_edge_alpha=1.0,
    line_colors=None,
    bar_width=1.0,   # user-specified width of bars (default 1.0)
    bar_spacing='auto',  # new: allow auto/bar_width-based spacing, or user can specify float
    figsize=(15,6),
    xlabel='Gene',
    ylabel='Value',
    title='Pearson per gene for each method',
    hide_plot_elements=False,
):
    if not per_gene_pearson_dict:
        raise ValueError("per_gene_pearson_dict must not be empty")

    if sort_by is None:
        sort_by = next(iter(per_gene_pearson_dict))
    if sort_by not in per_gene_pearson_dict:
        raise ValueError(f"{sort_by} is not a key in per_gene_pearson_dict")
    if bar_methods is None:
        bar_methods = []
    elif isinstance(bar_methods, str):
        bar_methods = [bar_methods]
    bar_methods = [m for m in bar_methods if m in per_gene_pearson_dict]

    sorted_genes = per_gene_pearson_dict[sort_by].sort_values(ascending=False).index
    n_genes = len(sorted_genes)
    methods = list(per_gene_pearson_dict.keys())
    non_bar_methods = [m for m in methods if m not in bar_methods]

    fig, ax = plt.subplots(figsize=figsize)

    handles = []
    labels = []

    # ---- Instead of x = np.arange(n_genes), space bars to avoid overlap if bar_width > 1
    if bar_spacing == 'auto' or bar_spacing is None:
        bar_spacing_val = bar_width * 1.05  # add 5% gap so bars never touch/overlap
    else:
        bar_spacing_val = float(bar_spacing)
    x = np.arange(n_genes) * bar_spacing_val

    if len(bar_methods) == 1:
        method = bar_methods[0]
        y = per_gene_pearson_dict[method].loc[sorted_genes].values
        # Handle color(s)
        if isinstance(bar_colors, dict) and method in bar_colors:
            bar_c = bar_colors[method]
        else:
            bar_c = bar_colors
        if bar_edge_color is None:
            edge_c = bar_c
        elif isinstance(bar_edge_color, dict) and method in bar_edge_color:
            edge_c = bar_edge_color[method]
        else:
            edge_c = bar_edge_color
        # edge alpha
        if isinstance(edge_c, str):
            rgba = list(plt.matplotlib.colors.to_rgba(edge_c))
            rgba[3] = bar_edge_alpha
            edge_c_rgba = tuple(rgba)
        elif isinstance(edge_c, tuple) and len(edge_c) in (3, 4):
            rgba = list(edge_c) + [1.0]*(4-len(edge_c))
            rgba[3] = bar_edge_alpha
            edge_c_rgba = tuple(rgba)
        else:
            edge_c_rgba = edge_c
        bar = ax.bar(
            x, y,
            width=bar_width,
            align='center',
            label=method,
            color=bar_c, alpha=bar_alpha,
            edgecolor=edge_c_rgba, linewidth=1.2
        )
        handles.append(bar)
        labels.append(method)

    for method in non_bar_methods:
        y = per_gene_pearson_dict[method].loc[sorted_genes].values
        line_c = None
        if isinstance(line_colors, dict) and method in line_colors:
            line_c = line_colors[method]
        (line,) = ax.plot(
            x, y, marker='o', label=method, color=line_c)
        handles.append(line)
        labels.append(method)

    if not hide_plot_elements:
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_genes, rotation=90)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title + f" (sorted by {sort_by})")
        ax.legend(handles=handles, labels=labels)
        plt.tight_layout()
        # Remove whitespace on left and right
        if n_genes > 0:
            min_x = x[0] - bar_width / 2
            max_x = x[-1] + bar_width / 2
            ax.set_xlim(min_x, max_x)
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        legend = ax.get_legend()
        if legend is not None:
            legend.set_visible(False)
        # Remove whitespace on left and right
        if n_genes > 0:
            min_x = x[0] - bar_width / 2
            max_x = x[-1] + bar_width / 2
            ax.set_xlim(min_x, max_x)
    plt.show()

# # Example usage:
# plot_per_gene_pearson(
#     {
#         "Patch": per_gene_pearson_patch.nlargest(ntop),
#         "Cell": per_gene_pearson_cell.nlargest(ntop),
#         "MAD": per_gene_pearson_mad.nlargest(ntop),
#     },
#     sort_by="MAD",
#     bar_methods=["MAD"],   # only one bar per gene, with no space in between
#     bar_colors={"MAD": "#fb8072"},
#     bar_alpha=0.3,
#     bar_edge_color={"MAD": "#fb8072"},
#     bar_edge_alpha=1.,
#     line_colors={
#         "Cell": "#ffd92f",
#         "CellDINO": "#b3de69",
#         "Patch": "#8dd3c7",
#         "MAD": "#fb8072",
#         "UNI": "#80b1d3",
#     },
#     hide_plot_elements=False,
#     bar_width=4,
#     figsize=(20,6)
#     # You can also specify bar_spacing=2.0 here if you want, or leave as default ('auto'), which gives ~5% gap.
# )

if __name__ == "__main__":
    main()

