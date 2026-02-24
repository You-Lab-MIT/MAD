import io
import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import h5py
import numpy as np
from PIL import Image, UnidentifiedImageError
from torchvision.datasets import VisionDataset
from torch.utils.data import get_worker_info


logger = logging.getLogger("dinov2")


class MorphNeighborhoodDataset(VisionDataset):
    """
    Dataset that pairs morphology (segmented single-cell) and neighborhood (context) images.

    The expected directory structure is:
        root/
            morphology/
                <group_id>/
                    <image_id>.png
            neighborhood/
                <group_id>/
                    <image_id>.png

    Within each pair of group directories, file names must match exactly so that samples
    can be aligned between the morphology and neighborhood views.
    """

    def __init__(
        self,
        *,
        root: str,
        morphology_dir: str = "morphology",
        neighborhood_dir: str = "neighborhood",
        extensions: Optional[Sequence[str]] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        self._morph_root = Path(self.root) / morphology_dir
        self._neighborhood_root = Path(self.root) / neighborhood_dir
        self._extensions = tuple(ext.lower() for ext in (extensions or (".png", ".jpg", ".jpeg", ".tif", ".tiff")))

        if not self._morph_root.is_dir():
            raise FileNotFoundError(f'Morphology directory "{self._morph_root}" was not found')
        if not self._neighborhood_root.is_dir():
            raise FileNotFoundError(f'Neighborhood directory "{self._neighborhood_root}" was not found')

        self._samples: List[Tuple[Path, Path]] = self._collect_pairs()
        self._bad_indices: Set[int] = set()
        logger.info(f"Loaded {len(self._samples):,d} paired morphology/neighborhood tiles from {self.root}")

    def _is_valid_file(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in self._extensions

    def _collect_pairs(self) -> List[Tuple[Path, Path]]:
        samples: List[Tuple[Path, Path]] = []
        missing_counter = 0

        morph_files: Iterable[Path] = sorted(self._morph_root.rglob("*"))
        for morph_path in morph_files:
            if not self._is_valid_file(morph_path):
                continue

            rel_path = morph_path.relative_to(self._morph_root)
            neighborhood_path = self._neighborhood_root / rel_path
            if not self._is_valid_file(neighborhood_path):
                missing_counter += 1
                logger.warning(
                    'Missing neighborhood counterpart for "%s"; expected file "%s"',
                    morph_path,
                    neighborhood_path,
                )
                continue

            samples.append((morph_path, neighborhood_path))

        if not samples:
            raise RuntimeError(
                f"No paired samples were found under {self._morph_root} and {self._neighborhood_root}. "
                "Check that file names match across the two folders."
            )
        if missing_counter > 0:
            logger.warning("Skipped %d samples without a matching neighborhood tile", missing_counter)

        return samples

    def __len__(self) -> int:
        return len(self._samples)

    @staticmethod
    def _load_image(path: Path) -> Image.Image:
        with path.open("rb") as f:
            image = Image.open(f)
            return image.convert("RGB")

    def __getitem__(self, index: int):
        dataset_len = len(self._samples)
        index = index % dataset_len
        attempts = 0

        while attempts < dataset_len:
            if index in self._bad_indices:
                index = (index + 1) % dataset_len
                attempts += 1
                continue

            morph_path, neighborhood_path = self._samples[index]
            try:
                morphology_image = self._load_image(morph_path)
                neighborhood_image = self._load_image(neighborhood_path)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                self._bad_indices.add(index)
                logger.warning(
                    'Skipping unreadable sample "%s" or "%s" (error: %s)',
                    morph_path,
                    neighborhood_path,
                    exc,
                )
                index = (index + 1) % dataset_len
                attempts += 1
                continue
            break
        else:
            raise RuntimeError("All samples appear to be unreadable.")

        morph_path, neighborhood_path = self._samples[index]

        sample = {
            "morphology": morphology_image,
            "neighborhood": neighborhood_image,
            "relative_path": str(morph_path.relative_to(self._morph_root)),
        }

        target = sample["relative_path"]
        if self.transforms is not None:
            sample, target = self.transforms(sample, target)

        return sample, target


class MorphNeighborhoodH5Dataset(VisionDataset):
    """
    Dataset variant that reads morphology/neighborhood image pairs from HDF5 files.

    Supports two modes:

    1. Single-file mode (h5_file specified): One HDF5 file with top-level keys
       ``morphology`` and ``neighborhood``, each storing (N, H, W, 3) uint8 arrays.
       Example: MorphNeighborhoodH5:root=/path/to/dir:h5_file=ALL_NCBI_combined_with_niche.h5

    2. Multi-file mode (legacy): root/morphology/*.h5 and root/neighborhood/*.h5.
       Each HDF5 file must expose datasets named ``filenames`` and ``images`` (PNG bytes).
    """

    def __init__(
        self,
        *,
        root: str,
        h5_file: Optional[str] = None,
        morphology_dir: str = "morphology",
        neighborhood_dir: str = "neighborhood",
        morphology_key: str = "morphology",
        neighborhood_key: str = "neighborhood",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        self._single_file_mode = h5_file is not None
        self._morphology_key = morphology_key
        self._neighborhood_key = neighborhood_key

        if self._single_file_mode:
            self._init_single_file(Path(self.root) / h5_file)
        else:
            self._init_multi_file(morphology_dir, neighborhood_dir)

    def _init_single_file(self, h5_path: Path) -> None:
        """Initialize from a single H5 file with morphology/neighborhood keys."""
        if not h5_path.exists():
            raise FileNotFoundError(f'HDF5 file "{h5_path}" was not found')

        with h5py.File(h5_path, "r") as f:
            for key in (self._morphology_key, self._neighborhood_key):
                if key not in f:
                    raise KeyError(
                        f'HDF5 file must contain "{key}" dataset. Found keys: {list(f.keys())}'
                    )
            morph_len = len(f[self._morphology_key])
            neigh_len = len(f[self._neighborhood_key])
            if morph_len != neigh_len:
                raise ValueError(
                    f'Mismatched lengths: morphology={morph_len}, neighborhood={neigh_len}'
                )

        self._h5_path = h5_path
        self._length = morph_len
        self._files = []  # unused in single-file mode
        self._worker_file_handles: Dict[int, Optional[Dict[str, object]]] = {}
        self._bad_indices: Set[int] = set()
        logger.info(
            "Loaded %d samples from single HDF5 file %s",
            self._length,
            self._h5_path.name,
        )

    def _init_multi_file(self, morphology_dir: str, neighborhood_dir: str) -> None:
        """Initialize from morphology/*.h5 and neighborhood/*.h5 directories."""
        self._morph_root = Path(self.root) / morphology_dir
        self._neigh_root = Path(self.root) / neighborhood_dir

        if not self._morph_root.is_dir():
            raise FileNotFoundError(f'Morphology directory "{self._morph_root}" was not found')
        if not self._neigh_root.is_dir():
            raise FileNotFoundError(f'Neighborhood directory "{self._neigh_root}" was not found')

        self._files = []
        cumulative = 0
        skipped_bad = 0
        skipped_missing = 0
        for morph_file in sorted(self._morph_root.glob("*.h5")):
            neigh_file = self._neigh_root / morph_file.name
            if not neigh_file.exists():
                skipped_missing += 1
                logger.warning(
                    'Neighborhood HDF5 file "%s" not found for morphology file "%s"; skipping shard.',
                    neigh_file,
                    morph_file,
                )
                continue

            try:
                length = self._validate_file_pair(morph_file, neigh_file)
            except Exception as exc:
                skipped_bad += 1
                logger.warning(
                    'Skipping shard "%s" due to unreadable HDF5 pair (%s).',
                    morph_file.stem,
                    exc,
                )
                continue

            self._files.append(
                {
                    "morph_path": morph_file,
                    "neigh_path": neigh_file,
                    "start": cumulative,
                    "end": cumulative + length,
                    "group": morph_file.stem,
                }
            )
            cumulative += length

        if cumulative == 0:
            raise RuntimeError(f"No samples discovered under {self._morph_root} and {self._neigh_root}")

        if skipped_missing > 0:
            logger.warning("Skipped %d HDF5 shard(s) without a matching neighborhood file.", skipped_missing)
        if skipped_bad > 0:
            logger.warning("Skipped %d HDF5 shard(s) that could not be opened or validated.", skipped_bad)

        self._length = cumulative
        self._worker_file_handles = {}
        logger.info("Loaded %d samples from %d HDF5 shards located in %s", self._length, len(self._files), self.root)
        self._bad_indices = {
            "morph": defaultdict(set),
            "neigh": defaultdict(set),
        }

    @staticmethod
    def _validate_file_pair(morph_file: Path, neigh_file: Path) -> int:
        with h5py.File(morph_file, "r") as morph_h5, h5py.File(
            neigh_file, "r"
        ) as neigh_h5:
            for name in ("filenames", "images"):
                if name not in morph_h5 or name not in neigh_h5:
                    raise KeyError(f'HDF5 files must contain datasets named "filenames" and "images"; missing "{name}"')
            morph_len = len(morph_h5["images"])
            neigh_len = len(neigh_h5["images"])
            if morph_len != neigh_len:
                raise ValueError(
                    f'Mismatched dataset lengths between "{morph_file}" ({morph_len}) and "{neigh_file}" ({neigh_len})'
                )
            return morph_len

    def __len__(self) -> int:
        return self._length

    def _find_file_index(self, index: int) -> int:
        left, right = 0, len(self._files)
        while left < right:
            mid = (left + right) // 2
            if index < self._files[mid]["start"]:
                right = mid
            elif index >= self._files[mid]["end"]:
                left = mid + 1
            else:
                return mid
        raise IndexError(f"Index {index} out of range for dataset of length {self._length}")

    def _get_worker_state(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else -1
        state = self._worker_file_handles.get(worker_id)
        if state is None:
            if self._single_file_mode:
                state = None  # will be populated by _ensure_single_file_handle
            else:
                state = {"morph": {}, "neigh": {}}
            self._worker_file_handles[worker_id] = state
        return state

    def _ensure_single_file_handle(self) -> Dict[str, object]:
        """Get or create per-worker handle for single-file mode."""
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else -1
        state = self._worker_file_handles.get(worker_id)
        if state is None:
            file_handle = h5py.File(self._h5_path, "r")
            state = {
                "file": file_handle,
                "morphology": file_handle[self._morphology_key],
                "neighborhood": file_handle[self._neighborhood_key],
            }
            if "cell_ids" in file_handle:
                state["cell_ids"] = file_handle["cell_ids"]
            else:
                state["cell_ids"] = None
            self._worker_file_handles[worker_id] = state
        return state

    @staticmethod
    def _bytes_to_image(raw: bytes) -> Image.Image:
        with Image.open(io.BytesIO(raw)) as img:
            return img.convert("RGB")

    @staticmethod
    def _array_to_image(arr) -> Image.Image:
        """Convert (H, W, 3) uint8 array to PIL Image."""
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            return Image.fromarray(arr).convert("RGB")
        raise ValueError(f"Expected (H, W, 3) array, got shape {arr.shape}")

    def _ensure_handle(self, view: str, path: Path):
        state = self._get_worker_state()
        if path not in state[view]:
            file_handle = h5py.File(path, "r")
            state[view][path] = {
                "file": file_handle,
                "images": file_handle["images"],
                "filenames": file_handle["filenames"],
            }
        return state[view][path]

    def _read_entry(self, view: str, path: Path, index: int):
        handle = self._ensure_handle(view, path)
        image_array = handle["images"][index]
        raw_bytes = image_array.tobytes() if hasattr(image_array, "tobytes") else bytes(image_array)
        filename_raw = handle["filenames"][index]
        filename = filename_raw.decode("utf-8") if isinstance(filename_raw, (bytes, bytearray)) else str(filename_raw)
        try:
            image = self._bytes_to_image(raw_bytes)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise _UnreadableSample(
                view=view,
                shard=str(path),
                local_index=index,
                filename=filename,
                error=str(exc),
            ) from exc
        return image, filename

    def _read_single_file_entry(self, index: int):
        """Read morphology and neighborhood from single H5 file."""
        handle = self._ensure_single_file_handle()
        morph_arr = handle["morphology"][index]
        neigh_arr = handle["neighborhood"][index]
        morphology_image = self._array_to_image(morph_arr)
        neighborhood_image = self._array_to_image(neigh_arr)
        if handle["cell_ids"] is not None:
            cell_id = handle["cell_ids"][index]
            if isinstance(cell_id, (bytes, bytearray)):
                cell_id = cell_id.decode("utf-8")
            else:
                cell_id = str(cell_id)
            relative_path = f"{index:08d}_{cell_id}"
        else:
            relative_path = f"{index:08d}"
        return morphology_image, neighborhood_image, relative_path

    def __getitem__(self, index: int):
        if index < 0:
            index = self._length + index

        index = index % self._length
        attempts = 0

        if self._single_file_mode:
            while attempts < self._length:
                if index in self._bad_indices:
                    index = (index + 1) % self._length
                    attempts += 1
                    continue
                try:
                    morphology_image, neighborhood_image, relative_path = self._read_single_file_entry(index)
                    break
                except (ValueError, OSError, KeyError) as exc:
                    self._bad_indices.add(index)
                    logger.warning(
                        "Skipping unreadable sample at index %d (error: %s)",
                        index,
                        exc,
                    )
                    index = (index + 1) % self._length
                    attempts += 1
                    continue
            else:
                raise RuntimeError("All samples appear to be unreadable.")
        else:
            while attempts < self._length:
                file_idx = self._find_file_index(index)
                file_info = self._files[file_idx]
                local_index = index - file_info["start"]

                morph_key = str(file_info["morph_path"])
                neigh_key = str(file_info["neigh_path"])

                if (
                    local_index in self._bad_indices["morph"][morph_key]
                    or local_index in self._bad_indices["neigh"][neigh_key]
                ):
                    index = (index + 1) % self._length
                    attempts += 1
                    continue

                try:
                    morphology_image, filename = self._read_entry("morph", file_info["morph_path"], local_index)
                    neighborhood_image, _ = self._read_entry("neigh", file_info["neigh_path"], local_index)
                    relative_path = f'{file_info["group"]}/{filename}'
                    break
                except _UnreadableSample as exc:
                    self._bad_indices[exc.view][exc.shard].add(exc.local_index)
                    logger.warning(
                        'Skipping unreadable %s sample "%s/%s" (error: %s)',
                        exc.view,
                        Path(exc.shard).name,
                        exc.filename,
                        exc.error,
                    )
                    index = (index + 1) % self._length
                    attempts += 1
                    continue
            else:
                raise RuntimeError("All samples appear to be unreadable.")

        sample = {
            "morphology": morphology_image,
            "neighborhood": neighborhood_image,
            "relative_path": relative_path,
        }
        target = sample["relative_path"]
        if self.transforms is not None:
            sample, target = self.transforms(sample, target)
        return sample, target

    def close(self):
        for worker_id, state in list(self._worker_file_handles.items()):
            if state is None:
                continue
            if self._single_file_mode:
                state["file"].close()
                self._worker_file_handles[worker_id] = None
            else:
                for view_handles in state.values():
                    for entry in view_handles.values():
                        entry["file"].close()
                    view_handles.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class _UnreadableSample(Exception):
    def __init__(self, *, view: str, shard: str, local_index: int, filename: str, error: str):
        super().__init__(error)
        self.view = view
        self.shard = shard
        self.local_index = local_index
        self.filename = filename
        self.error = error

