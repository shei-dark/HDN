import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from glob import glob
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import random, itertools
from collections import Counter
from typing import Dict, List, Tuple, Iterable, Any


class SemisupervisedDataset(Dataset):
    def __init__(
        self,
        images,
        labels,
        patch_size=64,
        label_size=1,
        mode="semisupervised",
        n_classes=4,
        ignore_lbl=-1,
        ratio=0.75,
        indices_dict=None,
        radius=5,
    ):
        self.patch_size = patch_size
        self.label_size = label_size
        self.half = patch_size // 2 - self.label_size
        self.images = images
        self.labels = labels
        self.ignore_lbl = ignore_lbl
        self.n_classes = n_classes
        self.ratio = ratio
        self.mode = mode
        self.indices_dict = indices_dict or {}
        self.radius = radius
        self.seed = 42
        self.rng = random.Random(self.seed)
        self.samples_per_class: Dict[int, int] = {1: 2}
        self.default_samples_per_class: int = 1
        self.groups = self._prepare_metadata()
        self.n_label_per_class = {
            c: len([g for g in self.groups if g["labels"][0] == c])
            for c in range(self.n_classes)
        }
        self.anchor_indices_by_label = {
            c: [i for i, g in enumerate(self.groups) if g["labels"][0] == c]
            for c in range(self.n_classes)
        }

    def set_mode(self, mode: str):
        """Switch between supervised and semisupervised modes."""
        if mode not in ("supervised", "semisupervised"):
            raise ValueError("stage must be 'supervised' or 'semisupervised'")
        self.mode = mode
        
    def increase_radius(self):
        """Increase the radius for neighbor sampling."""
        self.radius += 1
        self.groups = self._modify_metadata()

    def _is_valid_coord(self, name, z, y, x, H, W):
        valid = (
            self.half <= y < H - self.half - 1 and self.half <= x < W - self.half - 1
        )
        in_cell = self.labels[name][z, y, x] != self.ignore_lbl
        return valid and in_cell

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        g = self.groups[idx]
        name, z = g["name"], int(g["z"])
        img_vol = self.images[name]
        lbl_vol = self.labels[name]

        def patch_at(y, x):
            p = img_vol[
                z,
                y - self.half : y + self.half + 2,
                x - self.half : x + self.half + 2,
            ]
            return torch.from_numpy(p).unsqueeze(0)  # [1, H, W]

        def lbl_at(y, x):
            p = lbl_vol[
                z,
                y - self.half : y + self.half + 2,
                x - self.half : x + self.half + 2,
            ]
            return torch.from_numpy(p).unsqueeze(0)

        if self.mode == "supervised":
            cy, cx = map(int, g["coords"][0])
            patch = patch_at(cy, cx).unsqueeze(0)  # [1, 1, H, W]  <-- extra dim
            label = torch.tensor([int(g["labels"][0])], dtype=torch.long)  # [1]
            segment = lbl_at(cy, cx).unsqueeze(0)  # [1, 1, H, W]
            return patch, label, segment
        else:
            coords = [tuple(map(int, xy)) for xy in g["coords"]]
            patches = torch.stack([patch_at(y, x) for (y, x) in coords])  # [4, 1, H, W]
            # labels = torch.tensor([g["labels"][0], -1, -1, -1], dtype=torch.long)  # [4]
            labels = torch.tensor([g["labels"][0], g["labels"][0], g["labels"][0], g["labels"][0]], dtype=torch.long)
            # labels = torch.tensor(g["labels"], dtype=torch.long)
            segments = torch.stack([lbl_at(y, x) for (y, x) in coords])  # [4, 1, H, W]
            return patches, labels, segments

    def _prepare_metadata(self) -> List[dict]:
        groups: List[dict] = []

        for name, z_list in self.indices_dict.items():
            img = self.images[name]
            lbl = self.labels[name]
            _, H, W = img.shape

            for z in z_list:
                used_coords = set()
                stack = lbl[z]

                for c in range(self.n_classes):
                    for cy, cx in self._sample_coords_for_class(stack, c):
                        if not self._is_valid_coord(name, z, cy, cx, H, W):
                            continue
                        if (cy, cx) in used_coords:
                            continue

                        used_coords.add((cy, cx))
                        neighbors = self._sample_neighbors(
                            name=name,
                            z=z,
                            cy=cy,
                            cx=cx,
                            H=H,
                            W=W,
                            used_coords=used_coords,
                            lbl=lbl,
                            k=3,
                            max_tries=100,
                        )

                        if len(neighbors) == 3:
                            groups.append(
                                self._make_group_record(
                                    name=name,
                                    z=z,
                                    cy=cy,
                                    cx=cx,
                                    c=c,
                                    neighbors=neighbors,
                                )
                            )

        self._report_class_counts(groups)
        return groups

    def _modify_metadata(self) -> List[dict]:
        """Recompute metadata after changing radius."""

        for g in self.groups:
            name, z = g["name"], int(g["z"])
            img = self.images[name]
            lbl = self.labels[name]
            _, H, W = img.shape

            used_coords = set()
            cy, cx = g["coords"][0]

            used_coords.add((cy, cx))
            neighbors = self._sample_neighbors(
                name=name,
                z=z,
                cy=cy,
                cx=cx,
                H=H,
                W=W,
                used_coords=used_coords,
                lbl=lbl,
                k=3,
                max_tries=100,
            )

            if len(neighbors) == 3:
                
                modified_group = self._make_group_record(
                    name=name,
                    z=z,
                    cy=cy,
                    cx=cx,
                    c=g["labels"][0],
                    neighbors=neighbors,
                )
                g["coords"] = modified_group["coords"]
                g["labels"] = modified_group["labels"]
                

        self._report_class_counts(self.groups)
        return self.groups

    def _sample_coords_for_class(
        self, stack: np.ndarray, c: int
    ) -> Iterable[Tuple[int, int]]:
        """Return up to N (y, x) coordinates for class c from a 2D label stack."""

        n_needed = getattr(self, "samples_per_class", {}).get(
            c, getattr(self, "default_samples_per_class", 1)
        )

        label_coords = np.argwhere(stack == c)
        if len(label_coords) < n_needed:
            return []  # not enough to sample

        idx = self.rng.sample(range(len(label_coords)), n_needed)
        sampled = label_coords[idx]
        return [(int(y), int(x)) for (y, x) in sampled]

    def _sample_neighbors(
        self,
        name: str,
        z: int,
        cy: int,
        cx: int,
        H: int,
        W: int,
        used_coords: set,
        lbl: np.ndarray,
        k: int = 3,
        max_tries: int = 100,
    ) -> List[Dict[str, int]]:
        """Randomly sample up to k valid nearby coordinates within a disk (radius=self.radius)."""
        neighbors: List[Dict[str, int]] = []
        tries = 0

        while len(neighbors) < k and tries < max_tries:
            dy = self.rng.randint(-self.radius, self.radius)
            dx = self.rng.randint(-self.radius, self.radius)

            # reject outside disk or center itself
            if dx * dx + dy * dy > self.radius * self.radius or (dx == 0 and dy == 0):
                tries += 1
                continue

            ny, nx = cy + dy, cx + dx
            coord = (ny, nx)

            if coord in used_coords:
                tries += 1
                continue

            if self._is_valid_coord(name, z, ny, nx, H, W):
                used_coords.add(coord)
                neighbors.append(
                    {
                        "y": int(ny),
                        "x": int(nx),
                        "label": int(lbl[z, ny, nx].item()),
                    }
                )

            tries += 1

        return neighbors

    def _make_group_record(
        self,
        name: str,
        z: int,
        cy: int,
        cx: int,
        c: int,
        neighbors: List[Dict[str, int]],
    ) -> Dict[str, Any]:
        """Create the output dict for one (center + neighbors) group."""
        return {
            "name": name,
            "z": int(z),
            "coords": [(int(cy), int(cx))] + [(n["y"], n["x"]) for n in neighbors],
            "labels": [int(c)] + [n["label"] for n in neighbors],
        }

    def _report_class_counts(self, groups: List[dict]) -> None:
        """Print class counts for centers and neighbors separately."""
        centers = [g["labels"][0] for g in groups]
        neighbors = [lab for g in groups for lab in g["labels"][1:]]

        for title, labs in (("anchors", centers), ("neighbors", neighbors)):
            counts = Counter(labs)
            for k in sorted(counts):
                print(f"  Class {k} ({title}): {counts[k]} samples")


class ModeAwareBalancedAnchorBatchSampler(Sampler):
    """
    Yields balanced batches of anchor indices.
    Adapts to dataset.mode at the start of every epoch.
    - total_patches_per_batch is in *patch units* (e.g., 32).
    - Supervised: 1 patch per anchor
    - Semisupervised: 4 patches per anchor
    """

    def __init__(self, dataset, total_patches_per_batch=32, seed=42, shuffle=True):
        self.dataset = dataset
        self.total_patches_per_batch = total_patches_per_batch
        self.rng = random.Random(seed)
        self.shuffle = shuffle

        # Build per-class pools once (anchors only)
        self.pools = {
            c: [i for i, g in enumerate(dataset.groups) if g["labels"][0] == c]
            for c in range(dataset.n_classes)
        }
        self.labels = [c for c, v in self.pools.items() if len(v) > 0]
        if not self.labels:
            raise ValueError("No anchors available in any class.")

        # cycling iterators for oversampling
        self._iters = None
        self._len_cached = None

    def _reset_iters(self):
        self._iters = {}
        for c in self.labels:
            pool = list(self.pools[c])
            if self.shuffle:
                self.rng.shuffle(pool)
            self._iters[c] = itertools.cycle(pool)

    def _compute_epoch_plan(self):
        # anchors-per-batch depends on current mode
        if self.dataset.mode == "semisupervised":
            assert (
                self.total_patches_per_batch % 4 == 0
            ), "total_patches_per_batch must be divisible by 4 in semisupervised mode."
            anchors_per_batch = self.total_patches_per_batch // 4
        else:
            anchors_per_batch = self.total_patches_per_batch

        # split anchors-per-batch across labels (balanced, round-robin remainder)
        base = anchors_per_batch // len(self.labels)
        rem = anchors_per_batch % len(self.labels)
        per_label_counts = {c: base for c in self.labels}
        for c in self.labels[:rem]:
            per_label_counts[c] += 1

        # epoch length heuristic: sized to the largest class before a full cycle
        max_class = max(len(self.pools[c]) for c in self.labels)
        num_batches = max(1, (max_class * len(self.labels)) // anchors_per_batch)

        return anchors_per_batch, per_label_counts, num_batches

    def __iter__(self):
        self._reset_iters()
        anchors_per_batch, per_label_counts, num_batches = self._compute_epoch_plan()
        self._len_cached = num_batches

        label_order = list(self.labels)
        if self.shuffle:
            self.rng.shuffle(label_order)

        for _ in range(num_batches):
            batch = []
            for c in label_order:
                take = per_label_counts[c]
                batch.extend(next(self._iters[c]) for _ in range(take))
            if self.shuffle:
                self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        # compute against current mode so progress bars don't go crazy after mode flip
        anchors_per_batch, _, num_batches = self._compute_epoch_plan()
        return num_batches


def flex_collate(batch):
    # batch = list of (patches[M,1,H,W], labels[M], segs[M,1,H,W])
    patches = torch.cat([b[0] for b in batch], dim=0)   # [sum M, 1, H, W]
    labels  = torch.cat([b[1] for b in batch], dim=0)   # [sum M]
    segs    = torch.cat([b[2] for b in batch], dim=0)   # [sum M, 1, H, W]
    return patches, labels, segs
class Custom2DDataset(Dataset):
    def __init__(
        self,
        images,
        labels,
        patch_size=64,
        label_size=5,
        mode="supervised",  # Options: 'supervised', 'semisupervised', 'unsupervised'
        n_classes=4,
        sampling_ratio=1,
        ignore_lbl=-1,
        ratio=0.75,
    ):
        self.patch_size = patch_size
        self.label_size = label_size
        self.images = images
        self.labels = labels
        self.ignore_lbl = ignore_lbl
        if isinstance(images, dict):
            self.keys = list(images.keys())
        else:
            self.keys = None
        self.n_classes = n_classes
        self.sampling_ratio = sampling_ratio
        self.all_patches, self.patches_by_label = (
            self._compute_valid_patches()
        )  # Store only metadata of valid patches
        self.mode = mode
        self.ratio = ratio

    def set_mode(self, mode):
        """Set the current mode of the dataset."""
        self.mode = mode

    def _centre_consistent(self, patch_metadata):
        """Vectorized version to check if the center is label-consistent."""
        if self.keys:
            key, z, x, y = patch_metadata
            unique_label_area = self.labels[key][
                z, x : x + self.label_size, y : y + self.label_size
            ]
        else:
            z, x, y = patch_metadata
            unique_label_area = self.labels[
                z, x : x + self.label_size, y : y + self.label_size
            ]
        return (
            np.all(unique_label_area == unique_label_area[0, 0])
            and unique_label_area[0, 0] != self.ignore_lbl
        )

    def _compute_valid_patches(self):
        """Fast vectorized patch extraction."""
        all_patches = []
        patches_by_label = {c: [] for c in range(self.n_classes)}
        min_offset = (self.patch_size - self.label_size) // 2
        max_offset = self.patch_size - min_offset - self.label_size

        def process_image(lbl, img_idx, key=None):
            """Efficiently extract patches from one image-label pair."""
            valid_x, valid_y = np.where(
                lbl[
                    min_offset : -max_offset - self.label_size + 1,
                    min_offset : -max_offset - self.label_size + 1,
                ]
                != self.ignore_lbl
            )
            valid_x += min_offset
            valid_y += min_offset

            centers = lbl[valid_x, valid_y]

            for c in range(self.n_classes):
                mask = centers == c
                np.random.seed(42)  # Ensure reproducibility
                if np.where(mask)[0].shape[0] < self.sampling_ratio:
                    continue
                if c == 1:
                    sampled_indices = np.random.choice(
                        np.where(mask)[0],
                        self.sampling_ratio * 2,
                        replace=False,
                    )
                else:
                    sampled_indices = np.random.choice(
                        np.where(mask)[0],
                        self.sampling_ratio,
                        replace=False,
                    )

                for idx in sampled_indices:
                    i, j = valid_x[idx], valid_y[idx]
                    patch_metadata = (key, img_idx, i, j) if key else (img_idx, i, j)
                    if self._centre_consistent(patch_metadata):
                        all_patches.append(
                            (key, img_idx, i - min_offset, j - min_offset)
                            if key
                            else (img_idx, i - min_offset, j - min_offset)
                        )
                        patches_by_label[c].append(len(all_patches) - 1)

        if self.keys:
            for key in self.keys:
                for img_idx, lbl in enumerate(self.labels[key]):
                    process_image(lbl, img_idx, key)
        else:
            for img_idx, lbl in enumerate(self.labels):
                process_image(lbl, img_idx)

        for c in range(self.n_classes):
            random.shuffle(patches_by_label[c])

        return all_patches, patches_by_label

    def update_patches(self, new_label_size):
        self.label_size = new_label_size
        self.all_patches, self.patches_by_label = self._compute_valid_patches()

    def __len__(self):
        """Return dataset size based on mode."""
        if self.mode == "semisupervised":
            return int(len(self.all_patches) / self.ratio)
        else:
            return len(self.all_patches)

    def __getitem__(self, idx):
        if isinstance(idx, list):  # Batch request
            if self.mode == "supervised":
                # Fetch all labeled patches corresponding to the indices
                labeled_patches = [
                    self._get_patch_by_metadata(self.all_patches[i])
                    for i in idx
                    if i < len(self.all_patches)
                ]
                patches, clss, labels = zip(*labeled_patches)
                return torch.stack(patches), torch.tensor(clss), torch.stack(labels)

            elif self.mode == "unsupervised":
                # Fetch random patches for all indices
                # random_patches = self._get_random_patch(idx)
                # patches, clss, labels = zip(*random_patches)
                # return torch.stack(patches), torch.tensor(clss), torch.stack(labels)
                return self._get_random_patch(idx)

            # elif self.mode == "semisupervised":
            #     labeled_count = int(len(idx) * self.ratio)
            #     random_count = len(idx) - labeled_count

            #     # Fetch labeled and random patches
            #     labeled_indices = idx[:labeled_count]
            #     labeled_patches = [
            #         self._get_patch_by_metadata(self.all_patches[i])
            #         for i in labeled_indices
            #     ]
            #     random_patches = self._get_random_patch(random_count)

            #     # Combine and return
            #     all_patches = labeled_patches + random_patches
            #     patches, clss, labels = zip(*all_patches)
            #     return torch.stack(patches), torch.tensor(clss), torch.stack(labels)

        else:  # Single index
            if self.mode == "supervised":
                key, img_idx, y, x = self.all_patches[idx]
                return self._get_patch_by_metadata((key, img_idx, y, x))

            elif self.mode == "unsupervised":
                return self._get_random_patch(idx)

            # elif self.mode == "semisupervised":
            #     if idx < len(self.all_patches):
            #         key, img_idx, y, x = self.all_patches[idx]
            #         return self._get_patch_by_metadata((key, img_idx, y, x))
            #     else:
            #         return self._get_random_patch()

    def _get_patch_by_metadata(self, metadata):
        """Extract a patch dynamically based on metadata."""
        key, img_idx, y, x = metadata
        img = self.images[key][img_idx]
        lbl = self.labels[key][img_idx]
        patch = img[y : y + self.patch_size, x : x + self.patch_size]
        patch_label = lbl[y : y + self.patch_size, x : x + self.patch_size]
        start = (self.patch_size - self.label_size) // 2
        unique_label_area = patch_label[
            start : start + self.label_size,
            start : start + self.label_size,
        ]
        center_label = unique_label_area[0, 0]  # Valid by definition of valid_patches
        return (
            torch.tensor(patch, dtype=torch.float32).unsqueeze(0),
            torch.tensor(center_label, dtype=torch.float16),
            torch.tensor(patch_label, dtype=torch.float16).unsqueeze(0),
        )

    def _get_random_patch(self, idx):

        keys = list(self.images.keys())
        key = random.choice(keys)
        img = self.images[key]
        lbl = self.labels[key]
        depth, height, width = img.shape

        patches = []
        labels = []
        centers = []
        i = 0
        while i < len(idx):
            x = random.randrange(0, width - self.patch_size)
            y = random.randrange(0, height - self.patch_size)
            z = random.randrange(0, depth)
            patch = img[z, y : y + self.patch_size, x : x + self.patch_size]
            patch_label = lbl[z, y : y + self.patch_size, x : x + self.patch_size]
            if patch_label[31, 31] == self.ignore_lbl:
                continue
            center_y = y + self.patch_size // 2 - 1
            center_x = x + self.patch_size // 2 - 1
            if (z, center_y, center_x) in centers:
                print("Duplicate center found, skipping patch")
                continue
            centers.append((z, center_y, center_x))
            patches.append(torch.tensor(patch, dtype=torch.float32).unsqueeze(0))
            labels.append(torch.tensor(patch_label, dtype=torch.float16).unsqueeze(0))
            i += 1

        return (torch.stack(patches), torch.tensor(centers), torch.stack(labels))

    def switch_mode(self):
        if self.mode == "supervised":
            self.mode = "semisupervised"


class CustomLightDataset(Dataset):
    def __init__(
        self,
        images,
        labels,
        patch_size=64,
        label_size=5,
        mode="supervised",  # Options: 'supervised', 'semisupervised', 'unsupervised'
        n_classes=4,
        sampling_ratio=1,
        ignore_lbl=-1,
        ratio=0.75,
    ):
        self.patch_size = patch_size
        self.label_size = label_size
        self.images = images
        self.labels = labels
        self.ignore_lbl = ignore_lbl
        self.n_classes = n_classes
        self.sampling_ratio = sampling_ratio
        self.all_patches, self.patches_by_label = (
            self._compute_valid_patches()
        )  # Store only metadata of valid patches
        self.mode = mode
        self.ratio = ratio

    def set_mode(self, mode):
        """Set the current mode of the dataset."""
        self.mode = mode

    def _centre_consistent(self, patch_metadata):
        """Vectorized version to check if the center is label-consistent."""
        z, x, y = patch_metadata
        unique_label_area = self.labels[
            z, x : x + self.label_size, y : y + self.label_size
        ]
        return (
            np.all(unique_label_area == unique_label_area[0, 0])
            and unique_label_area[0, 0] != self.ignore_lbl
        )

    def _compute_valid_patches(self):
        """Fast vectorized patch extraction."""
        all_patches = []
        patches_by_label = {c: [] for c in range(self.n_classes)}
        min_offset = (self.patch_size - self.label_size) // 2
        max_offset = self.patch_size - min_offset - self.label_size

        def process_image(lbl, img_idx):
            """Efficiently extract patches from one image-label pair."""
            valid_x, valid_y = np.where(
                lbl[
                    min_offset : -max_offset - self.label_size + 1,
                    min_offset : -max_offset - self.label_size + 1,
                ]
                != self.ignore_lbl
            )
            valid_x += min_offset
            valid_y += min_offset

            centers = lbl[valid_x, valid_y]

            for c in range(self.n_classes):
                mask = centers == c
                np.random.seed(42)  # Ensure reproducibility
                if np.where(mask)[0].shape[0] < self.sampling_ratio:
                    continue
                sampled_indices = np.random.choice(
                    np.where(mask)[0],
                    self.sampling_ratio,
                    replace=False,
                )

                for idx in sampled_indices:
                    i, j = valid_x[idx], valid_y[idx]
                    patch_metadata = (img_idx, i, j)
                    if self._centre_consistent(patch_metadata):
                        all_patches.append((img_idx, i - min_offset, j - min_offset))
                        patches_by_label[c].append(len(all_patches) - 1)

        for img_idx, lbl in enumerate(self.labels):
            process_image(lbl, img_idx)

        for c in range(self.n_classes):
            random.shuffle(patches_by_label[c])

        return all_patches, patches_by_label

    def update_patches(self, new_label_size):
        self.label_size = new_label_size
        self.all_patches, self.patches_by_label = self._compute_valid_patches()

    def __len__(self):
        """Return dataset size based on mode."""
        if self.mode == "semisupervised":
            return int(len(self.all_patches) / self.ratio)
        else:
            return len(self.all_patches)

    def __getitem__(self, idx):
        if isinstance(idx, list):  # Batch request
            if self.mode == "supervised":
                # Fetch all labeled patches corresponding to the indices
                labeled_patches = [
                    self._get_patch_by_metadata(self.all_patches[i])
                    for i in idx
                    if i < len(self.all_patches)
                ]
                patches, clss, labels = zip(*labeled_patches)
                return torch.stack(patches), torch.tensor(clss), torch.stack(labels)

            elif self.mode == "unsupervised":
                # Fetch random patches for all indices
                random_patches = [self._get_random_patch() for _ in idx]
                patches, clss, labels = zip(*random_patches)
                return torch.stack(patches), torch.tensor(clss), torch.stack(labels)

            elif self.mode == "semisupervised":
                labeled_count = int(len(idx) * self.ratio)
                random_count = len(idx) - labeled_count

                # Fetch labeled and random patches
                labeled_indices = idx[:labeled_count]
                labeled_patches = [
                    self._get_patch_by_metadata(self.all_patches[i])
                    for i in labeled_indices
                ]
                random_patches = [self._get_random_patch() for _ in range(random_count)]

                # Combine and return
                all_patches = labeled_patches + random_patches
                patches, clss, labels = zip(*all_patches)
                return torch.stack(patches), torch.tensor(clss), torch.stack(labels)

        else:  # Single index
            if self.mode == "supervised":
                img_idx, y, x = self.all_patches[idx]
                return self._get_patch_by_metadata((img_idx, y, x))

            elif self.mode == "unsupervised":
                return self._get_random_patch()

            elif self.mode == "semisupervised":
                if idx < len(self.all_patches):
                    img_idx, y, x = self.all_patches[idx]
                    return self._get_patch_by_metadata((img_idx, y, x))
                else:
                    return self._get_random_patch()

    def _get_patch_by_metadata(self, metadata):
        """Extract a patch dynamically based on metadata."""
        img_idx, y, x = metadata
        img = self.images[img_idx]
        lbl = self.labels[img_idx]
        patch = img[:, y : y + self.patch_size, x : x + self.patch_size]
        patch_label = lbl[y : y + self.patch_size, x : x + self.patch_size]
        start = (self.patch_size - self.label_size) // 2
        unique_label_area = patch_label[
            start : start + self.label_size,
            start : start + self.label_size,
        ]
        center_label = unique_label_area[0, 0]  # Valid by definition of valid_patches
        return (
            torch.tensor(patch),
            torch.tensor(center_label),
            torch.tensor(patch_label),
        )

    def _get_random_patch(self):

        img_idx = random.randrange(0, len(self.images))
        img = self.images[img_idx]
        lbl = self.labels[img_idx]
        c, height, width = img.shape
        x = random.randrange(0, width - self.patch_size)
        y = random.randrange(0, height - self.patch_size)
        patch = img[:, y : y + self.patch_size, x : x + self.patch_size]
        patch_label = lbl[y : y + self.patch_size, x : x + self.patch_size]
        return (
            torch.tensor(patch),
            torch.tensor(-2),
            torch.tensor(patch_label),
        )

    def switch_mode(self):
        if self.mode == "supervised":
            self.mode = "semisupervised"


class Custom2DDatasetMarinoLiver(Custom2DDataset):
    def __init__(
        self,
        images,
        labels,
        patch_size=64,
        label_size=5,
        mode="supervised",  # Options: 'supervised', 'semisupervised', 'unsupervised'
        n_classes=4,
        sampling_ratio=1,
        ignore_lbl=-1,
        ratio=0.75,
    ):
        super().__init__(
            images,
            labels,
            patch_size,
            label_size,
            mode,
            n_classes,
            sampling_ratio,
            ignore_lbl,
            ratio,
        )

    def _compute_valid_patches(self):
        """Fast vectorized patch extraction."""
        all_patches = []
        patches_by_label = {c: [] for c in range(self.n_classes)}
        min_offset = (self.patch_size - self.label_size) // 2
        max_offset = self.patch_size - min_offset - self.label_size

        def process_image(lbl, img_idx, key=None):
            """Efficiently extract patches from one image-label pair."""
            valid_x, valid_y = np.where(
                lbl[
                    min_offset : -max_offset - self.label_size + 1,
                    min_offset : -max_offset - self.label_size + 1,
                ]
                != self.ignore_lbl
            )
            valid_x += min_offset
            valid_y += min_offset

            centers = lbl[valid_x, valid_y]

            for c in range(self.n_classes):
                mask = centers == c
                np.random.seed(42)  # Ensure reproducibility
                if np.where(mask)[0].shape[0] < self.sampling_ratio:
                    if np.where(mask)[0].shape[0] != 0:
                        sampled_indices = np.where(mask)[0]
                        continue
                    else:
                        continue
                if c == 5:
                    if np.where(mask)[0].shape[0] >= self.sampling_ratio * 5:
                        sampled_indices = np.random.choice(
                            np.where(mask)[0],
                            self.sampling_ratio * 5,
                            replace=False,
                        )
                    else:
                        sampled_indices = np.where(mask)[0]
                elif c == 6:
                    if np.where(mask)[0].shape[0] >= self.sampling_ratio * 50:
                        sampled_indices = np.random.choice(
                            np.where(mask)[0],
                            self.sampling_ratio * 50,
                            replace=False,
                        )
                    else:
                        sampled_indices = np.where(mask)[0]
                else:
                    sampled_indices = np.random.choice(
                        np.where(mask)[0],
                        self.sampling_ratio,
                        replace=False,
                    )

                for idx in sampled_indices:
                    i, j = valid_x[idx], valid_y[idx]
                    patch_metadata = (key, img_idx, i, j) if key else (img_idx, i, j)
                    if self._centre_consistent(patch_metadata):
                        all_patches.append(
                            (key, img_idx, i - min_offset, j - min_offset)
                            if key
                            else (img_idx, i - min_offset, j - min_offset)
                        )
                        patches_by_label[c].append(len(all_patches) - 1)

        if self.keys:
            for key in self.keys:
                for img_idx, lbl in enumerate(self.labels[key]):
                    process_image(lbl, img_idx, key)
        else:
            for img_idx, lbl in enumerate(self.labels):
                process_image(lbl, img_idx)

        for c in range(self.n_classes):
            random.shuffle(patches_by_label[c])

        return all_patches, patches_by_label


class Custom3DDataset(Dataset):
    """
    A custom dataset that extracts patches from 3D images and extract them based on their labels.
    """

    def __init__(self, images, labels, patch_size=(64, 64, 64), mask_size=(5, 5, 5)):
        """
        Initialize the Custom3DDataset by extracting valid 3D patches.

        Parameters:
        -----------
        images : dict
            A dictionary of 3D images.
        labels : dict
            A dictionary of corresponding labels for the 3D images.
        patch_size : tuple
            Size of the 3D patches (default is (64, 64, 64)).
        mask_size : tuple
            Size of the masked area in 3D (default is (5, 5, 5)).
        """
        self.patch_size = patch_size
        self.mask_size = mask_size
        self.all_patches = []  # List to store all patches (with different labels)
        self.patches_by_label = self._extract_valid_patches(images, labels)

    def __len__(self):
        return len(self.all_patches)

    def _extract_valid_patches(self, images, labels):
        """
        Extracts valid 3D patches from the given images based on the provided labels.

        Parameters:
        -----------
        images : list
            A list of 3D images.
        labels : list
            A list of corresponding labels for the 3D images.

        Returns:
        --------
        patches_by_label : dict
            A dictionary mapping labels to indices of patches.
        """
        patches_by_label = {}
        for img, lbl in zip(images, labels):
            depth, height, width = img.shape
            d_patch, h_patch, w_patch = self.patch_size
            z_center = d_patch // 2
            y_center = h_patch // 2
            x_center = w_patch // 2
            d_stride, h_stride, w_stride = self.mask_size
            d_stride *= 3
            h_stride *= 3
            w_stride *= 3
            # Iterate over 3D volumes (depth, height, width) to extract patches
            for z in tqdm(
                range(0, depth, d_stride), f"Extracting patches from volume: "
            ):
                for y in range(0, height, h_stride):
                    for x in range(0, width, w_stride):

                        # Extract 3D patch and corresponding label
                        patch = img[z : z + d_patch, y : y + h_patch, x : x + w_patch]
                        patch_label = lbl[
                            z : z + d_patch, y : y + h_patch, x : x + w_patch
                        ]
                        if patch.shape != self.patch_size:
                            continue
                        # Extract blind spot area in the center
                        center_label = patch_label[z_center, y_center, x_center]

                        if center_label != -1:
                            if center_label not in patches_by_label:
                                patches_by_label[center_label] = []

                            # Store patch, center label, and full patch label
                            self.all_patches.append(
                                (
                                    torch.tensor(patch).unsqueeze(
                                        0
                                    ),  # Add channel dimension
                                    torch.tensor(center_label),
                                    torch.tensor(patch_label).unsqueeze(
                                        0
                                    ),  # Add channel dimension
                                )
                            )
                            patches_by_label[center_label].append(
                                len(self.all_patches) - 1
                            )
        return patches_by_label

    def __getitem__(self, idx):
        """
        Retrieves the 3D patch, its class, and the label map.

        Parameters:
        -----------
        idx : int
            Index of the patch.

        Returns:
        --------
        patch : torch.Tensor
            The 3D patch extracted from the image.
        cls : torch.Tensor
            The label of the patch's center.
        label : torch.Tensor
            The full label map of the patch.
        """
        if isinstance(idx, list):
            patches = [self.all_patches[i] for i in idx]
            patches, clss, labels = zip(*patches)
            return (
                torch.stack(patches).squeeze(0),
                torch.tensor(clss),
                torch.stack(labels),
            )
        else:
            patch, cls, label = self.all_patches[idx]
        return patch, cls, label


class CustomTestDataset(Dataset):
    def __init__(self, image, patch_size=(64, 64, 64), index=1, stride=1, model="3D"):
        """
        Custom Dataset for extracting 2D/3D patches from test data.

        Args:
            image (ndarray): The input image (2D or 3D array).
            patch_size (tuple): Size of the patches to extract (depth, height, width for 3D, height, width for 2D).
            index (int): The depth slice index for 2D patching or center for 3D.
            stride (int): Stride for patch extraction.
            model (str): "2D" or "3D" mode to control patch dimensionality.
        """
        self.image = image
        self.patch_size = patch_size
        self.stride = stride
        self.model = model

        if model == "3D":
            assert len(patch_size) == 3, "3D model requires a 3D patch size."
            self.depth = index - (patch_size[0] // 2)
        elif model == "2D":
            assert len(patch_size) == 2, "2D model requires a 2D patch size."
            self.patch_size = (1, *patch_size)  # Add a dummy depth for uniform handling
            self.depth = index  # Fixed slice for 2D patches
        elif model == "2D_multichannel":
            assert len(patch_size) == 3, "2D model requires a 2D patch size."
            self.depth = index
            self.patch_size = patch_size
        else:
            raise ValueError("Model type must be '2D' or '3D'.")

        _, self.height, self.width = (
            image.shape if model == "3D" else (1, *image.shape[1:])
        )
        self.num_patches_y = (self.height - self.patch_size[1]) // stride + 1
        self.num_patches_x = (self.width - self.patch_size[2]) // stride + 1

    def __len__(self):
        """Returns the total number of patches."""
        return self.num_patches_y * self.num_patches_x

    def __getitem__(self, index):
        """
        Extracts a patch based on the index.

        Args:
            index (int): Index of the patch.

        Returns:
            torch.Tensor: Extracted patch as a tensor.
        """
        y = (index // self.num_patches_x) * self.stride
        x = (index % self.num_patches_x) * self.stride

        if self.model == "3D":
            patch = self.image[
                self.depth : self.depth + self.patch_size[0],
                y : y + self.patch_size[1],
                x : x + self.patch_size[2],
            ]
        elif self.model == "2D":  # For 2D
            patch = self.image[
                self.depth,
                y : y + self.patch_size[1],
                x : x + self.patch_size[2],
            ]
        elif self.model == "2D_multichannel":
            patch = self.image[
                :,
                y : y + self.patch_size[1],
                x : x + self.patch_size[2],
            ]

        patch_tensor = torch.tensor(patch)
        # Add a channel dimension for PyTorch compatibility
        if self.model != "2D_multichannel":
            patch = patch_tensor.unsqueeze(0)

        return patch


class LabeledPatchDataset(Dataset):
    def __init__(
        self,
        image,
        label_map,
        patch_size=(64, 64),
        num_per_class=100,
        classes=[0, 1, 2, 3],
    ):
        """
        Extracts 2D patches centered on labeled pixels from a 3D image.

        Args:
            image (ndarray): 3D image of shape (D, H, W).
            label_map (ndarray): 3D label map of same shape as image.
            patch_size (tuple): (H, W) patch size to extract.
            num_per_class (int): Number of patches to extract per class.
            classes (list): List of class labels to sample.
        """
        assert (
            image.shape == label_map.shape
        ), "Image and label_map must have same shape"
        self.image = image
        self.label_map = label_map
        self.patch_size = patch_size
        self.classes = classes
        self.num_per_class = num_per_class
        self.patches = []

        ph, pw = patch_size
        margin_h, margin_w = ph // 2, pw // 2

        D, H, W = image.shape

        for cls in classes:
            coords = np.argwhere(label_map == cls)
            # Remove border cases
            valid_coords = [
                (z, y, x)
                for z, y, x in coords
                if margin_h <= y < H - margin_h and margin_w <= x < W - margin_w
            ]
            if len(valid_coords) < num_per_class:
                print(
                    f"⚠️ Warning: Not enough samples for class {cls}, using {len(valid_coords)}"
                )
            selected = np.random.choice(
                len(valid_coords), min(num_per_class, len(valid_coords)), replace=False
            )
            for i in selected:
                self.patches.append((valid_coords[i], cls))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        (z, y, x), cls = self.patches[idx]
        ph, pw = self.patch_size
        half_h, half_w = ph // 2, pw // 2
        patch = self.image[
            z, y - half_h + 1 : y + half_h + 1, x - half_w + 1 : x + half_w + 1
        ]
        return torch.tensor(patch, dtype=torch.float32).unsqueeze(0), cls, (z, y, x)


class CombinedCustom3DDataset(Custom3DDataset):
    """
    A combined dataset class that handles labeled and unlabeled 3D data for training with
    contrastive loss and other unsupervised losses.
    """

    def __init__(
        self,
        images,
        labels,
        labeled_indices,
        patch_size=(64, 64, 64),
        mask_size=(5, 5, 5),
    ):
        """
        Initialize the CombinedCustom3DDataset, separating patches into labeled and unlabeled subsets.

        Parameters:
        -----------
        images : dict
            A dictionary of 3D images.
        labels : dict
            A dictionary of corresponding labels for the 3D images.
        labeled_indices : list
            A list of indices of `all_patches` that should keep their labels for contrastive loss.
        patch_size : tuple
            Size of the 3D patches (default is (64, 64, 64)).
        mask_size : tuple
            Size of the masked area in 3D (default is (5, 5, 5)).
        """
        super().__init__(images, labels, patch_size, mask_size)

        # Separate labeled and unlabeled patches based on labeled_indices
        self.labeled_indices = labeled_indices
        self._get_random_patch(images, labels)
        self._update_patches_by_label()

    def __getitem__(self, idx):
        """
        Retrieves the patch, its class, and the label map.

        Parameters:
        -----------
        idx : int
            Index of the patch.

        Returns:
        --------
        patch : torch.Tensor
            The 3D patch extracted from the image.
        cls : torch.Tensor
            The label of the patch's center (-2 if the patch is from the unlabeled subset).
        label : torch.Tensor
            The full label map of the patch.
        """

        # Return with label set to -2 if the index is not part of labeled indices
        if isinstance(idx, list):
            patches = [
                (
                    self.all_patches[i]
                    if i in self.labeled_indices
                    else self.all_patches[i]
                )
                for i in idx
            ]
            patches, clss, labels = zip(*patches)
            return (
                torch.stack(patches).squeeze(0),
                torch.tensor(clss),
                torch.stack(labels),
            )
        else:
            if idx in self.labeled_indices:
                patch, cls, label = self.all_patches[idx]
            else:
                patch, cls, label = self.all_patches[idx]
            return patch, cls, label

    def _get_random_patch(self, images, labels):
        """
        Retrieves a random patch from the dataset.

        Returns:
        --------
        patch : torch.Tensor
            The random 3D patch.
        label : torch.Tensor
            The label map of the random 3D patch.
        """
        indices = range(len(images))
        for i in tqdm(range(len(self.all_patches))):
            if i not in self.labeled_indices:
                idx = random.choice(indices)
                img = images[idx]
                lbl = labels[idx]
                depth, height, width = img.shape

                z = random.randrange(0, depth - self.patch_size[0])
                y = random.randrange(0, height - self.patch_size[1])
                x = random.randrange(0, width - self.patch_size[2])

                patch = img[
                    z : z + self.patch_size[0],
                    y : y + self.patch_size[1],
                    x : x + self.patch_size[2],
                ]
                patch_label = lbl[
                    z : z + self.patch_size[0],
                    y : y + self.patch_size[1],
                    x : x + self.patch_size[2],
                ]

                self.all_patches[i] = (
                    torch.tensor(patch).unsqueeze(0),  # Add channel dimension
                    torch.tensor(-2),  # Label set to -2 for unlabeled
                    torch.tensor(patch_label).unsqueeze(0),  # Add channel dimension
                )
        return

    def _update_patches_by_label(self):
        """
        Updates patches_by_label dictionary to include only labeled patches.
        """
        for key in self.patches_by_label:
            self.patches_by_label[key] = [
                value
                for value in self.patches_by_label[key]
                if value in self.labeled_indices
            ]


class BalancedBatchSampler(Sampler):
    """
    A custom sampler that generates balanced batches from a dataset by ensuring each batch
    contains a balanced number of samples from each label class.

    This sampler is useful when training models with imbalanced datasets, as it helps to
    maintain an equal representation of each class within each batch. The class ensures
    that samples from each label are included in the batch proportionally and handles
    scenarios where the number of samples for each label differs significantly.

    Attributes:
    -----------
    dataset : Dataset
        The dataset from which samples are drawn. The dataset should have a `patches_by_label`
        attribute, which is a dictionary mapping labels to indices of samples belonging to
        those labels.

    batch_size : int
        The total number of samples in each batch.

    label_to_indices : dict
        A dictionary that maps each label to a list of indices of samples that belong
        to that label.

    num_labels : int
        The number of unique labels in the dataset.

    samples_per_label : int
        The number of samples to include from each label in each batch.

    remaining_samples : int
        The number of extra samples to distribute across labels to fill the batch.

    max_batch : int
        The maximum number of batches that can be generated based on the size of the
        largest class and the number of samples per label.

    Methods:
    --------
    __init__(dataset, batch_size)
        Initializes the sampler with the dataset and batch size.

    __iter__()
        Returns an iterator that yields balanced batches of indices.

    __len__()
        Estimates the total number of batches that can be generated.
       pass
    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        # dictionary mapping labels to indices
        self.label_to_indices = dataset.patches_by_label
        for key in self.label_to_indices:
            random.shuffle(self.label_to_indices[key])

        # Determine number of labels
        self.num_labels = len(self.label_to_indices)
        self.samples_per_label = self.batch_size // self.num_labels
        self.remaining_samples = self.batch_size % self.num_labels
        self.max_batch = (
            max(len(indices) for indices in self.label_to_indices.values())
            // self.samples_per_label
        )

    def __iter__(self):
        max_class_size = max(len(indices) for indices in self.label_to_indices.values())
        num_batches_generated = 0
        # Generate balanced batches
        while num_batches_generated < self.max_batch:
            batch = []
            for label, indices in self.label_to_indices.items():
                if len(indices) < self.samples_per_label:
                    indices = random.choices(indices, k=max_class_size)  # Oversample
                selected_indices = random.sample(indices, self.samples_per_label)
                batch.extend(selected_indices)

            if len(batch) < self.batch_size:
                # Handle any remaining spots in the batch
                remaining_indices = []
                for indices in self.label_to_indices.values():
                    remaining_indices.extend(indices)
                random.shuffle(remaining_indices)
                batch.extend(remaining_indices[: self.batch_size - len(batch)])

            if len(batch) == self.batch_size:
                random.shuffle(batch)
                num_batches_generated += 1
                yield batch
            # Return the batch and pause execution until the next batch is requested

    def __len__(self):
        # Estimate the length based on the largest class
        max_class_size = max(len(indices) for indices in self.label_to_indices.values())
        return (max_class_size * self.num_labels) // self.batch_size


class CombinedBatchSampler(Sampler):
    """
    A custom sampler that generates batches containing 50% balanced labeled samples
    and 50% random samples. Inherits from BalancedBatchSampler.
    """

    def __init__(self, dataset, batch_size, labeled_ratio=0.50):
        """
        Initializes the CombinedBatchSampler.

        Parameters:
        -----------
        dataset : Dataset
            The dataset from which samples are drawn. Should have a `patches_by_label`
            attribute for labeled patches.
        batch_size : int
            The total number of samples in each batch.

        """
        self.label_to_indices = dataset.patches_by_label
        self.random_indices = range(int(len(dataset) * labeled_ratio), len(dataset))
        for key in self.label_to_indices:
            random.shuffle(self.label_to_indices[key])
        self.batch_size = batch_size
        self.small_batch_size = int(batch_size * labeled_ratio)
        self.num_labels = len(self.label_to_indices)
        self.samples_per_label = self.small_batch_size // self.num_labels
        self.remaining_samples = self.small_batch_size % self.num_labels
        self.max_batch = (
            max(len(indices) for indices in self.label_to_indices.values())
            // self.samples_per_label
        )

    def __iter__(self):
        max_class_size = max(len(indices) for indices in self.label_to_indices.values())
        num_batches_generated = 0

        while num_batches_generated < self.max_batch:
            # Step 1: Sample 25% of the batch using balanced sampling from labeled indices
            balanced_batch = []
            for label, indices in self.label_to_indices.items():
                if len(indices) < self.samples_per_label:
                    indices = random.choices(indices, k=max_class_size)
                selected_indices = random.sample(indices, self.samples_per_label)
                balanced_batch.extend(selected_indices)

            # Fill up if the balanced batch is not full (due to class imbalance or fewer labeled samples)
            while len(balanced_batch) < self.small_batch_size:
                remaining_labeled = []
                for indices in self.label_to_indices.values():
                    remaining_labeled.extend(indices)
                random.shuffle(remaining_labeled)
                balanced_batch.extend(
                    remaining_labeled[: self.small_batch_size - len(balanced_batch)]
                )

            # Step 2: Sample 75% of the batch randomly from unlabeled indices
            random_unlabeled = random.sample(
                self.random_indices, self.batch_size - self.small_batch_size
            )

            # Combine balanced labeled and random unlabeled samples
            combined_batch = balanced_batch + random_unlabeled

            # Ensure the final batch size is correct
            if len(combined_batch) == self.batch_size:
                num_batches_generated += 1
                yield combined_batch

    def __len__(self):
        max_class_size = max(len(indices) for indices in self.label_to_indices.values())
        return (max_class_size * self.num_labels) // self.small_batch_size


class UnsupervisedSampler(Sampler):
    """
    A custom sampler that generates batches containing random samples from the dataset.
    """

    def __init__(self, dataset, batch_size):
        """
        Initializes the UnsupervisedSampler.

        Parameters:
        -----------
        dataset : Dataset
            The dataset from which samples are drawn.

        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_batch = len(dataset) // batch_size

    def __iter__(self):
        indices = range(len(self.dataset))
        num_batches_generated = 0
        while num_batches_generated < self.max_batch:
            batch = random.sample(indices, self.batch_size)
            num_batches_generated += 1
            yield batch

    def __len__(self):
        return self.max_batch


class DynamicSampler(Sampler):
    def __init__(self, dataset, batch_size, labeled_ratio=0.25):
        self.dataset = dataset
        self.batch_size = batch_size
        self.labeled_ratio = labeled_ratio

    def __iter__(self):
        if self.dataset.mode == "supervised":
            sampler = BalancedBatchSampler(self.dataset, self.batch_size)
        elif self.dataset.mode == "semisupervised":
            sampler = CombinedBatchSampler(
                self.dataset, self.batch_size, labeled_ratio=self.labeled_ratio
            )
        elif self.dataset.mode == "unsupervised":
            sampler = UnsupervisedSampler(self.dataset, self.batch_size)
        yield from iter(sampler)

    def __len__(self):
        return len(self.dataset) // self.batch_size


def ordered_collate_fn(batch):
    """
    Batch structure:
        - First N: anchors
        - Next 3N: neighbors grouped by anchor
    """
    anchor_patches = []
    anchor_labels = []
    neighbor_patches = []
    neighbor_labels = []
    anchor_meta = []
    neighbor_meta = []

    # torch.stack(patches),  # shape [4, 1, 64, 64]
    # anchor["label"],
    # {
    #     "neighbor_labels": torch.tensor(neighbor_labels, dtype=torch.long),
    #     "anchor_meta": anchor,
    #     "neighbor_meta": neighbors,
    # }

    for sample in batch:
        anchor_patches.append(sample[0])  # shape [1, 64, 64]
        anchor_labels.append(sample[1])
        anchor_meta.append(sample[2]["anchor_meta"])

        # neighbors: patches[1:] = 3 unlabeled patches
        neighbor_patches.extend(sample["patches"][1:])  # 3 x [1, 64, 64]
        neighbor_labels.extend(sample["neighbor_labels"])  # list of 3
        neighbor_meta.extend(sample["neighbor_meta"])

    return {
        "patches": torch.cat(
            anchor_patches + neighbor_patches, dim=0
        ),  # [4n, 1, 64, 64]
        "labels": torch.tensor(anchor_labels, dtype=torch.long),  # [n]
        "neighbor_labels": torch.tensor(neighbor_labels, dtype=torch.long),  # [3n]
        "anchor_meta": anchor_meta,
        "neighbor_meta": neighbor_meta,
    }
