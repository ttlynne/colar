"""
qsa.py  ─  QuestionStepsAnswer dataset, extended to support image inputs.

Each JSON record may optionally contain an "image_path" key pointing to an
image file relative to the dataset directory.  When present, the image is
loaded as a PIL.Image and returned in the batch under the key "image".
Text-only records (no "image_path") return None for that key.

JSON record format (text-only, original):
  {"question": "...", "answer": "42", "steps": ["step1", "step2"]}

JSON record format (multimodal, new):
  {"question": "...", "answer": "42", "steps": ["step1"], "image_path": "images/001.png"}
"""

if __name__ == "__main__":
    import sys
    sys.path.append("../../")

import copy
from pathlib import Path
import json
from typing import List, Dict, Optional
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl


def _load_image(image_path: Optional[str]):
    """Load a PIL image from *image_path*, or return None."""
    if image_path is None:
        return None
    try:
        from PIL import Image
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[WARNING] Could not load image {image_path}: {e}")
        return None


def _collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function that handles heterogeneous batches where
    the "image" field can be a mix of PIL.Image objects and None values.

    All tensor fields are stacked normally by the default collate.
    The "image" field is returned as a plain Python list.
    """
    import torch
    from torch.utils.data._utils.collate import default_collate

    # Separate out the image field (cannot be stacked by default_collate)
    images = [sample.pop("image") for sample in batch]
    collated = default_collate(batch)
    collated["image"] = images          # List[PIL.Image | None]

    # Restore the image field in the original dicts so the dataset is unmodified
    for sample, img in zip(batch, images):
        sample["image"] = img

    return collated


class QuestionStepsAnswerDataset(Dataset):
    def __init__(
        self,
        data: List[dict],
        dataset_dir: Optional[Path] = None,   # needed for relative image paths
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.data = {}
        for idx, d in enumerate(data):
            # Resolve image path if present
            image_path = d.get("image_path", None)
            if image_path is not None and dataset_dir is not None:
                image_path = str(dataset_dir / image_path)

            self.data[idx] = {
                "idx": idx,
                "question": d["question"],
                "answer": d["answer"],
                "steps": "\n".join(d["steps"]),
                "n_steps": len(d["steps"]),
                "image_path": image_path,   # absolute str or None
            }
        self.all_indices = list(self.data.keys())
        self.indices = copy.deepcopy(self.all_indices)

    def get_all_indices(self):
        return self.all_indices

    def set_indices(self, indices: List[int]):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        data_idx = self.indices[idx]
        record = dict(self.data[data_idx])          # shallow copy
        # Lazy-load image at access time
        record["image"] = _load_image(record.pop("image_path"))
        return record


class QSADataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, tiny_dataset=False, epoch_scaling=1, all_config=None):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = Path(all_config.args.workspace_path, "datasets", "text_reasoning", dataset_name)
        self.tiny_dataset = tiny_dataset
        self.epoch_scaling = epoch_scaling
        self.all_config = all_config
        self.batch_size = all_config.dataloader.batch_size

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage: str = None):
        def load_split(split: str):
            with open(self.dataset_dir / f"{split}.json") as f:
                data = json.load(f)
                if self.tiny_dataset:
                    data = data[:32]
            return data

        if stage == "fit":
            self.train_set = self._create_dataset(load_split("train"))
            self.val_set = self._create_dataset(load_split("val"))
        elif stage == "test":
            self.test_set = self._create_dataset(load_split("test"))

    def _create_dataset(self, raw_data: List[dict]) -> QuestionStepsAnswerDataset:
        return QuestionStepsAnswerDataset(
            data=raw_data,
            dataset_dir=self.dataset_dir,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            shuffle=True,
            batch_size=self.all_config.dataloader.batch_size,
            num_workers=self.all_config.dataloader.get("num_workers", 4),
            pin_memory=self.all_config.dataloader.get("pin_memory", True),
            persistent_workers=self.all_config.dataloader.get("persistent_workers", True),
            collate_fn=_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.all_config.dataloader.get("val_batch_size", 1),
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.all_config.dataloader.get("val_batch_size", 1),
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=_collate_fn,
        )

    def get_dataloader_to_filter_indices(self):
        return DataLoader(
            self.train_set,
            batch_size=8,
            shuffle=False,
            collate_fn=_collate_fn,
        )

    def get_all_train_indices(self):
        return self.train_set.get_all_indices()

    def set_train_indices(self, train_indices):
        self.train_set.set_indices(train_indices)