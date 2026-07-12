# Efficacy test for multi-scale local crops in dataloader.py.
# Builds a tiny synthetic parquet shard and runs the REAL TCGATileDataset.__getitem__:
#   1. scalar local_size path is unchanged: one (local_views, 3, S, S) stacked tensor,
#   2. local_size_mix path yields a LIST of per-size stacks with the right counts/sizes,
#      totalling >= 8 crops, and default_collate batches each group correctly.
#
# Run: /admin/home/ryan.kim/nanopath/.venv/bin/python3 tests/test_local_size_mix.py
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataloader import TCGATileDataset


def _write_shard(dir_path, n=6):
    rows_path, rows_jpeg = [], []
    for i in range(n):
        # Distinct patient barcodes; RGB noise tile so tissue/aug paths run.
        arr = (np.random.default_rng(i).integers(0, 255, (224, 224, 3))).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr, "RGB").save(buf, format="JPEG")
        rows_jpeg.append(buf.getvalue())
        rows_path.append(f"TCGA-AA-{1000 + i}-01Z/tile_{i}.jpg")
    table = pa.table({"path": pa.array(rows_path, pa.string()), "jpeg": pa.array(rows_jpeg, pa.binary())})
    pq.write_table(table, str(Path(dir_path) / "shard-00000.parquet"), row_group_size=64)


def _base_cfg(dataset_dir, local_size=112, local_views=8, local_size_mix=None):
    train = {"global_views": 2, "local_views": local_views, "global_size": 224, "local_size": local_size}
    if local_size_mix is not None:
        train["local_size_mix"] = local_size_mix
    return {
        "data": {
            "dataset_dir": str(dataset_dir), "split_seed": 7777, "val_fraction": 0.0,
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "global_crop_scale": [0.4, 1.0], "local_crop_scale": [0.08, 0.35],
            "color_jitter": 0.3, "color_jitter_saturation": 0.3, "hed_jitter": 0.07, "tissue_thresh": 0.0,
        },
        "train": train,
    }


def test_scalar_path_unchanged(dataset_dir):
    ds = TCGATileDataset(_base_cfg(dataset_dir, local_size=112, local_views=8), is_train=True)
    assert ds.local_size_mix is None
    assert ds.local_views == 8
    out = ds[0]
    lv = out["local_views"]
    assert isinstance(lv, torch.Tensor), type(lv)
    assert tuple(lv.shape) == (8, 3, 112, 112), lv.shape
    print(f"  [scalar] local_views tensor shape={tuple(lv.shape)} (8 crops @112)  OK")


def test_mix_path_counts_and_sizes(dataset_dir):
    ds = TCGATileDataset(_base_cfg(dataset_dir, local_size_mix=[[112, 4], [128, 4]]), is_train=True)
    assert ds.local_size_mix == [(112, 4), (128, 4)]
    assert ds.local_views == 8, ds.local_views
    out = ds[0]
    lv = out["local_views"]
    assert isinstance(lv, list) and len(lv) == 2, type(lv)
    assert tuple(lv[0].shape) == (4, 3, 112, 112), lv[0].shape
    assert tuple(lv[1].shape) == (4, 3, 128, 128), lv[1].shape
    total = sum(t.shape[0] for t in lv)
    assert total == 8, total
    print(f"  [mix 4x112+4x128] groups={[tuple(t.shape) for t in lv]} total={total} crops  OK")


def test_mix_uneven_total_ge8(dataset_dir):
    ds = TCGATileDataset(_base_cfg(dataset_dir, local_size_mix=[[112, 8], [128, 4]]), is_train=True)
    assert ds.local_views == 12
    lv = ds[0]["local_views"]
    assert [tuple(t.shape) for t in lv] == [(8, 3, 112, 112), (4, 3, 128, 128)]
    print("  [mix 8x112+4x128] total=12 crops, per-size stacks correct  OK")


def test_mix_collate(dataset_dir):
    ds = TCGATileDataset(_base_cfg(dataset_dir, local_size_mix=[[112, 4], [128, 4]]), is_train=True)
    batch = default_collate([ds[i] for i in range(3)])
    lv = batch["local_views"]
    assert isinstance(lv, list) and len(lv) == 2
    assert tuple(lv[0].shape) == (3, 4, 3, 112, 112), lv[0].shape  # (B, count, 3, S, S)
    assert tuple(lv[1].shape) == (3, 4, 3, 128, 128), lv[1].shape
    # train.py's crop-major flatten per group: (B,count,...)->(count,B,...)->(count*B,...)
    flat0 = lv[0].transpose(0, 1).flatten(0, 1)
    flat1 = lv[1].transpose(0, 1).flatten(0, 1)
    assert tuple(flat0.shape) == (12, 3, 112, 112) and tuple(flat1.shape) == (12, 3, 128, 128)
    print(f"  [collate] batched groups={[tuple(lv[0].shape), tuple(lv[1].shape)]}; "
          f"flattened={[tuple(flat0.shape), tuple(flat1.shape)]}  OK")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        _write_shard(d)
        for fn in [test_scalar_path_unchanged, test_mix_path_counts_and_sizes,
                   test_mix_uneven_total_ge8, test_mix_collate]:
            print(f"{fn.__name__}:")
            fn(d)
    print("\nALL LOCAL_SIZE_MIX TESTS PASSED")
