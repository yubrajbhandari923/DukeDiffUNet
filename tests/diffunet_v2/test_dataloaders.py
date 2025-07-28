# tests/test_dataloaders.py


import pytest
import torch
import time
from hydra import initialize, compose
from monai.utils.enums import CommonKeys as Keys
# from tests.diffunet_v2.conftest import cfg
from train.diffunet_2_0 import get_dataloaders, prepare_batch, derive_experiment_metadata

# define the combos you care about
EXPS = [
    ("constraint=binary", "task=colon", 2, 2),
    ("constraint=multi_class", "task=colon", 14, 2),
    ("constraint=binary", "task=colon_bowel", 2, 3),
    ("constraint=multi_class", "task=colon_bowel", 14, 3),
]


@pytest.mark.parametrize("cstr,tsk,in_ch,out_ch", EXPS)
def test_loader_binary_vs_multiclass(tmp_path, cstr, tsk, in_ch, out_ch):
    # re‐compose the cfg with both constraint & task overrides
    with initialize(
        version_base=None,
        config_path="../../configs/diffunet_v2/",
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                cstr,
                tsk,
                "experiment.debug=true",
                "data.debug_save_data=true",
                "data.cache_dir=null",  # disable caching for tests
                "training.num_gpus=1",  # use CPU‐only for tests
                "data.batch_size_per_gpu=1",  # single‐batch for tests
                "data.num_workers_per_gpu=1",  # reasonable number of workers
            ],
        )
    derive_experiment_metadata(cfg)
    print(f"Testing with cfg: {cfg.experiment.name}")
    
    # Quick sanity on cfg
    assert cfg.data.shuffle_train_data is True, "Shuffle should be enabled for training data"
    assert cfg.data.cache_dir is None, "Cache directory should be disabled for tests"
    assert cfg.experiment.debug is True
    assert cfg.model.params.in_channels == in_ch
    assert cfg.model.params.out_channels == out_ch

    # 4) Time loader creation
    t0 = time.perf_counter()
    train_loader, val_loader = get_dataloaders(
        cfg,
        aim_logger=None,
    )
    dt_loader = time.perf_counter() - t0
    print(f"  ▶ get_dataloaders init: {dt_loader:.3f}s")

    # ----------- TRAIN BATCH TEST -----------
    t1 = time.perf_counter()
    batch = next(iter(train_loader))
    img, lbl = prepare_batch(batch, device="cpu")
    dt_prep = time.perf_counter() - t1
    print(f"  ▶ train batch prep: {dt_prep:.3f}s")

    # shape & channel checks
    assert img.ndim == 5 and lbl.ndim == 5
    B, C, H, W, D = img.shape
    assert B == 1
    assert C == in_ch
    assert lbl.shape[1] == out_ch

    # ----------- VAL BATCH TEST -----------
    v0 = time.perf_counter()
    val_batch = next(iter(val_loader))
    v_img, v_lbl = prepare_batch(val_batch, device="cpu")
    dt_val_prep = time.perf_counter() - v0
    print(f"  ▶ val batch prep   : {dt_val_prep:.3f}s")

    # val_img should also be 5-D: (B, C_img, H, W, D)
    assert v_img.ndim == 5
    assert v_img.shape[1] == cfg.model.params.in_channels

    # val_lbl is one-hot over all classes
    assert v_lbl.ndim == 5
    assert v_lbl.shape[1] == cfg.model.params.out_channels