# tests/conftest.py

import pytest
from hydra import initialize, compose
from omegaconf import DictConfig


@pytest.fixture(scope="function")
def cfg() -> DictConfig:
    # point to the directory holding your experiment/config.yaml
    with initialize(
        version_base=None,
        # config_path="configs/diffunet_v2/config.yaml",
        config_path="../../configs/diffunet_v2/",
    ):
        # load defaults (config.yaml) + any overrides
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment.debug=true",  # turn on debug for tests
                # you can also override constraint/task here if you want per-test
            ],
        )
    return cfg
