# tests/test_config_overrides.py
import pytest
from hydra import initialize, compose


@pytest.mark.parametrize(
    "override, attr, expected",
    [
        ("constraint=binary", "constraint", "binary"),
        ("constraint=multi_class", "constraint", "multi_class"),
        ("task=colon", "task", "colon"),
        ("task=colon_bowel", "task", "colon_bowel"),
        ("experiment.debug=true", "experiment.debug", True),
    ],
)
def test_simple_override(override, attr, expected):
    """
    Compose the config with a single override, then
    dot‐walk `cfg` to the attribute named in `attr`
    and assert it matches the expected value.
    """
    with initialize(
        version_base=None,
        config_path="../../configs/diffunet_v2/",
    ):
        cfg = compose(config_name="config", overrides=[override])

    if not "." in attr:
        # If attr is a simple attribute, we can access it directly
        assert getattr(cfg, attr).target == expected
    
    else:
        attr_parts = attr.split(".")
        val = cfg
        for part in attr_parts:
            val = getattr(val, part)
        assert val == expected


def test_defaults_and_debug_flag():
    """
    Without any overrides, config.constraint and config.task
    should be the defaults from config.yaml; and
    experiment.debug stays False unless overridden.
    """
    with initialize(
        version_base=None,
        config_path="../../configs/diffunet_v2/",
    ):
        cfg = compose(config_name="config", overrides=[])

    assert cfg.constraint.target == "binary"  # default from config.yaml
    assert cfg.task.target == "colon"  # default from config.yaml
    assert cfg.experiment.debug is False  # default should be False
    
    assert cfg.model.params.in_channels == 2  # default from config.yaml
    

    # Now flip debug on via override
    with initialize(version_base=None, config_path="../../configs/diffunet_v2/"):
        cfg2 = compose(config_name="config", overrides=["experiment.debug=true"])
    
    assert cfg2.experiment.debug is True
