"""Tests the config helper."""

from morai.utils import config_helper


def test_get_config():
    """Tests the get_config function."""
    config = config_helper.get_config("config.yaml")
    # has section config in dict
    assert "config" in config, "Config file does not have section config"


def test_get_config_options():
    """Tests the get_config_options function."""
    options = config_helper.get_config_options("config.yaml", "config", "integrations")
    assert "hmd_email" in options, "Config file does not have option hmd_email"


def test_config_reference(monkeypatch):
    """Tests the config_reference function."""
    config = {"section": {}, "static": {"mykey": "myval"}}
    monkeypatch.setenv("MY_ENV", "env_value")
    test_static = config_helper._config_reference(config, "static.mykey")
    test_env = config_helper._config_reference(config, "$MY_ENV")
    test_str = config_helper._config_reference(config, "just_a_value")

    # none
    assert config_helper._config_reference(config, "None") is None
    assert config_helper._config_reference(config, "null") is None
    assert config_helper._config_reference(config, "") is None

    # static
    assert test_static == "myval"

    # environment
    assert test_env == "env_value"

    # string
    assert test_str == "just_a_value"
