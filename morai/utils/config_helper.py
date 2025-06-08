"""Load config."""

import configparser
import os

import yaml

from morai.utils import custom_logger, helpers

logger = custom_logger.setup_logging(__name__)


def get_config(path: str) -> dict:
    """
    Get the config path.

    Parameters
    ----------
    path : str
       path to the config file

    Returns
    -------
    config : configparser.ConfigParser
        the config parser

    """
    config = configparser.ConfigParser()

    # test if path exists and try default directories
    # (FILES_PATH)
    paths_to_try = [
        path,
        os.path.join(helpers.FILES_PATH, path),
    ]
    for path_to_try in paths_to_try:
        try:
            with open(path_to_try):
                break
        except FileNotFoundError:
            continue
    else:
        paths_str = ", ".join(map(str, paths_to_try))
        raise FileNotFoundError(
            f"Config file not found at any of the following paths: {paths_str}"
        )

    with open(path_to_try, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_config_options(path: str, *sections: str) -> dict:
    """
    Load the configuration options.

    Parameters
    ----------
    path : str
       path to the config file
    *sections : str
       the section of the config file to load
       the sections can be nested, e.g. "section", "subsections"

    Returns
    -------
    options : dict
        dictionary of options

    """
    config = get_config(path)

    options = {}

    # traverse the sections
    section_data = config.copy()
    try:
        for section in sections:
            section_data = section_data[section]
    except KeyError:
        logger.warning(
            f"Section {' > '.join(sections)} not found in the configuration."
        )
        return {}
    for key, value in section_data.items():
        options[key] = _config_reference(config, value, *sections)

    return options


def _config_reference(config, value, *sections):
    """
    Get the value of references in config.

    Parameters
    ----------
    config : ConfigParser
        the config parser
    value : any
        The value in the section
    *sections : str
        the section of the config file to load
        the sections can be nested, e.g. "section", "subsections"

    Returns
    -------
    value : str
        the value of the option

    Notes
    -----
    There are certain special characters
    `static`: reference to static section
    `$`: reference to an environment variable

    """
    if isinstance(value, str):
        value = value.strip()
        # Handle 'static' references
        if value.startswith("static."):
            config_static = config
            if len(sections) > 1:
                for section in sections[:-1]:
                    config_static = config_static[section]
            config_static = config_static.get("static", {})
            ref_option = value[len("static.") :]
            ref_value = config_static.get(ref_option)
            if ref_value is not None:
                return ref_value
            else:
                logger.warning(f"Static reference `static.{ref_option}` not found.")
                return None
        # Handle environment variables
        elif value.startswith("$"):
            env_var = value[1:]
            env_value = os.getenv(env_var)
            if env_value is not None:
                return env_value
            else:
                logger.warning(f"Environment variable {env_var} not found.")
                return None
        # Handle None values
        elif value.lower() in ["", "none", "null"]:
            return None
        else:
            return value
    else:
        # If the value is not a string, return it as is
        return value


config_file = os.path.join(helpers.FILES_PATH, "config.yaml")

# integrations
_integrations_config = get_config_options(config_file, "config", "integrations")
HMD_EMAIL = _integrations_config.get("hmd_email", None)
HMD_PASSWORD = _integrations_config.get("hmd_password", None)
