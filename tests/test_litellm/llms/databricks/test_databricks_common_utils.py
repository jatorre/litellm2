import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path
from unittest.mock import MagicMock, patch

from litellm.llms.databricks.common_utils import DatabricksBase


def test_databricks_validate_environment():
    databricks_base = DatabricksBase()

    with patch.object(
        databricks_base, "_get_databricks_credentials"
    ) as mock_get_credentials:
        try:
            databricks_base.databricks_validate_environment(
                api_key=None,
                api_base="my_api_base",
                endpoint_type="chat_completions",
                custom_endpoint=False,
                headers=None,
            )
        except Exception:
            pass
        mock_get_credentials.assert_called_once()


def test_configure_user_agent_priority_1_metadata():
    """Test that per-request metadata has highest priority"""
    databricks_base = DatabricksBase()

    # Clear any existing env vars
    os.environ.pop("DATABRICKS_SDK_UPSTREAM", None)
    os.environ.pop("DATABRICKS_SDK_UPSTREAM_VERSION", None)

    litellm_params = {
        "metadata": {
            "databricks_partner": "carto",
            "databricks_product": "agentic-gis",
            "databricks_product_version": "1.0.0",
        },
        "litellm_settings": {
            "databricks_partner": "other-partner",  # Should be ignored
        },
    }

    optional_params = {
        "databricks_partner": "another-partner",  # Should be ignored
    }

    databricks_base._configure_databricks_user_agent(
        litellm_params=litellm_params, optional_params=optional_params
    )

    # Verify metadata values were used (highest priority)
    assert os.environ.get("DATABRICKS_SDK_UPSTREAM") == "carto"
    assert os.environ.get("DATABRICKS_SDK_UPSTREAM_VERSION") == "agentic-gis/1.0.0"

    # Cleanup
    os.environ.pop("DATABRICKS_SDK_UPSTREAM", None)
    os.environ.pop("DATABRICKS_SDK_UPSTREAM_VERSION", None)


def test_configure_user_agent_priority_2_optional_params():
    """Test that optional_params is used when metadata is not present"""
    databricks_base = DatabricksBase()

    # Clear any existing env vars
    os.environ.pop("DATABRICKS_SDK_UPSTREAM", None)
    os.environ.pop("DATABRICKS_SDK_UPSTREAM_VERSION", None)

    litellm_params = {
        "litellm_settings": {
            "databricks_partner": "other-partner",  # Should be ignored
        },
    }

    optional_params = {
        "databricks_partner": "carto",
        "databricks_product": "agentic-gis",
        "databricks_product_version": "2.0.0",
    }

    databricks_base._configure_databricks_user_agent(
        litellm_params=litellm_params, optional_params=optional_params
    )

    # Verify optional_params values were used
    assert os.environ.get("DATABRICKS_SDK_UPSTREAM") == "carto"
    assert os.environ.get("DATABRICKS_SDK_UPSTREAM_VERSION") == "agentic-gis/2.0.0"

    # Cleanup
    os.environ.pop("DATABRICKS_SDK_UPSTREAM", None)
    os.environ.pop("DATABRICKS_SDK_UPSTREAM_VERSION", None)


def test_configure_user_agent_priority_3_litellm_settings():
    """Test that litellm_settings is used when higher priorities are not present"""
    databricks_base = DatabricksBase()

    # Clear any existing env vars
    os.environ.pop("DATABRICKS_SDK_UPSTREAM", None)
    os.environ.pop("DATABRICKS_SDK_UPSTREAM_VERSION", None)

    litellm_params = {
        "litellm_settings": {
            "databricks_partner": "carto",
            "databricks_product": "agentic-gis",
            "databricks_product_version": "3.0.0",
        },
    }

    databricks_base._configure_databricks_user_agent(
        litellm_params=litellm_params, optional_params=None
    )

    # Verify litellm_settings values were used
    assert os.environ.get("DATABRICKS_SDK_UPSTREAM") == "carto"
    assert os.environ.get("DATABRICKS_SDK_UPSTREAM_VERSION") == "agentic-gis/3.0.0"

    # Cleanup
    os.environ.pop("DATABRICKS_SDK_UPSTREAM", None)
    os.environ.pop("DATABRICKS_SDK_UPSTREAM_VERSION", None)


def test_configure_user_agent_priority_4_env_vars_preserved():
    """Test that existing environment variables are not overridden"""
    databricks_base = DatabricksBase()

    # Set env vars first
    os.environ["DATABRICKS_SDK_UPSTREAM"] = "existing-partner"
    os.environ["DATABRICKS_SDK_UPSTREAM_VERSION"] = "existing-version"

    litellm_params = {
        "metadata": {
            "databricks_partner": "carto",
            "databricks_product": "agentic-gis",
            "databricks_product_version": "1.0.0",
        },
    }

    databricks_base._configure_databricks_user_agent(
        litellm_params=litellm_params, optional_params=None
    )

    # Verify existing env vars were NOT overridden
    assert os.environ.get("DATABRICKS_SDK_UPSTREAM") == "existing-partner"
    assert os.environ.get("DATABRICKS_SDK_UPSTREAM_VERSION") == "existing-version"

    # Cleanup
    os.environ.pop("DATABRICKS_SDK_UPSTREAM", None)
    os.environ.pop("DATABRICKS_SDK_UPSTREAM_VERSION", None)


def test_configure_user_agent_no_config():
    """Test that nothing happens when no configuration is provided"""
    databricks_base = DatabricksBase()

    # Clear any existing env vars
    os.environ.pop("DATABRICKS_SDK_UPSTREAM", None)
    os.environ.pop("DATABRICKS_SDK_UPSTREAM_VERSION", None)

    databricks_base._configure_databricks_user_agent(
        litellm_params=None, optional_params=None
    )

    # Verify no env vars were set
    assert os.environ.get("DATABRICKS_SDK_UPSTREAM") is None
    assert os.environ.get("DATABRICKS_SDK_UPSTREAM_VERSION") is None


def test_configure_user_agent_version_only():
    """Test that version-only configuration works"""
    databricks_base = DatabricksBase()

    # Clear any existing env vars
    os.environ.pop("DATABRICKS_SDK_UPSTREAM", None)
    os.environ.pop("DATABRICKS_SDK_UPSTREAM_VERSION", None)

    litellm_params = {
        "metadata": {
            "databricks_partner": "carto",
            "databricks_product_version": "standalone-version",
        },
    }

    databricks_base._configure_databricks_user_agent(
        litellm_params=litellm_params, optional_params=None
    )

    # Verify partner and version-only were set
    assert os.environ.get("DATABRICKS_SDK_UPSTREAM") == "carto"
    assert os.environ.get("DATABRICKS_SDK_UPSTREAM_VERSION") == "standalone-version"

    # Cleanup
    os.environ.pop("DATABRICKS_SDK_UPSTREAM", None)
    os.environ.pop("DATABRICKS_SDK_UPSTREAM_VERSION", None)
