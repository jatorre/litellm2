import os
import pytest
from litellm.llms.snowflake.common_utils import SnowflakeBase


class TestSnowflakePartnerAttribution:
    """Test suite for Snowflake ISV partner attribution configuration"""

    def setup_method(self):
        """Clear environment variables before each test"""
        os.environ.pop("SNOWFLAKE_PARTNER", None)
        os.environ.pop("SNOWFLAKE_PRODUCT", None)
        os.environ.pop("SNOWFLAKE_PRODUCT_VERSION", None)

    def teardown_method(self):
        """Clean up environment variables after each test"""
        os.environ.pop("SNOWFLAKE_PARTNER", None)
        os.environ.pop("SNOWFLAKE_PRODUCT", None)
        os.environ.pop("SNOWFLAKE_PRODUCT_VERSION", None)

    def test_priority_1_per_request_metadata(self):
        """Test Priority 1: Per-request metadata (highest priority)"""
        snowflake_base = SnowflakeBase()

        litellm_params = {
            "metadata": {
                "snowflake_partner": "carto",
                "snowflake_product": "agentic-gis",
                "snowflake_product_version": "1.0.0",
            }
        }

        user_agent = snowflake_base._configure_snowflake_partner_attribution(
            litellm_params=litellm_params
        )

        assert user_agent == "carto/agentic-gis-1.0.0"

    def test_priority_2_model_level_params(self):
        """Test Priority 2: Model-level optional_params"""
        snowflake_base = SnowflakeBase()

        optional_params = {
            "snowflake_partner": "carto",
            "snowflake_product": "agentic-gis",
            "snowflake_product_version": "2.0.0",
        }

        user_agent = snowflake_base._configure_snowflake_partner_attribution(
            optional_params=optional_params
        )

        assert user_agent == "carto/agentic-gis-2.0.0"

    def test_priority_3_global_settings(self):
        """Test Priority 3: Global litellm_settings"""
        snowflake_base = SnowflakeBase()

        litellm_params = {
            "litellm_settings": {
                "snowflake_partner": "carto",
                "snowflake_product": "agentic-gis",
                "snowflake_product_version": "3.0.0",
            }
        }

        user_agent = snowflake_base._configure_snowflake_partner_attribution(
            litellm_params=litellm_params
        )

        assert user_agent == "carto/agentic-gis-3.0.0"

    def test_priority_4_environment_variables(self):
        """Test Priority 4: Environment variables (lowest priority)"""
        snowflake_base = SnowflakeBase()

        # Set environment variables
        os.environ["SNOWFLAKE_PARTNER"] = "carto"
        os.environ["SNOWFLAKE_PRODUCT"] = "agentic-gis"
        os.environ["SNOWFLAKE_PRODUCT_VERSION"] = "4.0.0"

        user_agent = snowflake_base._configure_snowflake_partner_attribution()

        assert user_agent == "carto/agentic-gis-4.0.0"

    def test_priority_order_metadata_wins(self):
        """Test that per-request metadata (Priority 1) overrides all other sources"""
        snowflake_base = SnowflakeBase()

        # Set all levels with different values
        os.environ["SNOWFLAKE_PARTNER"] = "env-partner"
        os.environ["SNOWFLAKE_PRODUCT"] = "env-product"
        os.environ["SNOWFLAKE_PRODUCT_VERSION"] = "env-version"

        litellm_params = {
            "metadata": {
                "snowflake_partner": "metadata-partner",
                "snowflake_product": "metadata-product",
                "snowflake_product_version": "metadata-version",
            },
            "litellm_settings": {
                "snowflake_partner": "settings-partner",
                "snowflake_product": "settings-product",
                "snowflake_product_version": "settings-version",
            },
        }

        optional_params = {
            "snowflake_partner": "optional-partner",
            "snowflake_product": "optional-product",
            "snowflake_product_version": "optional-version",
        }

        user_agent = snowflake_base._configure_snowflake_partner_attribution(
            litellm_params=litellm_params,
            optional_params=optional_params,
        )

        # Metadata should win
        assert user_agent == "metadata-partner/metadata-product-metadata-version"

    def test_priority_order_optional_params_over_settings(self):
        """Test that model-level params (Priority 2) override global settings (Priority 3)"""
        snowflake_base = SnowflakeBase()

        litellm_params = {
            "litellm_settings": {
                "snowflake_partner": "settings-partner",
                "snowflake_product": "settings-product",
                "snowflake_product_version": "settings-version",
            }
        }

        optional_params = {
            "snowflake_partner": "optional-partner",
            "snowflake_product": "optional-product",
            "snowflake_product_version": "optional-version",
        }

        user_agent = snowflake_base._configure_snowflake_partner_attribution(
            litellm_params=litellm_params,
            optional_params=optional_params,
        )

        # Optional params should win
        assert user_agent == "optional-partner/optional-product-optional-version"

    def test_priority_order_settings_over_env(self):
        """Test that global settings (Priority 3) override environment variables (Priority 4)"""
        snowflake_base = SnowflakeBase()

        os.environ["SNOWFLAKE_PARTNER"] = "env-partner"
        os.environ["SNOWFLAKE_PRODUCT"] = "env-product"
        os.environ["SNOWFLAKE_PRODUCT_VERSION"] = "env-version"

        litellm_params = {
            "litellm_settings": {
                "snowflake_partner": "settings-partner",
                "snowflake_product": "settings-product",
                "snowflake_product_version": "settings-version",
            }
        }

        user_agent = snowflake_base._configure_snowflake_partner_attribution(
            litellm_params=litellm_params
        )

        # Settings should win
        assert user_agent == "settings-partner/settings-product-settings-version"

    def test_no_configuration_returns_none(self):
        """Test that None is returned when no configuration is provided"""
        snowflake_base = SnowflakeBase()

        user_agent = snowflake_base._configure_snowflake_partner_attribution()

        assert user_agent is None

    def test_partner_only(self):
        """Test User-Agent with only partner configured"""
        snowflake_base = SnowflakeBase()

        litellm_params = {
            "metadata": {
                "snowflake_partner": "carto",
            }
        }

        user_agent = snowflake_base._configure_snowflake_partner_attribution(
            litellm_params=litellm_params
        )

        assert user_agent == "carto"

    def test_partner_and_product_only(self):
        """Test User-Agent with partner and product but no version"""
        snowflake_base = SnowflakeBase()

        litellm_params = {
            "metadata": {
                "snowflake_partner": "carto",
                "snowflake_product": "agentic-gis",
            }
        }

        user_agent = snowflake_base._configure_snowflake_partner_attribution(
            litellm_params=litellm_params
        )

        assert user_agent == "carto/agentic-gis"

    def test_partner_and_version_only(self):
        """Test User-Agent with partner and version but no product"""
        snowflake_base = SnowflakeBase()

        litellm_params = {
            "metadata": {
                "snowflake_partner": "carto",
                "snowflake_product_version": "1.0.0",
            }
        }

        user_agent = snowflake_base._configure_snowflake_partner_attribution(
            litellm_params=litellm_params
        )

        assert user_agent == "carto/1.0.0"

    def test_complete_configuration_all_fields(self):
        """Test User-Agent with all fields configured"""
        snowflake_base = SnowflakeBase()

        litellm_params = {
            "metadata": {
                "snowflake_partner": "carto",
                "snowflake_product": "agentic-gis",
                "snowflake_product_version": "1.0.0",
            }
        }

        user_agent = snowflake_base._configure_snowflake_partner_attribution(
            litellm_params=litellm_params
        )

        assert user_agent == "carto/agentic-gis-1.0.0"

    def test_empty_metadata_dict(self):
        """Test that empty metadata dict returns None"""
        snowflake_base = SnowflakeBase()

        litellm_params = {"metadata": {}}

        user_agent = snowflake_base._configure_snowflake_partner_attribution(
            litellm_params=litellm_params
        )

        assert user_agent is None

    def test_empty_optional_params_dict(self):
        """Test that empty optional_params dict returns None"""
        snowflake_base = SnowflakeBase()

        optional_params = {}

        user_agent = snowflake_base._configure_snowflake_partner_attribution(
            optional_params=optional_params
        )

        assert user_agent is None

    def test_env_var_partial_config(self):
        """Test environment variables with partial configuration"""
        snowflake_base = SnowflakeBase()

        os.environ["SNOWFLAKE_PARTNER"] = "carto"
        os.environ["SNOWFLAKE_PRODUCT"] = "agentic-gis"
        # No version set

        user_agent = snowflake_base._configure_snowflake_partner_attribution()

        assert user_agent == "carto/agentic-gis"
