import os
from typing import Optional


class SnowflakeBase:
    def _configure_snowflake_partner_attribution(
        self,
        litellm_params: Optional[dict] = None,
        optional_params: Optional[dict] = None,
    ) -> Optional[str]:
        """
        Configure Snowflake User-Agent for ISV partner attribution.

        Follows ISV integration best practices by setting User-Agent metadata
        to identify partner applications and track usage.

        Priority order (specific to general):
        1. Per-request metadata (highest priority)
        2. Model-level litellm_params
        3. Global litellm_settings
        4. Environment variables (lowest priority)

        Args:
            litellm_params: LiteLLM parameters containing metadata and settings
            optional_params: Optional parameters that may contain snowflake-specific config

        Returns:
            User-Agent string in format: "partner/product-version" or None if not configured
        """
        partner = None
        product = None
        version = None

        # Priority 1: Check metadata parameter (per-request)
        if litellm_params:
            metadata = litellm_params.get("metadata", {})
            if metadata:
                partner = metadata.get("snowflake_partner")
                product = metadata.get("snowflake_product")
                version = metadata.get("snowflake_product_version")

        # Priority 2: Check optional_params (model-level config)
        if not partner and optional_params:
            partner = optional_params.get("snowflake_partner")
            product = optional_params.get("snowflake_product")
            version = optional_params.get("snowflake_product_version")

        # Priority 3: Check litellm_settings (global proxy config)
        if not partner and litellm_params:
            settings = litellm_params.get("litellm_settings", {})
            if settings:
                partner = settings.get("snowflake_partner")
                product = settings.get("snowflake_product")
                version = settings.get("snowflake_product_version")

        # Priority 4: Environment variables (fallback)
        if not partner:
            partner = os.getenv("SNOWFLAKE_PARTNER")
            product = os.getenv("SNOWFLAKE_PRODUCT")
            version = os.getenv("SNOWFLAKE_PRODUCT_VERSION")

        # Build User-Agent string
        if partner:
            if product and version:
                return f"{partner}/{product}-{version}"
            elif product:
                return f"{partner}/{product}"
            elif version:
                return f"{partner}/{version}"
            else:
                return partner

        return None

    def validate_environment(
        self,
        headers: dict,
        JWT: Optional[str] = None,
    ) -> dict:
        """
        Return headers to use for Snowflake completion request

        Snowflake REST API Ref: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-llm-rest-api#api-reference
        Expected headers:
        {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer " + <JWT>,
            "X-Snowflake-Authorization-Token-Type": "KEYPAIR_JWT"
        }
        """

        if JWT is None:
            raise ValueError("Missing Snowflake JWT key")

        headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": "Bearer " + JWT,
                "X-Snowflake-Authorization-Token-Type": "KEYPAIR_JWT",
            }
        )
        return headers
