import os
from typing import Literal, Optional, Tuple

from litellm.llms.base_llm.chat.transformation import BaseLLMException


class DatabricksException(BaseLLMException):
    pass


class DatabricksBase:
    def _configure_databricks_user_agent(
        self,
        litellm_params: Optional[dict] = None,
        optional_params: Optional[dict] = None,
    ) -> None:
        """
        Configure Databricks SDK User-Agent for partner attribution.

        Follows ISV integration best practices by setting User-Agent metadata.
        Reference: https://github.com/databricks/databricks-sdk-py#user-agent-request-attribution

        Priority order (specific to general):
        1. Per-request metadata (highest priority)
        2. Model-level litellm_params
        3. Global litellm_settings
        4. Environment variables (lowest priority, handled by SDK)

        Args:
            litellm_params: LiteLLM parameters containing metadata and settings
            optional_params: Optional parameters that may contain databricks-specific config
        """
        partner = None
        product = None
        version = None

        # Priority 1: Check metadata parameter (per-request)
        if litellm_params:
            metadata = litellm_params.get("metadata", {})
            if metadata:
                partner = metadata.get("databricks_partner")
                product = metadata.get("databricks_product")
                version = metadata.get("databricks_product_version")

        # Priority 2: Check optional_params (model-level config)
        if not partner and optional_params:
            partner = optional_params.get("databricks_partner")
            product = optional_params.get("databricks_product")
            version = optional_params.get("databricks_product_version")

        # Priority 3: Check litellm_settings (global proxy config)
        if not partner and litellm_params:
            settings = litellm_params.get("litellm_settings", {})
            if settings:
                partner = settings.get("databricks_partner")
                product = settings.get("databricks_product")
                version = settings.get("databricks_product_version")

        # Priority 4: Environment variables (handled automatically by SDK)
        # Only set env vars if we have values and they're not already set
        # This allows env vars to act as the lowest-priority fallback
        if partner and not os.getenv("DATABRICKS_SDK_UPSTREAM"):
            os.environ["DATABRICKS_SDK_UPSTREAM"] = partner

        if product and version:
            # Combine product name and version for DATABRICKS_SDK_UPSTREAM_VERSION
            # Format: product/version (e.g., "agentic-gis/1.0.0")
            upstream_version = f"{product}/{version}"
            if not os.getenv("DATABRICKS_SDK_UPSTREAM_VERSION"):
                os.environ["DATABRICKS_SDK_UPSTREAM_VERSION"] = upstream_version
        elif version and not os.getenv("DATABRICKS_SDK_UPSTREAM_VERSION"):
            # If only version is provided, use it directly
            os.environ["DATABRICKS_SDK_UPSTREAM_VERSION"] = version

    def _get_api_base(self, api_base: Optional[str]) -> str:
        if api_base is None:
            try:
                from databricks.sdk import WorkspaceClient

                databricks_client = WorkspaceClient()

                api_base = (
                    api_base or f"{databricks_client.config.host}/serving-endpoints"
                )

                return api_base
            except ImportError:
                raise DatabricksException(
                    status_code=400,
                    message=(
                        "Either set the DATABRICKS_API_BASE and DATABRICKS_API_KEY environment variables, "
                        "or install the databricks-sdk Python library."
                    ),
                )
        return api_base

    def _get_databricks_credentials(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        headers: Optional[dict],
        litellm_params: Optional[dict] = None,
        optional_params: Optional[dict] = None,
    ) -> Tuple[str, dict]:
        headers = headers or {"Content-Type": "application/json"}
        try:
            # Configure User-Agent for partner attribution before initializing SDK
            self._configure_databricks_user_agent(
                litellm_params=litellm_params, optional_params=optional_params
            )

            from databricks.sdk import WorkspaceClient

            databricks_client = WorkspaceClient()

            api_base = api_base or f"{databricks_client.config.host}/serving-endpoints"

            if api_key is None:
                databricks_auth_headers: dict[
                    str, str
                ] = databricks_client.config.authenticate()
                headers = {**databricks_auth_headers, **headers}

            return api_base, headers
        except ImportError:
            raise DatabricksException(
                status_code=400,
                message=(
                    "If the Databricks base URL and API key are not set, the databricks-sdk "
                    "Python library must be installed. Please install the databricks-sdk, set "
                    "{LLM_PROVIDER}_API_BASE and {LLM_PROVIDER}_API_KEY environment variables, "
                    "or provide the base URL and API key as arguments."
                ),
            )

    def databricks_validate_environment(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        endpoint_type: Literal["chat_completions", "embeddings"],
        custom_endpoint: Optional[bool],
        headers: Optional[dict],
        litellm_params: Optional[dict] = None,
        optional_params: Optional[dict] = None,
    ) -> Tuple[str, dict]:
        if api_key is None and not headers:  # handle empty headers
            if custom_endpoint is True:
                raise DatabricksException(
                    status_code=400,
                    message="Missing API Key - A call is being made to LLM Provider but no key is set either in the environment variables ({LLM_PROVIDER}_API_KEY) or via params",
                )
            else:
                api_base, headers = self._get_databricks_credentials(
                    api_base=api_base,
                    api_key=api_key,
                    headers=headers,
                    litellm_params=litellm_params,
                    optional_params=optional_params,
                )

        if api_base is None:
            if custom_endpoint:
                raise DatabricksException(
                    status_code=400,
                    message="Missing API Base - A call is being made to LLM Provider but no api base is set either in the environment variables ({LLM_PROVIDER}_API_KEY) or via params",
                )
            else:
                api_base, headers = self._get_databricks_credentials(
                    api_base=api_base,
                    api_key=api_key,
                    headers=headers,
                    litellm_params=litellm_params,
                    optional_params=optional_params,
                )

        if headers is None:
            headers = {
                "Authorization": "Bearer {}".format(api_key),
                "Content-Type": "application/json",
            }
        else:
            if api_key is not None:
                headers.update({"Authorization": "Bearer {}".format(api_key)})

        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"

        if endpoint_type == "chat_completions" and custom_endpoint is not True:
            api_base = "{}/chat/completions".format(api_base)
        elif endpoint_type == "embeddings" and custom_endpoint is not True:
            api_base = "{}/embeddings".format(api_base)
        return api_base, headers
