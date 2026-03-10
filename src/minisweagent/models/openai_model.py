import logging
import time
from collections.abc import Callable
from typing import Any

import openai
from pydantic import BaseModel

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content
from minisweagent.models.utils.retry import retry

logger = logging.getLogger("openai_model")


class OpenAIModelConfig(BaseModel):
    model_name: str
    base_url: str
    """OpenAI-compatible base URL, e.g. http://localhost:8000/v1"""
    api_key: str = "none"
    """API key (use 'none' for local servers that don't require auth)"""
    seed: int | None = None
    """Random seed for deterministic generation (e.g. with llama.cpp). Omitted from request if None."""
    model_kwargs: dict[str, Any] = {}
    format_error_template: str = "{{ error }}"
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    multimodal_regex: str = ""


class OpenAIModel:
    """Model that uses the openai Python client against any OpenAI-compatible endpoint.

    Designed for offline use on HPC clusters (e.g. DRAC) where litellm is unavailable
    but the openai package is. Point base_url at a local vLLM or llama-server instance.
    """

    abort_exceptions: list[type[Exception]] = [
        openai.AuthenticationError,
        openai.NotFoundError,
        openai.PermissionDeniedError,
        openai.BadRequestError,
        KeyboardInterrupt,
    ]

    def __init__(self, *, config_class: Callable = OpenAIModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.client = openai.OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )

    def _query(self, messages: list[dict], **kwargs):
        extra_kwargs = {}
        if self.config.seed is not None:
            extra_kwargs["seed"] = self.config.seed
        return self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            tools=[BASH_TOOL],
            **(extra_kwargs | self.config.model_kwargs | kwargs),
        )

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        return [{k: v for k, v in msg.items() if k != "extra" and v is not None} for msg in messages]

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(self._prepare_messages_for_api(messages), **kwargs)
        GLOBAL_MODEL_STATS.add(0.0)  # local server — no cost tracking
        message = response.choices[0].message.model_dump()
        message["extra"] = {
            "actions": self._parse_actions(response),
            "cost": 0.0,
            "timestamp": time.time(),
            "response": response.model_dump(),
        }
        return message

    def _parse_actions(self, response) -> list[dict]:
        tool_calls = response.choices[0].message.tool_calls or []
        return parse_toolcall_actions(tool_calls, format_error_template=self.config.format_error_template)

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }
