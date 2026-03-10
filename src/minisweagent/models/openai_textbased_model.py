from minisweagent.models.openai_model import OpenAIModel, OpenAIModelConfig
from minisweagent.models.utils.actions_text import format_observation_messages, parse_regex_actions


class OpenAITextbasedModelConfig(OpenAIModelConfig):
    action_regex: str = r"```mswea_bash_command\s*\n(.*?)\n```"
    """Regex to extract the action from the LM's output."""
    format_error_template: str = (
        "Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions."
    )
    """Template used when the LM's output is not in the expected format."""


class OpenAITextbasedModel(OpenAIModel):
    def __init__(self, **kwargs):
        super().__init__(config_class=OpenAITextbasedModelConfig, **kwargs)

    def _query(self, messages: list[dict], **kwargs):
        extra_kwargs = {}
        if self.config.seed is not None:
            extra_kwargs["seed"] = self.config.seed
        return self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            **(extra_kwargs | self.config.model_kwargs | kwargs),
        )

    def _parse_actions(self, response) -> list[dict]:
        content = response.choices[0].message.content or ""
        return parse_regex_actions(
            content, action_regex=self.config.action_regex, format_error_template=self.config.format_error_template
        )

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        return format_observation_messages(
            outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )
