from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SlackAppConfig(BaseSettings):
    """Configuration for the Slack app."""

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Slack API tokens
    SLACK_BOT_TOKEN: str = Field(
        ..., description="The Slack bot token for API interactions", validation_alias="SLACK_BOT_TOKEN"
    )
    SLACK_APP_TOKEN: str = Field(
        ..., description="The Slack app token for Socket Mode", validation_alias="SLACK_APP_TOKEN"
    )

    # Bot configuration
    application_name: str = Field("slack", description="Identifier for the application")
    intro_message: str = Field(
        "Hi <@{user}>! I'm DocsBot. I can help answer questions about W&B, Weave, and CoreWeave. How can I help you today?",
        description="Intro message for new threads",
    )
    outro_message: str = Field(
        "\n\n_React with 👍 or 👎 to provide feedback on this response._", description="Message to append to responses"
    )
    database_uri: str = Field(
        ..., description="Database URI for the Slack app", validation_alias="SLACK_BOT_DATABASE_URI"
    )
