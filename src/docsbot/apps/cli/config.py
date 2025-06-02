"""Configuration for the CLI app."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CLIBotConfig(BaseSettings):
    """Configuration for the CLI app."""

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # CLI app settings
    application_name: str = Field("cli", description="Identifier for the application")
    database_uri: str = Field(..., description="Database URL for persistence", validation_alias="CLI_BOT_DATABASE_URI")

    # UI messages
    intro_message: str = Field(
        "Welcome to DocsBot CLI. I can help answer questions about W&B, Weave, and CoreWeave.\nType 'exit' to quit.",
        description="Introduction message shown at startup",
    )
    exit_message: str = Field("Exiting... Here's your conversation history:", description="Message shown when exiting")
