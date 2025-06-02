"""CLI application for DocsBot."""

import uuid

from rich.prompt import Prompt

from docsbot.apps.cli.config import CLIBotConfig
from docsbot.apps.cli.formatter import CLIFormatter
from docsbot.chat.chat import ChatApp
from docsbot.chat.session import PersistentChatSession
from docsbot.storage.client import SQLAlchemyDBClient
from docsbot.utils import get_logger

logger  = get_logger(__name__)




class CLIBot:
    """CLI application for interacting with DocsBot."""

    def __init__(self, config: CLIBotConfig = None, user_id: str | None = None):
        """Initialize the CLI application.

        Args:
            config: Application configuration
            user_id: Optional user identifier, generated if not provided
        """
        self.config = config or CLIBotConfig()
        self.user_id = user_id or f"cli-user-{uuid.uuid4().hex[:8]}"

        # Session ID is generated for each app restart
        self.session_id = f"cli-session-{uuid.uuid4().hex}"

        # Initialize database client
        self.db_client = SQLAlchemyDBClient(database_url=self.config.database_uri)

        # Initialize formatter
        self.formatter = CLIFormatter()

        # Initialize chat app
        self.chat_app = ChatApp()

    async def save_session_metadata(self):
        """Save the session metadata to the database."""
        try:
            await self.db_client.save_session_metadata(
                session_id=self.session_id, platform=self.config.application_name, user_id=self.user_id
            )
        except Exception as e:
            logger.exception(f"Error saving session metadata: {e}", stacklevel=2)

    async def run(self):
        """Run the CLI application."""
        self.formatter.print_info(f"Starting session as user: {self.user_id}")
        self.formatter.print_info(f"Session ID: {self.session_id}")
        self.formatter.print_info(self.config.intro_message)

        # Save initial session data
        await self.save_session_metadata()

        # Initialize persistent chat session
        async with PersistentChatSession(
            chat_app=self.chat_app, session_id=self.session_id, db_client=self.db_client
        ) as session:
            try:
                while session.is_active:
                    # Get user input
                    user_input = Prompt.ask("[bold blue]User[/bold blue]")
                    print("\033[F\033[K", end="")  # Clear the input line

                    # Process the input
                    response = await session.process_message(user_input)

                    # Check if we're exiting
                    if not session.is_active:
                        self.formatter.print_info(self.config.exit_message)
                        self.formatter.print_conversation_history(session.history)
                        break

                    # Format and display the response
                    formatted_response = self.formatter.format_assistant_response(response)
                    self.formatter.print_assistant_message(formatted_response)

            except KeyboardInterrupt:
                self.formatter.print_info("\nKeyboard interrupt detected. Conversation history:")
                self.formatter.print_conversation_history(session.history)
            except Exception as e:
                self.formatter.print_error(str(e))

    async def get_previous_sessions(self):
        """Get previous sessions for the current user.

        Returns:
            A list of session IDs
        """
        try:
            return await self.db_client.get_session_by_user(user_id=self.user_id, platform=self.config.application_name)
        except Exception as e:
            logger.exception(f"Error getting previous sessions: {e}", stacklevel=2)
            return []

    async def get_user_profile(self):
        """Get the user profile with usage statistics.

        Returns:
            User profile data
        """
        try:
            return await self.db_client.get_user_profile(user_id=self.user_id, platform=self.config.application_name)
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return {"user_id": self.user_id, "platform": self.config.application_name}
