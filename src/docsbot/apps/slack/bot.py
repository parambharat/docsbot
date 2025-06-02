"""Slack bot implementation using DocsBot."""

import asyncio
import contextlib
import re
from functools import partial

from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.web import SlackResponse

from docsbot.apps.slack.config import SlackAppConfig
from docsbot.apps.slack.formatter import MrkdwnFormatter
from docsbot.chat.chat import ChatApp
from docsbot.chat.session import ThreadSessionManager
from docsbot.storage.client import SQLAlchemyDBClient
from docsbot.utils import get_logger

logger  = get_logger(__name__)




class SlackBot:
    """Slack bot implementation using DocsBot."""

    def __init__(self, config: SlackAppConfig = None):
        """Initialize the Slack bot.

        Args:
            config: Configuration for the Slack app
            database_url: SQLAlchemy connection URL for the database
        """
        self.config = config or SlackAppConfig()

        # Initialize Slack app
        self.app = AsyncApp(token=self.config.SLACK_BOT_TOKEN)

        # Initialize database client with SQLAlchemy
        self.db_client = SQLAlchemyDBClient(database_url=self.config.database_uri)

        # Initialize session manager with ChatApp
        self.session_manager = ThreadSessionManager(chat_app=ChatApp(), db_client=self.db_client)

        # Register event handlers
        self._register_handlers()

        # Initialize formatter
        self.formatter = MrkdwnFormatter()

    def _register_handlers(self):
        """Register event handlers for Slack events."""
        self.app.event("app_mention")(self.handle_mention)
        self.app.event("reaction_added")(self.handle_reaction)

    async def send_message(self, say, message: str, thread: str = None) -> SlackResponse:
        """Send a message to Slack.

        Args:
            say: Slack say function
            message: The message text
            thread: Thread timestamp for replies

        Returns:
            Slack response
        """
        formatted_message = self.formatter(message)

        if thread is not None:
            return await say(text=formatted_message, thread_ts=thread)
        else:
            return await say(text=formatted_message)

    def _normalize_thread_id(self, thread_ts, ts):
        """Normalize thread ID to ensure consistent handling.

        Returns the thread_ts if available, otherwise the ts.
        """
        result = str(thread_ts or ts)
        logger.debug(f"Normalizing thread_id: thread_ts={thread_ts}, ts={ts}, result={result}")
        return result

    async def handle_mention(self, body, say, logger):
        """Handle when the bot is mentioned.

        Args:
            body: The event body
            say: Function to send messages
            logger: Logger instance
        """
        try:
            # Extract information
            query = body["event"].get("text", "")
            user = body["event"].get("user")

            # Clean up bot mentions from the query
            # Replace <@BOT_ID>: or <@BOT_ID> with the bot name
            query = re.sub(r"<@[A-Z0-9]+>:?\s*", "docsbot: ", query).strip()

            # Ensure consistent thread ID handling
            thread_ts = body["event"].get("thread_ts")
            message_ts = body["event"].get("ts")
            logger.warning(f"Original thread_ts: {thread_ts}, message_ts: {message_ts}")
            thread_id = self._normalize_thread_id(thread_ts, message_ts)
            logger.warning(f"Normalized thread_id: {thread_id}")

            channel_id = body["event"].get("channel")

            logger.debug(f"Handling mention in thread: {thread_id}")

            # Bind token to say function
            say = partial(say, token=self.config.SLACK_BOT_TOKEN)

            # Check if this is a new thread
            conversation = await self.db_client.get_conversation(thread_id)
            if not conversation:
                # Send intro message for new threads
                await self.send_message(say=say, message=self.config.intro_message.format(user=user), thread=thread_id)

                # Update session with user information for tracking
                try:
                    await self.db_client.save_session_metadata(
                        session_id=thread_id, platform="slack", user_id=user, channel_id=channel_id
                    )
                except Exception as e:
                    logger.exception(f"Error saving session metadata: {e}", stacklevel=2)

            # Process the query
            response = await self.session_manager.process_message(
                thread_id, query, user_info={"user_id": user, "platform": "slack", "channel_id": channel_id}
            )

            # Format response with citations
            formatted_response = self.formatter.format_structured_response(response)

            # Add outro message for feedback
            full_response = formatted_response + self.config.outro_message

            # Send the response
            sent_message = await self.send_message(say=say, message=full_response, thread=thread_id)

            # Add reactions for feedback
            await self.app.client.reactions_add(
                channel=channel_id,
                timestamp=sent_message["ts"],
                name="thumbsup",
                token=self.config.SLACK_BOT_TOKEN,
            )
            await self.app.client.reactions_add(
                channel=channel_id,
                timestamp=sent_message["ts"],
                name="thumbsdown",
                token=self.config.SLACK_BOT_TOKEN,
            )

            # Save message ID for linking feedback
            try:
                await self.db_client.update_message_id(
                    session_id=thread_id, content=response, message_id=sent_message["ts"]
                )
            except Exception as e:
                logger.exception(f"Error updating message ID: {e}", stacklevel=2)

        except Exception as e:
            logger.exception(f"Error handling mention: {e}", stacklevel=2)

    async def handle_reaction(self, event, logger):
        """Handle reactions added to messages.

        Args:
            event: The reaction event data
            logger: Logger instance
        """
        try:
            # Extract information
            reaction = event.get("reaction", "")
            message_id = event.get("item", {}).get("ts", "")
            thread_ts = event.get("item", {}).get("thread_ts", "")
            thread_id = self._normalize_thread_id(thread_ts, message_id)
            user_id = event.get("user", "")

            # Only process thumbs up/down
            if reaction not in ["thumbsup", "thumbsdown"]:
                return

            # Convert to rating
            rating = 1 if reaction == "thumbsup" else -1

            # Save the feedback
            await self.db_client.save_feedback(
                session_id=thread_id, message_id=message_id, rating=rating, user_id=user_id
            )

            logger.debug(f"Saved feedback: {rating} for message {message_id}", stacklevel=2)

        except Exception as e:
            logger.exception(f"Error handling reaction: {e}", stacklevel=2)

    async def cleanup_inactive_sessions(self):
        """Periodically clean up inactive sessions."""
        while True:
            try:
                await self.session_manager.cleanup_inactive_sessions()
            except Exception as e:
                logger.exception(f"Error cleaning up sessions: {e}", stacklevel=2)

            # Run cleanup every 10 minutes
            await asyncio.sleep(600)

    async def start(self):
        """Start the Slack bot."""
        handler = AsyncSocketModeHandler(self.app, self.config.SLACK_APP_TOKEN)

        # Start the cleanup task
        cleanup_task = asyncio.create_task(self.cleanup_inactive_sessions())

        try:
            # Start the bot
            await handler.start_async()
        finally:
            # Cancel cleanup when bot stops
            cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cleanup_task
