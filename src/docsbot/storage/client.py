"""Database client implementation using SQLAlchemy."""

import datetime
import json
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from docsbot.storage.models import Base, ChatMessage, ChatSession, Feedback
from docsbot.utils import get_logger

logger  = get_logger(__name__)




class Database:
    """Database connection manager."""

    def __init__(self, database_url: str):
        """Initialize the database connection.

        Args:
            database_url: SQLAlchemy connection URL (e.g., sqlite:///app.db)
        """
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # Create tables if they don't exist
        Base.metadata.create_all(bind=self.engine)

    def get_session(self) -> Session:
        """Get a new database session.

        Returns:
            A new SQLAlchemy session
        """
        return self.SessionLocal()


class DBClientInterface:
    """Interface defining required methods for database clients used with PersistentChatSession."""

    async def get_conversation(self, session_id: str) -> list:
        """Retrieve conversation history for a session.

        Args:
            session_id: The unique session identifier.

        Returns:
            List of conversation messages in the format expected by ChatSession.
        """
        raise NotImplementedError

    async def save_conversation(self, session_id: str, conversation: list) -> None:
        """Save the entire conversation for a session.

        Args:
            session_id: The unique session identifier.
            conversation: List of conversation messages.
        """
        raise NotImplementedError

    async def save_message(self, session_id: str, user_input: str, response: str, timestamp: str) -> None:
        """Save a single message exchange.

        Args:
            session_id: The unique session identifier.
            user_input: The user's message text.
            response: The assistant's response.
            timestamp: ISO-format timestamp for when the message was processed.
        """
        raise NotImplementedError


class SQLAlchemyDBClient(DBClientInterface):
    """SQLAlchemy implementation of the database client interface."""

    def __init__(self, database_url: str = "sqlite:///docsbot.db"):
        """Initialize the SQLAlchemy database client.

        Args:
            database_url: SQLAlchemy connection URL
        """
        self.database = Database(database_url)

    async def get_conversation(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve conversation history for a session.

        Args:
            session_id: The unique session identifier

        Returns:
            List of conversation messages in the format expected by ChatSession
        """
        logger.debug(f"Getting conversation for session {session_id}")
        db = self.database.get_session()

        try:
            # Query the session and its messages
            chat_session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()

            if not chat_session:
                return []

            # Query messages ordered by timestamp
            messages = (
                db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.timestamp).all()
            )

            # Format messages as expected by ChatSession
            formatted_messages = []
            for msg in messages:
                formatted_messages.append(
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                    }
                )

            return formatted_messages

        except Exception as e:
            logger.exception(f"Error getting conversation: {e}", stacklevel=2)
            return []
        finally:
            db.close()

    async def save_conversation(self, session_id: str, conversation: list[dict[str, Any]]) -> None:
        """Save the entire conversation for a session.

        Args:
            session_id: The unique session identifier
            conversation: List of conversation messages
        """
        logger.debug(f"Saving conversation for session {session_id}")
        db = self.database.get_session()

        try:
            # Check if session exists
            chat_session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()

            if not chat_session:
                # Create new session if it doesn't exist
                chat_session = ChatSession(
                    session_id=session_id,
                    platform="unknown",  # Will be updated if known
                )
                db.add(chat_session)
                db.flush()

            # Clear existing messages (to be replaced with current conversation)
            db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()

            # Add all messages in the conversation
            for idx, message in enumerate(conversation):
                # Ensure content is a string
                content = message.get("content", "")
                if not isinstance(content, str):
                    # If it's a complex structure, serialize it
                    try:
                        content = json.dumps(content)
                    except (TypeError, ValueError):
                        # If it can't be serialized to JSON, use string representation
                        content = str(content)

                msg = ChatMessage(
                    session_id=session_id,
                    role=message.get("role", "unknown"),
                    content=content,
                    timestamp=(
                        datetime.datetime.fromisoformat(message.get("timestamp"))
                        if message.get("timestamp")
                        else datetime.datetime.utcnow()
                    ),
                )
                db.add(msg)

            # Update session timestamp
            chat_session.updated_at = datetime.datetime.utcnow()

            db.commit()

        except Exception as e:
            logger.exception(f"Error saving conversation: {e}", stacklevel=2)
            db.rollback()
        finally:
            db.close()

    async def save_message(self, session_id: str, user_input: str, response: str, timestamp: str) -> None:
        """Save a single message exchange.

        Args:
            session_id: The unique session identifier
            user_input: The user's message text
            response: The assistant's response (may be JSON-serialized content)
            timestamp: ISO-format timestamp for when the message was processed
        """
        logger.debug(f"Saving message exchange for session {session_id}")
        db = self.database.get_session()

        try:
            # Check if session exists
            chat_session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()

            if not chat_session:
                # Create new session if it doesn't exist
                chat_session = ChatSession(
                    session_id=session_id,
                    platform="unknown",
                )
                db.add(chat_session)
                db.flush()

            # Ensure user_input is a string
            if not isinstance(user_input, str):
                try:
                    user_input = json.dumps(user_input)
                except (TypeError, ValueError):
                    user_input = str(user_input)

            # Create user message
            user_message = ChatMessage(
                session_id=session_id,
                role="user",
                content=user_input,
                timestamp=datetime.datetime.fromisoformat(timestamp) if timestamp else datetime.datetime.utcnow(),
            )
            db.add(user_message)

            # Ensure response is a string
            if not isinstance(response, str):
                try:
                    response = json.dumps(response)
                except (TypeError, ValueError):
                    response = str(response)

            # Create assistant message - store the raw response which may be JSON
            assistant_message = ChatMessage(
                session_id=session_id,
                role="assistant",
                content=response,  # Store the raw response (may be JSON)
                timestamp=datetime.datetime.fromisoformat(timestamp) if timestamp else datetime.datetime.utcnow(),
            )
            db.add(assistant_message)

            # Update session timestamp
            chat_session.updated_at = datetime.datetime.utcnow()

            db.commit()

        except Exception as e:
            logger.exception(f"Error saving message: {e}", stacklevel=2)
            db.rollback()
        finally:
            db.close()

    async def save_feedback(
        self,
        session_id: str,
        message_id: str | None,
        rating: int,
        feedback_text: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Save feedback for a message or session.

        Args:
            session_id: The session identifier
            message_id: Optional message identifier
            rating: Numeric rating (-1, 0, 1)
            feedback_text: Optional text feedback
            user_id: User who gave the feedback
        """
        logger.debug(f"Saving feedback for session {session_id}")
        db = self.database.get_session()

        try:
            # Create feedback record
            feedback = Feedback(
                session_id=session_id,
                message_id=message_id,
                rating=rating,
                feedback_text=feedback_text,
                user_id=user_id,
                timestamp=datetime.datetime.utcnow(),
            )
            db.add(feedback)
            db.commit()

        except Exception as e:
            logger.exception(f"Error saving feedback: {e}", stacklevel=2)
            db.rollback()
        finally:
            db.close()

    async def update_message_id(self, session_id: str, content: str, message_id: str) -> None:
        """Update a message with its platform-specific ID for tracking feedback.

        Args:
            session_id: The session identifier
            content: Message content to match (may be JSON-serialized content)
            message_id: Platform-specific message ID
        """
        logger.debug(f"Updating message ID for message in session {session_id}")
        db = self.database.get_session()

        try:
            # Find the latest message from assistant in this session
            # Note: We don't try to match on content since it may be formatted differently
            # Just get the most recent assistant message in this session
            message = (
                db.query(ChatMessage)
                .filter(
                    ChatMessage.session_id == session_id,
                    ChatMessage.role == "assistant",
                )
                .order_by(ChatMessage.timestamp.desc())
                .first()
            )

            if message:
                message.message_id = message_id
                db.commit()
            else:
                logger.warning(f"No matching message found in session {session_id}")

        except Exception as e:
            logger.exception(f"Error updating message ID: {e}", stacklevel=2)
            db.rollback()
        finally:
            db.close()

    async def save_session_metadata(
        self, session_id: str, platform: str, user_id: str | None = None, channel_id: str | None = None
    ) -> None:
        """Save or update metadata for a session.

        Args:
            session_id: The session identifier
            platform: Platform name (slack, discord, cli, etc.)
            user_id: User identifier in the platform
            channel_id: Channel identifier in the platform
        """
        logger.debug(f"Saving metadata for session {session_id}")
        db = self.database.get_session()

        try:
            # Find or create session
            chat_session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()

            if not chat_session:
                chat_session = ChatSession(
                    session_id=session_id,
                    platform=platform,
                    user_id=user_id,
                )
                db.add(chat_session)
            else:
                # Update fields
                chat_session.platform = platform
                if user_id:
                    chat_session.user_id = user_id

            # Store channel_id as metadata if needed
            # In a real implementation, you might have a metadata table or field

            db.commit()

        except Exception as e:
            logger.exception(f"Error saving session metadata: {e}", stacklevel=2)
            db.rollback()
        finally:
            db.close()

    async def get_session_by_user(self, user_id: str, platform: str) -> list[str]:
        """Get all session IDs for a user.

        Args:
            user_id: The user identifier
            platform: The platform name

        Returns:
            List of session IDs
        """
        logger.debug(f"Getting sessions for user {user_id} on {platform}")
        db = self.database.get_session()

        try:
            sessions = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user_id, ChatSession.platform == platform)
                .order_by(ChatSession.updated_at.desc())
                .all()
            )

            return [session.session_id for session in sessions]

        except Exception as e:
            logger.exception(f"Error getting user sessions: {e}", stacklevel=2)
            return []
        finally:
            db.close()

    async def get_user_profile(self, user_id: str, platform: str) -> dict[str, Any]:
        """Get a user profile with summarized information from past conversations.

        This could be extended to include preferences, common topics, etc.

        Args:
            user_id: The user identifier
            platform: The platform name

        Returns:
            User profile data
        """
        logger.debug(f"Getting user profile for {user_id} on {platform}")
        db = self.database.get_session()

        try:
            # Get basic user info
            session_count = (
                db.query(ChatSession).filter(ChatSession.user_id == user_id, ChatSession.platform == platform).count()
            )

            # Get message count
            message_count = (
                db.query(ChatMessage)
                .join(ChatSession, ChatSession.session_id == ChatMessage.session_id)
                .filter(ChatSession.user_id == user_id, ChatSession.platform == platform)
                .count()
            )

            # Could be extended with more analytics or NLP-based insights
            return {
                "user_id": user_id,
                "platform": platform,
                "session_count": session_count,
                "message_count": message_count,
                "last_active": (
                    db.query(ChatSession)
                    .filter(ChatSession.user_id == user_id, ChatSession.platform == platform)
                    .order_by(ChatSession.updated_at.desc())
                    .first()
                    .updated_at
                    if session_count > 0
                    else None
                ),
            }

        except Exception as e:
            logger.exception(f"Error getting user profile: {e}", stacklevel=2)
            return {"user_id": user_id, "platform": platform}
        finally:
            db.close()
