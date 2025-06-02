"""Database models for chat session storage."""

import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class ChatSession(Base):
    """Model representing a chat session."""

    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, index=True)
    platform = Column(String(50), index=True)  # slack, discord, cli, etc.
    user_id = Column(String(255), index=True, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationship to messages
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<ChatSession(session_id={self.session_id}, platform={self.platform})>"


class ChatMessage(Base):
    """Model representing a message in a chat session."""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), ForeignKey("chat_sessions.session_id", ondelete="CASCADE"), index=True)
    role = Column(String(50), index=True)  # user, assistant, system, tool, etc.
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    message_id = Column(String(255), index=True, nullable=True)  # Optional platform-specific ID

    # Relationship to session
    session = relationship("ChatSession", back_populates="messages")

    def __repr__(self) -> str:
        return f"<ChatMessage(session_id={self.session_id}, role={self.role})>"


class Feedback(Base):
    """Model representing feedback on a chat message."""

    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(String(255), ForeignKey("chat_messages.message_id", ondelete="CASCADE"), nullable=True)
    session_id = Column(String(255), ForeignKey("chat_sessions.session_id", ondelete="CASCADE"), index=True)
    user_id = Column(String(255), index=True, nullable=True)  # User who gave the feedback
    rating = Column(Integer)  # e.g., -1, 0, 1
    feedback_text = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Feedback(message_id={self.message_id}, rating={self.rating})>"
