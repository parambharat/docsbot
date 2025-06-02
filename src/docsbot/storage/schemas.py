"""Pydantic schemas for chat storage models."""

import datetime

from pydantic import BaseModel, Field


class ChatMessageSchema(BaseModel):
    """Schema for chat message data."""

    role: str = Field(..., description="The role of the message sender (user, assistant, etc.)")
    content: str = Field(..., description="The content of the message")
    timestamp: str | None = Field(None, description="ISO format timestamp of when the message was sent")
    message_id: str | None = Field(None, description="Platform-specific ID for the message")

    class Config:
        orm_mode = True


class ChatSessionSchema(BaseModel):
    """Schema for chat session data."""

    session_id: str = Field(..., description="Unique identifier for the session")
    platform: str = Field(..., description="Platform where the session is taking place")
    user_id: str | None = Field(None, description="ID of the user in the platform")
    created_at: datetime.datetime | None = Field(None, description="When the session was created")
    updated_at: datetime.datetime | None = Field(None, description="When the session was last updated")
    messages: list[ChatMessageSchema] = Field(default_factory=list, description="Messages in the session")

    class Config:
        orm_mode = True


class ChatSessionCreateSchema(BaseModel):
    """Schema for creating a new chat session."""

    session_id: str = Field(..., description="Unique identifier for the session")
    platform: str = Field(..., description="Platform where the session is taking place")
    user_id: str | None = Field(None, description="ID of the user in the platform")


class ChatMessageCreateSchema(BaseModel):
    """Schema for creating a new chat message."""

    session_id: str = Field(..., description="ID of the session this message belongs to")
    role: str = Field(..., description="The role of the message sender")
    content: str = Field(..., description="The content of the message")
    message_id: str | None = Field(None, description="Platform-specific ID for the message")


class FeedbackSchema(BaseModel):
    """Schema for feedback data."""

    message_id: str | None = Field(None, description="ID of the message being rated")
    session_id: str = Field(..., description="ID of the session this feedback is for")
    rating: int = Field(..., description="Numeric rating (-1, 0, 1)")
    feedback_text: str | None = Field(None, description="Optional text feedback")

    class Config:
        orm_mode = True
