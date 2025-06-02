import datetime
import uuid

import weave
from openai.types.responses import EasyInputMessageParam
from weave.trace.context import call_context
from weave.trace.context import weave_client_context as weave_client_context

from docsbot.chat.chat import ChatApp, ChatRequest
from docsbot.utils import get_logger

logger = get_logger(__name__)


class ChatSession:
    def __init__(self, chat_app=None, session_id=None):
        self.chat_app = chat_app or ChatApp()
        self.weave_client = weave_client_context.get_weave_client()
        self.session_id = session_id or str(uuid.uuid4())
        self.conversation = []
        self.is_active = True
        self.call = None

    async def __aenter__(self):
        # Create a top-level call for the entire session
        self.call = self.weave_client.create_call(
            op="chat_session",
            display_name=f"Chat Session {self.session_id[:8]}",
            inputs={"session_id": self.session_id},
        )
        logger.warning(f"Created new weave call with ID: {self.call.id} for session {self.session_id}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.is_active = False
        # Finish the call with either the result or exception
        if exc_type:
            self.weave_client.finish_call(call=self.call, exception=str(exc_val))
        else:
            self.weave_client.finish_call(call=self.call, output={"message_count": len(self.conversation)})

    async def process_message(self, user_input):
        """Process a single user message and get a response"""
        # Create a child call within the session's main call context
        child_call = self.weave_client.create_call(
            op="process_message",
            display_name=f"Message {len(self.conversation)//2 + 1}",
            inputs={"user_input": user_input, "session_id": self.session_id},
            parent=self.call,
        )
        logger.warning(
            f"ChatSession: Created child call {child_call.id} under parent {self.call.id} for session {self.session_id}"
        )

        try:
            if user_input.lower() == "exit":
                self.is_active = False
                self.weave_client.finish_call(call=child_call, output=None)
                return None

            # Get user_info from the runner_config if available
            user_info = {}
            if hasattr(self.chat_app, "docsbot") and hasattr(self.chat_app.docsbot, "runner_config"):
                user_info = getattr(self.chat_app.docsbot.runner_config, "user_info", {})

            # Set the child call in the context so nested operations are properly parented
            call_context.push_call(child_call)

            try:
                # Process through ChatApp with session context
                chat_response = await self.chat_app(
                    ChatRequest(
                        request=EasyInputMessageParam(role="user", content=user_input),
                        conversation=self.conversation,
                        session_id=self.session_id,
                        user_info={"is_active": self.is_active, **(user_info or {})},
                    )
                )

                # Update conversation history
                chat_response_dict = chat_response.model_dump(mode="json")
                self.conversation = chat_response_dict["conversation"]
                response = chat_response_dict["response"]["content"]

                # Finish the call with the successful result
                self.weave_client.finish_call(call=child_call, output={"response": response})
                return response

            finally:
                # Always pop the call from context
                call_context.pop_call(child_call.id)

        except Exception as e:
            # Finish the call with the exception
            self.weave_client.finish_call(call=child_call, exception=str(e))
            raise

    @property
    def history(self):
        """Get the current conversation history"""
        return self.conversation


class PersistentChatSession(ChatSession):
    """ChatSession with persistence capabilities to store conversation history in a database."""

    def __init__(self, chat_app=None, session_id=None, db_client=None):
        """Initialize a persistent chat session.

        Args:
            chat_app: The chat application instance.
            session_id: Unique identifier for this session (e.g., thread_id in Slack).
            db_client: Database client for persisting conversations.
        """
        super().__init__(chat_app, session_id)
        self.db_client = db_client

    async def __aenter__(self):
        """Load existing conversation from database when session starts."""
        try:
            # Load previous conversation if it exists
            stored_conversation = await self.db_client.get_conversation(self.session_id)
            if stored_conversation:
                self.conversation = stored_conversation
        except Exception as e:
            # Log but don't fail if we can't load history
            logger.exception(f"Failed to load conversation history: {e}", stacklevel=2)

        # Complete normal initialization
        return await super().__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Save conversation to database when session ends."""
        try:
            # Save the full conversation state
            await self.db_client.save_conversation(session_id=self.session_id, conversation=self.conversation)
        except Exception as e:
            logger.exception(f"Failed to save conversation: {e}", stacklevel=2)

        # Complete normal cleanup
        return await super().__aexit__(exc_type, exc_val, exc_tb)

    async def process_message(self, user_input):
        """Process a message and persist it to the database."""
        # Create a child call within the session's main call context
        child_call = self.weave_client.create_call(
            op="process_message",
            display_name=f"Message {len(self.conversation)//2 + 1}",
            inputs={"user_input": user_input, "session_id": self.session_id},
            parent=self.call,
        )
        logger.warning(
            f"PersistentChatSession: Created child call {child_call.id} under parent {self.call.id} for session {self.session_id}"
        )

        try:
            # Check for exit command
            if user_input.lower() == "exit":
                self.is_active = False
                self.weave_client.finish_call(call=child_call, output=None)
                return None

            # Get user_info from the runner_config if available
            user_info = {}
            if hasattr(self.chat_app, "docsbot") and hasattr(self.chat_app.docsbot, "runner_config"):
                user_info = getattr(self.chat_app.docsbot.runner_config, "user_info", {})

            # Set the child call in the context so nested operations are properly parented
            call_context.push_call(child_call)

            try:
                # Process through ChatApp with session context including db_client
                chat_response = await self.chat_app(
                    ChatRequest(
                        request=EasyInputMessageParam(role="user", content=user_input),
                        conversation=self.conversation,
                        session_id=self.session_id,
                        user_info={"is_active": self.is_active, **(user_info or {})},
                    )
                )

                # Update conversation history
                chat_response_dict = chat_response.model_dump(mode="json")
                self.conversation = chat_response_dict["conversation"]
                response = chat_response_dict["response"]["content"]

                try:
                    # Save the latest message and response
                    # This allows for real-time persistence, not just at session end
                    await self.db_client.save_message(
                        session_id=self.session_id,
                        user_input=user_input,
                        response=response,
                        timestamp=datetime.datetime.now().isoformat(),
                    )
                except Exception as e:
                    logger.exception(f"Failed to save message: {e}", stacklevel=2)

                # Finish the call with the successful result
                self.weave_client.finish_call(call=child_call, output={"response": response})
                return response

            finally:
                # Always pop the call from context
                call_context.pop_call(child_call.id)

        except Exception as e:
            # Finish the call with the exception
            self.weave_client.finish_call(call=child_call, exception=str(e))
            raise


class ThreadSessionManager:
    """Manages chat sessions for thread-based platforms like Slack and Discord."""

    def __init__(self, chat_app=None, db_client=None, session_timeout_minutes=60):
        """Initialize a thread session manager.

        Args:
            chat_app: The chat application instance.
            db_client: Optional database client for persistent sessions.
            session_timeout_minutes: How long to keep inactive sessions in memory.
        """
        self.chat_app = chat_app or ChatApp()
        self.db_client = db_client
        self.active_sessions = {}  # thread_id -> ChatSession
        self.session_timeouts = {}  # thread_id -> last_active_timestamp
        self.session_timeout_minutes = session_timeout_minutes

    async def get_or_create_session(self, thread_id):
        """Get an existing session or create a new one for the thread.

        Args:
            thread_id: The unique identifier for the thread.

        Returns:
            ChatSession: An active chat session for this thread.
        """
        current_time = datetime.datetime.now()

        # Log the request
        logger.warning(f"Session request for thread_id: {thread_id}")

        # Check if we have an active session
        if thread_id in self.active_sessions:
            session = self.active_sessions[thread_id]
            logger.warning(
                f"Reusing existing session for thread {thread_id}, weave_call_id: {session.call.id if session.call else 'None'}"
            )
            # Update the last active time
            self.session_timeouts[thread_id] = current_time
            return session

        # Log that we're creating a new session
        logger.warning(f"Creating NEW session for thread {thread_id} - not found in active_sessions")

        # If we have a DB client, try loading from DB first
        if self.db_client:
            try:
                stored_conversation = await self.db_client.get_conversation(thread_id)
                if stored_conversation:
                    logger.debug(f"Found stored conversation for thread {thread_id}, creating new session")
                else:
                    logger.debug(f"No stored conversation found for thread {thread_id}, creating brand new session")
            except Exception as e:
                logger.exception(f"Error checking stored conversation: {e}", stacklevel=2)

        # Create a new session
        if self.db_client:
            # Create persistent session if we have a DB client
            session = PersistentChatSession(chat_app=self.chat_app, session_id=thread_id, db_client=self.db_client)
        else:
            # Otherwise create a regular session
            session = ChatSession(chat_app=self.chat_app, session_id=thread_id)

        logger.debug(f"Creating new session for thread {thread_id}")

        # Initialize the session
        await session.__aenter__()

        # Log active sessions for debugging
        logger.warning(f"Active sessions after add: {list(self.active_sessions.keys())}")

        # Store it for future use
        self.active_sessions[thread_id] = session
        self.session_timeouts[thread_id] = current_time

        # Log active sessions for debugging
        logger.warning(f"New session initialized with weave_call_id: {session.call.id if session.call else 'None'}")

        return session

    @weave.op
    async def process_message(self, thread_id, user_input, user_info=None):
        """Process a message in the context of its thread.

        Args:
            thread_id: The thread identifier.
            user_input: The user's message text.
            user_info: Optional dictionary with user information.

        Returns:
            The response from the chat session.
        """
        # Ensure thread_id is a string to avoid hash issues
        thread_id = str(thread_id)
        logger.debug(f"Processing message for thread {thread_id}")

        session = await self.get_or_create_session(thread_id)

        # Add user_info to the session context if provided
        if user_info and hasattr(session, "chat_app") and hasattr(session.chat_app, "docsbot"):
            session.chat_app.docsbot.runner_config.user_info = user_info

        return await session.process_message(user_input)

    async def cleanup_inactive_sessions(self):
        """Close and remove inactive sessions to free up resources."""
        current_time = datetime.datetime.now()
        timeout_delta = datetime.timedelta(minutes=self.session_timeout_minutes)

        for thread_id in list(self.active_sessions.keys()):
            last_active = self.session_timeouts.get(thread_id)
            if last_active and (current_time - last_active > timeout_delta):
                # Session has timed out, close it
                session = self.active_sessions[thread_id]
                logger.debug(f"Cleaning up inactive session for thread {thread_id}")
                await session.__aexit__(None, None, None)

                # Remove from active sessions
                del self.active_sessions[thread_id]
                del self.session_timeouts[thread_id]
