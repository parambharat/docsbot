from dataclasses import dataclass, field

from agents import RunContextWrapper


@dataclass
class DocsContext:
    """Context object for DocsBot Agents"""

    # Session information
    session_id: str

    # User information
    user_info: dict = field(default_factory=dict)

    # Conversation history
    conversation: list = field(default_factory=list)

    # Additional state that may be useful across tools/agents
    metadata: dict = field(default_factory=dict)

    # Function to help tools get typed context
    @staticmethod
    def get_context(wrapper: RunContextWrapper["DocsContext"]) -> "DocsContext":
        """Helper method to extract the context from a wrapper"""
        return wrapper.context
