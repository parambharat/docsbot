"""CLI output formatter."""

from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.theme import Theme


class CLIFormatter:
    """Formatter for CLI output."""

    def __init__(self):
        """Initialize the CLI formatter."""
        self.console = Console(
            theme=Theme(
                {
                    "user": "bold blue",
                    "assistant": "bold green",
                    "tool_call": "bold yellow",
                    "tool_output": "dim yellow",
                }
            )
        )

    def print_user_message(self, content: str):
        """Print a user message.

        Args:
            content: The message content
        """
        self.console.print(
            Panel(
                content,
                title="[user]User[/user]",
                border_style="blue",
            )
        )

    def print_assistant_message(self, content: str):
        """Print an assistant message.

        Args:
            content: The message content
        """
        self.console.print(
            Panel(
                Markdown(content),
                title="[assistant]Assistant[/assistant]",
                border_style="green",
            )
        )

    def format_assistant_response(self, content):
        """Format a structured assistant response with citations.

        Args:
            content: A string containing plain text, JSON string, or complex message structure

        Returns:
            Formatted markdown string
        """
        import json

        # If content is None, return empty string
        if content is None:
            return ""

        # If not a string, try to serialize it first
        if not isinstance(content, str):
            try:
                # Check if it's a list of message chunks
                if isinstance(content, list):
                    # Extract text from output_text chunks
                    texts = []
                    for chunk in content:
                        if isinstance(chunk, dict) and chunk.get("type") == "output_text":
                            texts.append(chunk.get("text", ""))

                    if texts:
                        content = "".join(texts)
                    else:
                        content = json.dumps(content)
                else:
                    content = json.dumps(content)
            except (TypeError, ValueError):
                content = str(content)

        # Try to parse as JSON
        try:
            data = json.loads(content)
            # Check if it's our expected structure
            if isinstance(data, dict) and "answer" in data:
                answer = data.get("answer", "")
                citations = data.get("citations", [])

                # Format citations
                citations_text = ""
                for idx, citation in enumerate(citations):
                    citations_text += f"  {idx}. [{citation.get('source', 'Unknown')}]({citation.get('url', '#')})\n"

                # Only add citations section if there are citations
                if citations_text:
                    citations_text = citations_text.strip("\n")
                    return f"{answer}\n\n---\n\n{citations_text}\n\n---\n\n"
                else:
                    return answer
        except (json.JSONDecodeError, TypeError):
            # Not JSON or not our format, treat as plain text
            pass

        # Return as is if not JSON or not our expected format
        return content

    def print_function_call(self, name: str, arguments: dict[str, Any] = None):
        """Print a function call.

        Args:
            name: The function name
            arguments: The function arguments
        """
        try:
            args_str = str(arguments) if arguments else "{}"
            self.console.print(
                Panel(
                    f"Tool: [bold]{name}[/bold]\nArguments: {args_str}",
                    title="[tool_call]Tool Call[/tool_call]",
                    border_style="yellow",
                )
            )
        except Exception:
            self.console.print(
                Panel(
                    f"Tool: [bold]{name}[/bold]",
                    title="[tool_call]Tool Call[/tool_call]",
                    border_style="yellow",
                )
            )

    def print_function_output(self, output: str):
        """Print a function output.

        Args:
            output: The function output
        """
        self.console.print(
            Panel(
                output or "No output",
                title="[tool_output]Tool Output[/tool_output]",
                border_style="dim yellow",
            )
        )

    def print_conversation_history(self, conversation: list[dict[str, Any]]):
        """Print the conversation history.

        Args:
            conversation: The conversation history
        """
        import sys

        sys.stdout.flush()
        if not conversation:
            self.console.print("[italic]No conversation history[/italic]")
            return

        self.console.print("\n[bold]Conversation History:[/bold]\n")

        for message in conversation:
            if message.get("role") == "user":
                self.print_user_message(message["content"])
            elif message.get("role") == "assistant":
                if isinstance(message.get("content"), list):
                    # Handle structured content
                    content = "\n".join(
                        [item.get("text", "") for item in message["content"] if item.get("type") == "output_text"]
                    )
                else:
                    content = message.get("content", "")
                self.print_assistant_message(content)
            elif message.get("type") == "function_call":
                # Format function call
                self.print_function_call(
                    name=message.get("name", "Unknown tool"), arguments=message.get("arguments", "{}")
                )
            elif message.get("type") == "function_call_output":
                # Format function output
                self.print_function_output(message.get("output", "No output"))

            self.console.print()

    def print_info(self, message: str):
        """Print an info message.

        Args:
            message: The message to print
        """
        self.console.print(f"[bold]{message}[/bold]")

    def print_error(self, message: str):
        """Print an error message.

        Args:
            message: The error message
        """
        self.console.print(f"[bold red]Error: {message}[/bold red]")
