"""Slack message formatters."""

import json

import regex as re

from docsbot.utils import get_logger

logger  = get_logger(__name__)




class MrkdwnFormatter:
    def __init__(self):
        self.code_block_pattern = re.compile(r"(```.*?```)", re.DOTALL)
        self.language_spec_pattern = re.compile(r"^```[a-zA-Z]+\n", re.MULTILINE)
        self.markdown_link_pattern = re.compile(r"\[([^\[]+)\]\((.*?)\)", re.MULTILINE)
        self.bold_pattern = re.compile(r"\*\*([^*]+)\*\*", re.MULTILINE)
        self.strike_pattern = re.compile(r"~~([^~]+)~~", re.MULTILINE)
        self.header_pattern = re.compile(r"^#+\s*(.*?)\n", re.MULTILINE)

    @staticmethod
    def replace_markdown_link(match):
        text = match.group(1)
        url = match.group(2)
        return f"<{url}|{text}>"

    @staticmethod
    def replace_bold(match):
        return f"*{match.group(1)}*"

    @staticmethod
    def replace_strike(match):
        return f"~{match.group(1)}~"

    @staticmethod
    def replace_headers(match):
        header_text = match.group(1)
        return f"\n*{header_text}*\n"

    def format_structured_response(self, content):
        """Format a structured response with citations for Slack.

        Args:
            content: A string containing plain text, JSON string, or complex message structure

        Returns:
            Formatted string for Slack
        """

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

                # Format citations for Slack
                citations_text = ""
                for idx, citation in enumerate(citations):
                    source = citation.get("source", "Unknown")
                    url = citation.get("url", "#")
                    citations_text += f"  {idx}. <{url}|{source}>\n"

                # Only add citations section if there are citations
                if citations_text:
                    citations_text = citations_text.strip("\n")
                    formatted_answer = self(answer)
                    return f"{formatted_answer}\n\n---\n\n{citations_text}\n\n---\n\n"
                else:
                    return self(answer)
        except (json.JSONDecodeError, TypeError):
            # Not JSON or not our format, treat as plain text
            pass

        # If not JSON or not our expected format, format as plain text
        return self(content)

    def __call__(self, text):
        try:
            segments = self.code_block_pattern.split(text)

            for i, segment in enumerate(segments):
                if segment.startswith("```") and segment.endswith("```"):
                    segment = self.language_spec_pattern.sub("```\n", segment)
                    segments[i] = segment
                else:
                    segment = self.markdown_link_pattern.sub(self.replace_markdown_link, segment)
                    segment = self.bold_pattern.sub(self.replace_bold, segment)
                    segment = self.strike_pattern.sub(self.replace_strike, segment)
                    segment = self.header_pattern.sub(self.replace_headers, segment)
                    segments[i] = segment

            return "".join(segments)
        except Exception:
            logger.exception(f"Error parsing Slack response: {text}")
            return text
