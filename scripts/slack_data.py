import asyncio
import json
import logging
import os

from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

class AsyncSlackScraper:
    def __init__(self, token, channel_id, bot_id, batch_size=10, delay=3):
        self.token = token
        self.channel_id = channel_id
        self.bot_id = bot_id
        self.client = AsyncWebClient(token=self.token)
        self.logger = logging.getLogger(__name__)
        self.batch_size = batch_size
        self.delay = delay

    async def get_conversation_replies(self, channel_id, message_ts):
        """Fetch the full thread of a specific message asynchronously."""
        try:
            result = await self.client.conversations_replies(channel=channel_id, ts=message_ts)
            all_messages = result["messages"]
            while result.get("has_more"):
                result = await self.client.conversations_replies(
                    channel=channel_id,
                    ts=message_ts,
                    cursor=result["response_metadata"]["next_cursor"],
                )
                all_messages += result["messages"]
            return all_messages
        except SlackApiError as e:
            self.logger.error(f"Error fetching replies: {e}")
            return []

    async def fetch_messages_batch(self, messages):
        """Fetch threads for a batch of messages."""
        tasks = [
            self.get_conversation_replies(self.channel_id, message["ts"])
            for message in messages
        ]
        return await asyncio.gather(*tasks)

    async def get_and_write_messages(self, filename, debug_limit=None):
        """Fetch messages mentioning the bot and their full threads, with batching and rate-limiting."""
        cursor = None
        conversation_count = 0  # Counter for debug mode
        with open(filename, "a+") as f:
            while True:
                try:
                    result = await self.client.conversations_history(
                        channel=self.channel_id, cursor=cursor, limit=1000
                    )
                    if result is None or result.get("response_metadata") is None:
                        break

                    messages = [
                        message for message in result["messages"]
                        if f"<@{self.bot_id}>" in message.get("text", "")
                    ]

                    # Process messages in batches
                    for i in range(0, len(messages), self.batch_size):
                        batch = messages[i:i + self.batch_size]
                        threads = await self.fetch_messages_batch(batch)

                        for message, thread_messages in zip(batch, threads):
                            message_dict = {
                                "text": message["text"],
                                "timestamp": message["ts"],
                                "thread": [
                                    {
                                        "text": thread_message["text"],
                                        "timestamp": thread_message["ts"],
                                    }
                                    for thread_message in thread_messages
                                ],
                            }
                            f.write(json.dumps(message_dict) + "\n")
                            conversation_count += 1

                            if debug_limit is not None and conversation_count >= debug_limit:
                                return

                        # Wait before processing the next batch
                        await asyncio.sleep(self.delay)

                    cursor = result["response_metadata"].get("next_cursor")
                    if not cursor:
                        break

                except SlackApiError as e:
                    self.logger.error(f"Error fetching messages: {e}")
                    break

async def main():
    scraper = AsyncSlackScraper(
        token=os.environ.get("SLACK_BOT_TOKEN"),
        channel_id=os.environ.get("SLACK_CHANNEL_ID"),
        bot_id=os.environ.get("SLACK_BOT_ID"),
        batch_size=10,  # Number of requests to run simultaneously
        delay=30,  # Delay in seconds between batches
    )
    await scraper.get_and_write_messages("data/slack_questions.jsonl")

if __name__ == "__main__":
    asyncio.run(main())
