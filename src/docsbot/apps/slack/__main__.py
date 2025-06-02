"""Entry point for the Slack bot."""

import argparse
import asyncio

from docsbot.apps.slack.bot import SlackBot
from docsbot.apps.slack.config import SlackAppConfig
from docsbot.utils import get_logger

logger  = get_logger(__name__)




def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Start the DocsBot Slack integration")
    return parser.parse_args()


async def main():
    """Main entry point for the Slack bot."""
    args = parse_args()

    # Load configuration
    try:
        config = SlackAppConfig()
        logger.debug("Starting Slack bot")

        # Create and start the bot
        bot = SlackBot(config)
        await bot.start()

    except KeyboardInterrupt:
        logger.debug("Keyboard interrupt received, shutting down")
    except Exception as e:
        logger.exception(f"Error starting Slack bot: {e}", stacklevel=2)


if __name__ == "__main__":
    asyncio.run(main())
