"""Entry point for the CLI application."""

import argparse
import asyncio

from docsbot.apps.cli.bot import CLIBot
from docsbot.apps.cli.config import CLIBotConfig
from docsbot.utils import get_logger

logger  = get_logger(__name__)

# Configure logging



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Start the DocsBot CLI application")
    parser.add_argument("--user-id", type=str, help="User ID for the session")
    return parser.parse_args()


async def main():
    """Main entry point for the CLI application."""
    args = parse_args()

    config = CLIBotConfig()

    try:
        # Create and run the CLI app
        cli_app = CLIBot(config=config, user_id=args.user_id)

        # Display user profile before starting
        if args.user_id:
            profile = await cli_app.get_user_profile()
            if profile.get("session_count", 0) > 0:
                logger.debug(
                    f"Welcome back! You have {profile.get('session_count')} previous sessions "
                    f"with {profile.get('message_count')} messages."
                )

        # Run the application
        await cli_app.run()

    except KeyboardInterrupt:
        logger.debug("Keyboard interrupt received, shutting down")
    except Exception as e:
        logger.error(f"Error running CLI app: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
