import json

# import os
from datetime import datetime

# import nextcord
# from nextcord.ext import commands
#
# # Enable the message content intent for reading messages properly.
# intents = nextcord.Intents.default()
# intents.message_content = True
# intents.typing = False
# intents.presences = False
#
# bot = commands.Bot(command_prefix="!", intents=intents)
#
# @bot.event
# async def on_ready():
#     print(f"Logged in as {bot.user.name} ({bot.user.id})")
#     channel_id = os.environ["DISCORD_BOT_CHANNEL"]  # Replace with your channel ID
#     channel = bot.get_channel(channel_id)
#     messages = []
#     async for message in channel.history(limit=None, oldest_first=False):
#         messages.append(message)
#
#     with open("exported_chats.jsonl", "w", encoding="utf-8") as file:
#         for message in messages:
#             message_data = {
#                 "author_name": message.author.name,
#                 "author_id": str(message.author.id),
#                 "content": message.content,
#                 "cleaned_content": message.clean_content,
#                 "thread_messages": [],
#                 "message_id": message.id,
#                 "created_at": str(message.created_at),
#             }
#             # Check if the message has an associated thread
#             if message.thread is not None:
#                 # Iterate over the thread's history.
#                 # If you want to exclude the root message (which might duplicate the parent message),
#                 # check the ID and skip if necessary.
#                 async for thread_message in message.thread.history(limit=100, oldest_first=True):
#                     # Optionally skip the root message if it duplicates the parent message.
#                     if thread_message.id == message.id:
#                         continue
#                     message_data["thread_messages"].append(
#                         {
#                             "author_name": thread_message.author.name,
#                             "author_id": str(thread_message.author.id),
#                             "content": thread_message.content,
#                             "cleaned_content": thread_message.clean_content,  # FIX: use thread_message's clean content
#                             "thread_id": message.thread.id,
#                             "thread_name": message.thread.name,
#                             "created_at": str(thread_message.created_at),
#                             "reactions": [
#                                 (str(reaction), reaction.count)
#                                 for reaction in thread_message.reactions
#                             ] if thread_message.reactions else [],
#                         }
#                     )
#             file.write(json.dumps(message_data) + "\n")
#
#     print("Export completed!")
#
# bot.run(os.environ["DISCORD_BOT_TOKEN"])


def parse_discord_data(data):
    print(f"Total messages: {len(data)}")
    threads = [
        message for message in data if message["author_name"] == "wandbot (beta)"
    ]
    prompts = [
        message for message in data if message["author_name"] != "wandbot (beta)"
    ]
    responses = []

    for idx, thread in enumerate(threads):
        try:
            thread_time = datetime.fromisoformat(
                thread["created_at"].replace("Z", "+00:00")
            )

            # Find the closest prompt by time difference.
            closest_prompt = min(
                prompts,
                key=lambda prompt: abs(
                    datetime.fromisoformat(prompt["created_at"].replace("Z", "+00:00"))
                    - thread_time
                ),
            )
            question = closest_prompt["content"].strip()

            # Validate thread message count
            if len(thread["thread_messages"]) < 3:
                print(
                    f"Skipping thread index {idx}: Expected at least 3 thread messages, got {len(thread['thread_messages'])}"
                )
                continue

            answer = thread["thread_messages"][1]["content"].strip()
            reaction_data = thread["thread_messages"][2].get("reactions", [])
            reaction = sum([(1 if r[0] == "👍" else -1) * r[1] for r in reaction_data])

            responses.append(
                {"question": question, "answer": answer, "reaction": reaction}
            )
        except Exception as e:
            print(f"Error processing thread index {idx}: {e}")
    return responses


# Example usage:
# Assuming your data is already loaded in variable 'data'
with open("exported_chats.jsonl", "r", encoding="utf-8") as file:
    data = [json.loads(line) for line in file]
responses = parse_discord_data(data)
with open("../data/discord_questions.jsonl", "w", encoding="utf-8") as file:
    for response in responses:
        file.write(json.dumps(response) + "\n")
