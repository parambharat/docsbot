import json
from datetime import datetime


def parse_slack_entry(entry):
    try:
        question = entry["text"]
        answer = ""
        thread = []
        if "thread" in entry and len(entry["thread"]) > 1:
            thread = [
                {
                    "author": msg.get("user", "unknown"),
                    "content": msg["text"],
                    "timestamp": datetime.utcfromtimestamp(
                        float(msg["timestamp"])
                    ).isoformat(),
                }
                for msg in entry["thread"][1:]
            ]
            answer = thread[0]["content"] if thread else ""
        return {
            "question": question,
            "answer": answer,
            "source": "slack",
            "timestamp": datetime.utcfromtimestamp(
                float(entry["timestamp"])
            ).isoformat(),
            "metadata": {"reactions": [], "thread": thread, "tags": []},
        }
    except Exception as e:
        print(f"Error parsing Slack entry: {e}")
        return None


def parse_discord_entry(entry):
    try:
        question = entry.get("question", "")
        answer = entry.get("answer", "")
        reactions = []

        if "reaction" in entry:
            reactions.append({"count": entry["reaction"]})

        return {
            "question": question,
            "answer": answer,
            "source": "discord",
            "timestamp": entry.get("timestamp", ""),
            "metadata": {"reactions": reactions, "thread": [], "tags": []},
        }
    except Exception as e:
        print(f"Error parsing Discord entry: {e}")
        return None


def parse_discourse_entry(entry):
    try:
        posts = entry.get("posts", [])
        if not posts:
            return None
        question = posts[0].get("cooked", "")
        answer = ""
        thread = []
        for post in posts[1:]:
            content = post.get("cooked", "")
            timestamp = post.get("created_at", "")
            thread.append(
                {"author": "unknown", "content": content, "timestamp": timestamp}
            )
        answer = thread[0]["content"] if thread else ""
        return {
            "question": question,
            "answer": answer,
            "source": "discourse",
            "timestamp": datetime.utcfromtimestamp(
                entry.get("created_at", 0) / 1000
            ).isoformat(),
            "metadata": {
                "reactions": [],
                "thread": thread,
                "tags": entry.get("tags", []),
            },
        }
    except Exception as e:
        print(f"Error parsing Discourse entry: {e}")
        return None


def unify_datasets(slack_path, discord_path, discourse_path, output_path):
    unified_entries = []

    # Process Slack data
    with open(slack_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            parsed = parse_slack_entry(entry)
            if parsed:
                unified_entries.append(parsed)

    # Process Discord data
    with open(discord_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            parsed = parse_discord_entry(entry)
            if parsed:
                unified_entries.append(parsed)

    # Process Discourse data
    with open(discourse_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            parsed = parse_discourse_entry(entry)
            if parsed:
                unified_entries.append(parsed)

    # Write unified data to output file
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in unified_entries:
            f.write(json.dumps(entry) + "\n")


# Example usage
unify_datasets(
    slack_path="data/slack_data.jsonl",
    discord_path="data/discord_data.jsonl",
    discourse_path="data/discourse_data.jsonl",
    output_path="data/unified_dataset.jsonl",
)
