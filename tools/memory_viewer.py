#!/home/admin/Desktop/Repos/Watkins/venv/bin/python3
"""
Watkins Memory Viewer
Browse and search conversation history
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class MemoryViewer:
    """View and search Watkins conversation history"""

    def __init__(self, history_file: str = "logs/conversation_full_history.jsonl"):
        self.history_file = history_file
        self.conversations = []
        self._load_conversations()

    def _load_conversations(self):
        """Load all conversations from JSONL file"""
        if not os.path.exists(self.history_file):
            print(f"No conversation history found at {self.history_file}")
            return

        with open(self.history_file, 'r') as f:
            for line in f:
                try:
                    self.conversations.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        print(f"Loaded {len(self.conversations)} conversations")

    def list_conversations(self, limit: int = 20):
        """List recent conversations with summaries"""
        print("\n" + "=" * 80)
        print("RECENT CONVERSATIONS")
        print("=" * 80)

        recent = self.conversations[-limit:] if len(self.conversations) > limit else self.conversations

        for conv in reversed(recent):
            conv_id = conv.get("conversation_id", "?")
            start_time = datetime.fromtimestamp(conv.get("start_time", 0))
            turns = conv.get("turns", 0)
            summary = conv.get("summary", "No summary available")

            print(f"\n[{conv_id}] {start_time.strftime('%Y-%m-%d %H:%M:%S')} ({turns} turns)")
            print(f"    {summary}")

        print()

    def view_conversation(self, conversation_id: int):
        """View full conversation details"""
        conv = next((c for c in self.conversations if c.get("conversation_id") == conversation_id), None)

        if not conv:
            print(f"Conversation {conversation_id} not found")
            return

        print("\n" + "=" * 80)
        print(f"CONVERSATION #{conversation_id}")
        print("=" * 80)

        start_time = datetime.fromtimestamp(conv.get("start_time", 0))
        end_time = datetime.fromtimestamp(conv.get("end_time", 0))
        duration = end_time - start_time

        print(f"Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {duration.total_seconds():.0f} seconds")
        print(f"Turns:    {conv.get('turns', 0)}")

        summary = conv.get("summary")
        if summary:
            print(f"Summary:  {summary}")

        print("\nMESSAGES:")
        print("-" * 80)

        messages = conv.get("messages", [])
        for msg in messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            timestamp = datetime.fromtimestamp(msg.get("timestamp", 0))

            print(f"\n[{timestamp.strftime('%H:%M:%S')}] {role}:")
            print(f"  {content}")

        print("\n" + "=" * 80 + "\n")

    def search_conversations(self, query: str):
        """Search conversations for keyword"""
        print(f"\nSearching for: '{query}'")
        print("=" * 80)

        found_count = 0
        for conv in self.conversations:
            matches = []

            # Search in summary
            summary = conv.get("summary", "")
            if query.lower() in summary.lower():
                matches.append("summary")

            # Search in messages
            messages = conv.get("messages", [])
            for msg in messages:
                if query.lower() in msg.get("content", "").lower():
                    matches.append(f"{msg.get('role')} message")
                    break

            if matches:
                found_count += 1
                conv_id = conv.get("conversation_id", "?")
                start_time = datetime.fromtimestamp(conv.get("start_time", 0))
                print(f"\n[{conv_id}] {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"    Matched in: {', '.join(matches)}")
                print(f"    Summary: {summary}")

        print(f"\nFound {found_count} matching conversations\n")

    def export_conversation(self, conversation_id: int, output_file: str):
        """Export conversation to text file"""
        conv = next((c for c in self.conversations if c.get("conversation_id") == conversation_id), None)

        if not conv:
            print(f"Conversation {conversation_id} not found")
            return

        with open(output_file, 'w') as f:
            f.write(f"Watkins Conversation #{conversation_id}\n")
            f.write("=" * 80 + "\n\n")

            start_time = datetime.fromtimestamp(conv.get("start_time", 0))
            f.write(f"Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Turns: {conv.get('turns', 0)}\n")

            summary = conv.get("summary")
            if summary:
                f.write(f"Summary: {summary}\n")

            f.write("\n" + "-" * 80 + "\n\n")

            messages = conv.get("messages", [])
            for msg in messages:
                role = msg.get("role", "unknown").capitalize()
                content = msg.get("content", "")
                timestamp = datetime.fromtimestamp(msg.get("timestamp", 0))

                f.write(f"[{timestamp.strftime('%H:%M:%S')}] {role}: {content}\n\n")

        print(f"Exported conversation to {output_file}")

    def get_statistics(self):
        """Display conversation statistics"""
        print("\n" + "=" * 80)
        print("CONVERSATION STATISTICS")
        print("=" * 80)

        total_conversations = len(self.conversations)
        total_turns = sum(c.get("turns", 0) for c in self.conversations)
        total_messages = sum(len(c.get("messages", [])) for c in self.conversations)

        if total_conversations > 0:
            oldest = datetime.fromtimestamp(self.conversations[0].get("start_time", 0))
            newest = datetime.fromtimestamp(self.conversations[-1].get("end_time", 0))
            avg_turns = total_turns / total_conversations

            print(f"Total Conversations: {total_conversations}")
            print(f"Total Turns:         {total_turns}")
            print(f"Total Messages:      {total_messages}")
            print(f"Average Turns:       {avg_turns:.1f}")
            print(f"Oldest Conversation: {oldest.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Newest Conversation: {newest.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("No conversations found")

        print("=" * 80 + "\n")


def print_help():
    """Print help message"""
    print("""
Watkins Memory Viewer

Usage:
    ./tools/memory_viewer.py [command] [options]

Commands:
    list [N]              List recent N conversations (default: 20)
    view <ID>             View full conversation by ID
    search <query>        Search conversations for keyword
    export <ID> <file>    Export conversation to text file
    stats                 Show conversation statistics
    help                  Show this help message

Examples:
    ./tools/memory_viewer.py list
    ./tools/memory_viewer.py view 5
    ./tools/memory_viewer.py search "weather"
    ./tools/memory_viewer.py export 3 conversation_3.txt
    ./tools/memory_viewer.py stats

Privacy Note:
    All conversations are stored locally on your device at:
    logs/conversation_full_history.jsonl
    """)


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1].lower()

    viewer = MemoryViewer()

    if command == "list":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        viewer.list_conversations(limit)

    elif command == "view":
        if len(sys.argv) < 3:
            print("Error: Please specify conversation ID")
            print("Usage: ./tools/memory_viewer.py view <ID>")
            return
        conv_id = int(sys.argv[2])
        viewer.view_conversation(conv_id)

    elif command == "search":
        if len(sys.argv) < 3:
            print("Error: Please specify search query")
            print("Usage: ./tools/memory_viewer.py search <query>")
            return
        query = " ".join(sys.argv[2:])
        viewer.search_conversations(query)

    elif command == "export":
        if len(sys.argv) < 4:
            print("Error: Please specify conversation ID and output file")
            print("Usage: ./tools/memory_viewer.py export <ID> <file>")
            return
        conv_id = int(sys.argv[2])
        output_file = sys.argv[3]
        viewer.export_conversation(conv_id, output_file)

    elif command == "stats":
        viewer.get_statistics()

    elif command == "help":
        print_help()

    else:
        print(f"Unknown command: {command}")
        print_help()


if __name__ == "__main__":
    main()
