"""
Conversation Manager for Watkins
Manages conversation history, context, and state
"""

import logging
from typing import List, Dict, Optional
import time
import json
from datetime import datetime


class ConversationManager:
    """Manages conversation state and history"""

    def __init__(
        self,
        max_history: int = 10,
        timeout_seconds: int = 30,
        save_history: bool = False,
        history_file: Optional[str] = None,
        auto_load_history: bool = False,
        retention_days: int = 30,
        save_summaries: bool = False,
        llm_client: Optional[any] = None
    ):
        """
        Initialize Conversation Manager

        Args:
            max_history: Maximum number of turns to keep
            timeout_seconds: Conversation timeout (resets after inactivity)
            save_history: Save conversation history to file
            history_file: Path to history file
            auto_load_history: Load previous conversations on startup
            retention_days: Keep conversations for N days (0 = forever)
            save_summaries: Save conversation summaries before timeout
            llm_client: LLM client for generating summaries
        """
        self.max_history = max_history
        self.timeout_seconds = timeout_seconds
        self.save_history = save_history
        self.history_file = history_file or "logs/conversation_history.json"
        self.auto_load_history = auto_load_history
        self.retention_days = retention_days
        self.save_summaries = save_summaries
        self.llm_client = llm_client

        self.logger = logging.getLogger(__name__)

        # Conversation state
        self.history: List[Dict] = []
        self.last_interaction_time = time.time()
        self.conversation_count = 0
        self.total_turns = 0
        self.conversation_summary = None

        # Full history storage (JSONL format)
        self.full_history_file = "logs/conversation_full_history.jsonl"

        self.logger.info(
            f"ConversationManager initialized: max_history={max_history}, "
            f"timeout={timeout_seconds}s, save_history={save_history}, "
            f"retention_days={retention_days}"
        )

        # Auto-load previous session if enabled
        if self.auto_load_history and self.save_history:
            self.load_from_file()
            self._cleanup_old_conversations()

    def add_user_message(self, message: str):
        """
        Add user message to history

        Args:
            message: User's message text
        """
        self._check_timeout()

        self.history.append({
            "role": "user",
            "content": message,
            "timestamp": time.time()
        })

        self._trim_history()
        self.last_interaction_time = time.time()
        self.total_turns += 1

        self.logger.debug(f"User: {message}")

        if self.save_history:
            self._save_to_file()

    def add_assistant_message(self, message: str):
        """
        Add assistant message to history

        Args:
            message: Assistant's message text
        """
        self.history.append({
            "role": "assistant",
            "content": message,
            "timestamp": time.time()
        })

        self._trim_history()
        self.last_interaction_time = time.time()

        self.logger.debug(f"Assistant: {message}")

        if self.save_history:
            self._save_to_file()

    def get_history(self, include_timestamps: bool = False) -> List[Dict]:
        """
        Get conversation history

        Args:
            include_timestamps: Include timestamps in history

        Returns:
            List of message dicts
        """
        if include_timestamps:
            return self.history.copy()
        else:
            return [
                {"role": msg["role"], "content": msg["content"]}
                for msg in self.history
            ]

    def get_recent_context(self, num_turns: int = 5) -> List[Dict]:
        """
        Get recent conversation context

        Args:
            num_turns: Number of recent turns to include

        Returns:
            List of recent messages
        """
        recent = self.history[-(num_turns * 2):] if self.history else []
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in recent
        ]

    def _check_timeout(self):
        """Check if conversation has timed out and reset if needed"""
        current_time = time.time()
        time_since_last = current_time - self.last_interaction_time

        if time_since_last > self.timeout_seconds and self.history:
            self.logger.info(
                f"Conversation timed out after {time_since_last:.1f}s, resetting"
            )
            # Generate and save summary before resetting
            if self.save_summaries and len(self.history) > 0:
                self._generate_and_save_summary()
            self.reset()

    def reset(self):
        """Reset conversation history"""
        if self.history:
            self.conversation_count += 1

        self.history.clear()
        self.last_interaction_time = time.time()

        self.logger.info("Conversation reset")

    def _trim_history(self):
        """Trim history to maximum length"""
        if len(self.history) > self.max_history * 2:  # *2 for user+assistant pairs
            removed = len(self.history) - (self.max_history * 2)
            self.history = self.history[removed:]
            self.logger.debug(f"Trimmed {removed} messages from history")

    def _save_to_file(self):
        """Save conversation history to file"""
        try:
            import os
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)

            with open(self.history_file, 'w') as f:
                data = {
                    "conversation_count": self.conversation_count,
                    "total_turns": self.total_turns,
                    "current_history": self.get_history(include_timestamps=True),
                    "last_updated": datetime.now().isoformat()
                }
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save history: {e}")

    def load_from_file(self):
        """Load conversation history from file"""
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                self.conversation_count = data.get("conversation_count", 0)
                self.total_turns = data.get("total_turns", 0)
                self.history = data.get("current_history", [])
                self.logger.info(f"Loaded history from {self.history_file}")

        except FileNotFoundError:
            self.logger.debug("No history file found")
        except Exception as e:
            self.logger.error(f"Failed to load history: {e}")

    def get_statistics(self) -> Dict:
        """
        Get conversation statistics

        Returns:
            Dict with statistics
        """
        return {
            "conversation_count": self.conversation_count,
            "total_turns": self.total_turns,
            "current_history_length": len(self.history),
            "time_since_last_interaction": time.time() - self.last_interaction_time
        }

    def get_last_user_message(self) -> Optional[str]:
        """Get the last user message"""
        for msg in reversed(self.history):
            if msg["role"] == "user":
                return msg["content"]
        return None

    def get_last_assistant_message(self) -> Optional[str]:
        """Get the last assistant message"""
        for msg in reversed(self.history):
            if msg["role"] == "assistant":
                return msg["content"]
        return None

    def has_context(self) -> bool:
        """Check if there is conversation context"""
        return len(self.history) > 0

    def clear_old_conversations(self):
        """Clear conversation if it has timed out"""
        self._check_timeout()

    def _generate_and_save_summary(self):
        """Generate conversation summary using LLM and save to full history"""
        if not self.llm_client or not self.history:
            return

        try:
            # Create a summary prompt
            conversation_text = "\n".join([
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in self.history
            ])

            summary_prompt = f"""Summarize this conversation in 1-2 sentences, focusing on key topics and any important information mentioned:

{conversation_text}

Summary:"""

            # Generate summary
            summary = self.llm_client.generate_response(
                summary_prompt,
                conversation_history=[],
                prefer_local=True  # Use local model for summaries to save costs
            )

            self.conversation_summary = summary
            self.logger.info(f"Generated conversation summary: {summary}")

            # Save to full history file
            self._save_to_full_history(summary)

        except Exception as e:
            self.logger.error(f"Failed to generate summary: {e}")

    def _save_to_full_history(self, summary: Optional[str] = None):
        """Append conversation to full history JSONL file"""
        try:
            import os
            os.makedirs(os.path.dirname(self.full_history_file), exist_ok=True)

            # Create conversation entry
            entry = {
                "conversation_id": self.conversation_count,
                "start_time": self.history[0]["timestamp"] if self.history else time.time(),
                "end_time": time.time(),
                "turns": len(self.history) // 2,
                "summary": summary or self.conversation_summary,
                "messages": self.get_history(include_timestamps=True),
                "saved_at": datetime.now().isoformat()
            }

            # Append to JSONL file
            with open(self.full_history_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')

            self.logger.debug(f"Saved conversation to full history: {len(self.history)} messages")

        except Exception as e:
            self.logger.error(f"Failed to save to full history: {e}")

    def _cleanup_old_conversations(self):
        """Remove conversations older than retention_days"""
        if self.retention_days == 0:  # 0 means keep forever
            return

        try:
            import os
            if not os.path.exists(self.full_history_file):
                return

            cutoff_time = time.time() - (self.retention_days * 24 * 60 * 60)
            kept_conversations = []

            # Read all conversations
            with open(self.full_history_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("end_time", 0) >= cutoff_time:
                            kept_conversations.append(line)
                    except json.JSONDecodeError:
                        continue

            # Rewrite file with only kept conversations
            with open(self.full_history_file, 'w') as f:
                f.writelines(kept_conversations)

            removed_count = sum(1 for _ in open(self.full_history_file)) - len(kept_conversations)
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} conversations older than {self.retention_days} days")

        except Exception as e:
            self.logger.error(f"Failed to cleanup old conversations: {e}")

    def get_conversation_summaries(self, limit: int = 10) -> List[Dict]:
        """Get recent conversation summaries from full history"""
        try:
            import os
            if not os.path.exists(self.full_history_file):
                return []

            summaries = []
            with open(self.full_history_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        summaries.append({
                            "conversation_id": entry.get("conversation_id"),
                            "start_time": entry.get("start_time"),
                            "turns": entry.get("turns"),
                            "summary": entry.get("summary")
                        })
                    except json.JSONDecodeError:
                        continue

            # Return most recent conversations
            return summaries[-limit:] if len(summaries) > limit else summaries

        except Exception as e:
            self.logger.error(f"Failed to get conversation summaries: {e}")
            return []


if __name__ == "__main__":
    # Test the Conversation Manager
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    cm = ConversationManager(max_history=5, save_history=False)

    # Simulate conversation
    cm.add_user_message("Hello, Watkins!")
    cm.add_assistant_message("Hello! How can I help you today?")

    cm.add_user_message("What's the weather like?")
    cm.add_assistant_message("I'm sorry, I don't have access to weather information.")

    cm.add_user_message("Tell me a joke")
    cm.add_assistant_message("Why did the robot go to therapy? It had too many bugs!")

    # Get history
    history = cm.get_history()
    logger.info(f"\nConversation history ({len(history)} messages):")
    for msg in history:
        logger.info(f"  {msg['role']}: {msg['content']}")

    # Get statistics
    stats = cm.get_statistics()
    logger.info(f"\nStatistics: {stats}")

    # Test timeout
    logger.info("\nTesting timeout (waiting 2s with 1s timeout)...")
    cm2 = ConversationManager(max_history=5, timeout_seconds=1)
    cm2.add_user_message("Test message")
    time.sleep(2)
    cm2.add_user_message("This should reset")
    logger.info(f"History length after timeout: {len(cm2.get_history())}")
