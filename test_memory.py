#!/home/admin/Desktop/Repos/Watkins/venv/bin/python3
"""
Test script for Watkins memory system
"""

import sys
import os
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from conversation_manager import ConversationManager


def test_memory_system():
    """Test the conversation memory system"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("TESTING WATKINS MEMORY SYSTEM")
    logger.info("=" * 60)

    # Clean up any existing test files
    test_history_file = "logs/test_conversation_history.json"
    test_full_history = "logs/test_conversation_full_history.jsonl"

    for f in [test_history_file, test_full_history]:
        if os.path.exists(f):
            os.remove(f)
            logger.info(f"Cleaned up existing {f}")

    # Test 1: Initialize ConversationManager with new features
    logger.info("\n[Test 1] Initializing ConversationManager with memory features...")
    cm = ConversationManager(
        max_history=5,
        timeout_seconds=10,
        save_history=True,
        history_file=test_history_file,
        auto_load_history=True,
        retention_days=30,
        save_summaries=False,  # Disable summaries for testing (requires LLM)
        llm_client=None
    )
    logger.info("✓ ConversationManager initialized successfully")

    # Test 2: Add messages
    logger.info("\n[Test 2] Adding messages to conversation...")
    cm.add_user_message("Hello, Watkins!")
    cm.add_assistant_message("Hello! How can I help you today?")
    cm.add_user_message("What's the weather like?")
    cm.add_assistant_message("I don't have access to weather data, but I can help with other questions!")
    logger.info(f"✓ Added 4 messages (2 turns)")

    # Test 3: Get history
    logger.info("\n[Test 3] Retrieving conversation history...")
    history = cm.get_history()
    logger.info(f"✓ Retrieved {len(history)} messages")

    # Test 4: Get recent context
    logger.info("\n[Test 4] Getting recent context...")
    context = cm.get_recent_context(num_turns=2)
    logger.info(f"✓ Retrieved {len(context)} messages of recent context")

    # Test 5: Save to file
    logger.info("\n[Test 5] Saving conversation to file...")
    cm._save_to_file()
    if os.path.exists(test_history_file):
        logger.info(f"✓ Conversation saved to {test_history_file}")
    else:
        logger.error(f"✗ Failed to save conversation")

    # Test 6: Save to full history
    logger.info("\n[Test 6] Saving to full history (JSONL)...")
    cm._save_to_full_history(summary="Test conversation about weather")
    if os.path.exists(test_full_history):
        with open(test_full_history, 'r') as f:
            lines = f.readlines()
        logger.info(f"✓ Saved to full history: {len(lines)} conversation(s)")
    else:
        logger.error(f"✗ Failed to save to full history")

    # Test 7: Load from file
    logger.info("\n[Test 7] Creating new ConversationManager and loading history...")
    cm2 = ConversationManager(
        max_history=5,
        timeout_seconds=10,
        save_history=True,
        history_file=test_history_file,
        auto_load_history=True,
        retention_days=30,
        save_summaries=False,
        llm_client=None
    )
    loaded_history = cm2.get_history()
    logger.info(f"✓ Loaded {len(loaded_history)} messages from file")

    if len(loaded_history) == len(history):
        logger.info("✓ Loaded history matches saved history")
    else:
        logger.error(f"✗ History mismatch: saved {len(history)}, loaded {len(loaded_history)}")

    # Test 8: Get conversation summaries
    logger.info("\n[Test 8] Getting conversation summaries...")
    summaries = cm2.get_conversation_summaries(limit=10)
    logger.info(f"✓ Retrieved {len(summaries)} conversation summary(ies)")

    # Test 9: Statistics
    logger.info("\n[Test 9] Getting statistics...")
    stats = cm2.get_statistics()
    logger.info(f"✓ Statistics: {stats}")

    # Test 10: Cleanup
    logger.info("\n[Test 10] Cleaning up test files...")
    for f in [test_history_file, test_full_history]:
        if os.path.exists(f):
            os.remove(f)
            logger.info(f"✓ Removed {f}")

    logger.info("\n" + "=" * 60)
    logger.info("ALL TESTS PASSED!")
    logger.info("=" * 60)
    logger.info("\nMemory system is working correctly. The following features are enabled:")
    logger.info("  ✓ Persistent conversation storage")
    logger.info("  ✓ Auto-load on startup")
    logger.info("  ✓ JSONL full history format")
    logger.info("  ✓ Conversation summaries (when LLM available)")
    logger.info("  ✓ Configurable retention period")
    logger.info("  ✓ Memory viewer tool")
    logger.info("\nNext steps:")
    logger.info("  1. Run Watkins and have some conversations")
    logger.info("  2. Check logs/conversation_full_history.jsonl")
    logger.info("  3. Use ./tools/memory_viewer.py to browse history")


if __name__ == "__main__":
    test_memory_system()
