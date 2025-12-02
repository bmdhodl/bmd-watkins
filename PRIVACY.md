# Privacy Policy - Watkins Voice Assistant

Last Updated: December 1, 2025

## Overview

Watkins is designed with **privacy-first** principles. Unlike commercial voice assistants like Amazon Alexa, Google Assistant, or Apple Siri, Watkins keeps your data on **your device**, under **your control**.

## What Data is Collected

### Conversation Data
- **Voice Recordings**: Processed in real-time, not permanently stored
- **Transcribed Text**: Your spoken words converted to text
- **Assistant Responses**: Watkins' replies to your questions
- **Timestamps**: When each interaction occurred
- **Conversation Summaries**: Automatically generated summaries of conversations

### Configuration Data
- **Settings**: Your preferences in `config/config.yaml`
- **API Keys**: Stored in `.env` file (never committed to git)

## Where Your Data is Stored

### Local Storage (On Your Device)
All conversation data is stored locally on your Raspberry Pi:

```
/home/admin/Desktop/Repos/Watkins/logs/
├── conversation_history.json          # Current session state
├── conversation_full_history.jsonl    # Complete conversation archive
└── watkins.log                        # Application logs
```

**Important**: These files NEVER leave your device unless you explicitly copy or export them.

### Cloud Services (Optional)

If you enable cloud LLM mode, the following data is sent to external services:

#### Anthropic Claude API
- **What's sent**: Your current question + recent conversation context (last 5 turns)
- **What's NOT sent**: Your full conversation history, summaries, or personal data
- **Purpose**: To generate intelligent responses
- **Retention**: Per [Anthropic's privacy policy](https://www.anthropic.com/privacy)
- **Your control**: Disable by switching to `local` mode in config

#### Picovoice Porcupine (Wake Word)
- **What's sent**: Encrypted wake word detection queries (for validation)
- **What's NOT sent**: Your voice recordings or conversations
- **Purpose**: Wake word licensing validation
- **Your control**: Disable wake word mode in config

### What is NEVER Sent to the Cloud
- ❌ Full conversation history
- ❌ Conversation summaries
- ❌ Voice recordings
- ❌ Personal information mentioned in conversations
- ❌ System logs
- ❌ Configuration files

## Data Retention

### Local Retention (Your Device)
You have full control over how long conversations are kept:

**Default**: 30 days
**Configurable**: Edit `config/config.yaml`:
```yaml
conversation:
  retention_days: 30  # Change to 7, 90, 365, or 0 (keep forever)
```

**Automatic Cleanup**: Old conversations are automatically removed based on your retention setting.

**Manual Deletion**:
```bash
# Delete all conversation history
rm logs/conversation_full_history.jsonl
rm logs/conversation_history.json

# Or delete specific conversations using the memory viewer
./tools/memory_viewer.py  # (deletion feature coming soon)
```

### Cloud Retention
- **Anthropic**: Refer to [Anthropic's data retention policy](https://www.anthropic.com/privacy)
- **Picovoice**: Refer to [Picovoice's privacy policy](https://picovoice.ai/privacy-policy/)

## Your Privacy Rights

### You Have the Right To:
1. **Access**: View all your conversations at any time
2. **Export**: Download your data in readable format
3. **Delete**: Remove all or specific conversations
4. **Control**: Decide what's saved and for how long
5. **Opt-Out**: Run completely offline (local-only mode)

### How to Exercise Your Rights

**View Your Data**:
```bash
./tools/memory_viewer.py list
./tools/memory_viewer.py view <conversation_id>
```

**Export Your Data**:
```bash
./tools/memory_viewer.py export <conversation_id> output.txt
```

**Delete Your Data**:
```bash
rm logs/conversation_full_history.jsonl
```

**Disable Memory**:
Edit `config/config.yaml`:
```yaml
conversation:
  save_history: false
```

**Run Completely Offline**:
Edit `config/config.yaml`:
```yaml
llm:
  mode: "local"  # Use only Ollama (no internet required)

wake_word:
  enabled: false  # Disable wake word (no Picovoice API)
```

## Data Security

### On Your Device
- **File Permissions**: Conversation files are readable only by your user account
- **No Encryption**: Files are stored in plain text (it's your device, you can encrypt the filesystem)
- **No Network Access**: Conversation files never transmitted over network

### In Transit (Cloud Mode Only)
- **HTTPS**: All API calls to Anthropic are encrypted in transit
- **No Logs**: We don't log API requests or responses
- **Minimal Data**: Only current question + minimal context sent

### API Keys
- **Environment Variables**: Stored in `.env` file (not committed to git)
- **File Permissions**: Readable only by your user account
- **Rotation**: You should rotate keys periodically

## Comparison with Commercial Assistants

| Privacy Feature | Watkins | Alexa | Google Assistant | Siri |
|----------------|---------|-------|------------------|------|
| Local Storage | ✅ Yes | ❌ No | ❌ No | ❌ No |
| Cloud Storage | ⚠️ Optional | ✅ Always | ✅ Always | ✅ Always |
| Data Access | You only | Amazon + partners | Google + partners | Apple |
| Conversation History | You control | Amazon controls | Google controls | Apple controls |
| Export Data | ✅ Easy | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited |
| Delete Data | ✅ Instant | ⚠️ Delayed | ⚠️ Delayed | ⚠️ Delayed |
| Offline Mode | ✅ Yes | ❌ No | ❌ No | ⚠️ Limited |
| Open Source | ✅ Yes | ❌ No | ❌ No | ❌ No |
| Corporate Surveillance | ❌ Never | ✅ Yes | ✅ Yes | ✅ Yes |

## Privacy Best Practices

### Recommended Settings for Maximum Privacy:
```yaml
# config/config.yaml
llm:
  mode: "local"  # Use Ollama only

wake_word:
  enabled: false  # Disable wake word

conversation:
  save_history: true  # Save locally
  retention_days: 30  # Auto-delete after 30 days
```

### Additional Security Steps:
1. **Encrypt Your Filesystem**: Use LUKS or similar to encrypt your Pi's SD card
2. **Secure Your Network**: Use WPA3, strong passwords, firewall rules
3. **Rotate API Keys**: Change Anthropic/Porcupine keys periodically
4. **Review Logs**: Periodically check what's being saved
5. **Backup Safely**: If backing up, encrypt backups

## Open Source Transparency

Watkins is 100% open source. You can:
- **Inspect the Code**: See exactly what data is collected and where it goes
- **Audit Network Traffic**: Use tools like Wireshark to verify no unexpected data transmission
- **Modify Behavior**: Change the code to suit your privacy needs
- **Report Issues**: File privacy concerns at [GitHub Issues](https://github.com/bmdhodl/bmd-watkins/issues)

## Contact

For privacy questions or concerns:
- **GitHub Issues**: https://github.com/bmdhodl/bmd-watkins/issues
- **Review the Code**: All privacy-related code is in `src/conversation_manager.py`

## Changes to This Policy

This privacy policy may be updated as features are added. Check the "Last Updated" date at the top of this file. Major changes will be announced in the repository README.

## License

Watkins is licensed under the MIT License. You are free to modify the code, including privacy-related features, to meet your needs.

---

**Remember**: With Watkins, your voice assistant works for YOU, not for a corporation. Your conversations, your data, your control.
