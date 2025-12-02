# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Watkins is a privacy-focused voice assistant for Raspberry Pi 5 with hybrid local/cloud LLM support and persistent memory capabilities. The architecture follows a modular pipeline: Audio I/O → Wake Word Detection → VAD → STT → LLM → TTS.

**Key Differentiator**: Unlike Alexa or Google Assistant, Watkins stores all conversation history locally on the device, with intelligent summaries and full privacy control.

## Running and Testing

### Running the Application
```bash
# Main application
./watkins.py

# Or explicitly with venv
./venv/bin/python watkins.py
```

### Testing Individual Modules
Each module in `src/` can be run independently for testing:
```bash
./venv/bin/python src/audio_manager.py    # Test audio devices
./venv/bin/python src/vad_processor.py     # Test VAD
./venv/bin/python src/stt_engine.py        # Test speech-to-text
./venv/bin/python src/llm_client.py        # Test LLM
./venv/bin/python src/tts_engine.py        # Test text-to-speech
./venv/bin/python tools/audio_config.py    # Audio device configuration utility
```

### Testing with Timeout
When testing the main application, use a 10-second timeout to prevent hanging:
```bash
timeout 10 ./watkins.py
```

## Architecture

### Core Pipeline
1. **AudioManager** (`src/audio_manager.py`): Manages microphone input/speaker output using sounddevice. Uses queue-based audio buffering with callback-driven recording.
2. **WakeWordDetector** (`src/wake_word_detector.py`): Detects wake words using Picovoice Porcupine. Processes 512-sample chunks at 16kHz.
3. **VADProcessor** (`src/vad_processor.py`): Voice activity detection using Silero VAD. Distinguishes speech from silence.
4. **STTEngine** (`src/stt_engine.py`): Speech-to-text using Faster-Whisper. Converts audio to text.
5. **LLMClient** (`src/llm_client.py`): Hybrid LLM client supporting both Anthropic Claude (cloud) and Ollama (local).
6. **TTSEngine** (`src/tts_engine.py`): Text-to-speech using Piper.
7. **ConversationManager** (`src/conversation_manager.py`): Tracks conversation history with automatic timeout/reset.

### Key Design Patterns

**Audio Chunk Size**: Must be exactly 512 samples at 16kHz for Silero VAD and Porcupine compatibility.

**Recording States**: The AudioManager has two modes:
- Continuous recording (wake word mode): `start_recording()` called once, chunks consumed from queue
- Single-shot recording (push-to-talk mode): `start_recording()` / `stop_recording()` per interaction

**Hybrid LLM Mode**: In hybrid mode, cloud (Claude) is preferred unless explicitly requesting local. Automatic fallback to local on cloud failure.

**Conversation Flow**:
- Wake word detected → Clear audio buffer → Handle interaction → 5-second follow-up window
- If speech detected in window, continue conversation
- If no speech, return to wake word mode

## Configuration

**Main config**: `config/config.yaml`
- Audio settings (sample_rate, chunk_size, device IDs)
- Wake word configuration (keyword, sensitivity)
- VAD parameters (threshold, duration limits)
- STT model selection (tiny, base, small, medium)
- LLM mode (cloud, local, hybrid) and model selection
- TTS voice model
- **Memory settings**: save_history, auto_load_history, retention_days, save_summaries

**Environment variables**: `.env` file
- `ANTHROPIC_API_KEY`: Required for cloud/hybrid LLM mode
- `PORCUPINE_ACCESS_KEY`: Required for wake word detection
- Use `.env.template` as reference

## Memory System

**Storage Format**:
- `logs/conversation_history.json`: Current session state (for quick loading)
- `logs/conversation_full_history.jsonl`: Complete append-only conversation archive
- Each conversation includes: messages, timestamps, summary, metadata

**Key Features**:
- **Persistent Memory**: Conversations automatically saved and loaded on startup
- **Intelligent Summaries**: Claude generates concise summaries before timeout (configurable)
- **Privacy-First**: All data stored locally, never in the cloud
- **Configurable Retention**: Auto-delete conversations older than N days
- **Search & Export**: Memory viewer tool for browsing history

**Memory Viewer Tool**:
```bash
./tools/memory_viewer.py list           # List recent conversations
./tools/memory_viewer.py view 5         # View specific conversation
./tools/memory_viewer.py search "topic" # Search for keywords
./tools/memory_viewer.py export 3 file.txt  # Export to text file
./tools/memory_viewer.py stats          # Show statistics
```

**Implementation Details**:
- ConversationManager handles all memory operations
- Summaries generated using local LLM (when save_summaries=true) to save API costs
- Auto-cleanup runs on startup based on retention_days
- JSONL format allows easy append and parsing
- System prompt updated to reference available memory

## Dependencies and Installation

**Python environment**: Uses venv at `./venv/`
```bash
source venv/bin/activate
./venv/bin/pip install -r requirements.txt
```

**External services**:
- Ollama: Must be running for local LLM mode (`systemctl status ollama`, `ollama list`)
- Models auto-download on first use (Whisper, Piper voices)

**System requirements**: Raspberry Pi OS 64-bit with portaudio19-dev, alsa-utils

## Common Development Tasks

### Changing LLM Models
Edit `config/config.yaml`:
```yaml
llm:
  mode: "hybrid"  # or "cloud", "local"
  cloud:
    model: "claude-sonnet-4-5-20250929"
  local:
    model: "tinyllama"  # or "phi3.5"
```

Download new Ollama model: `ollama pull <model_name>`

### Adjusting Wake Word Sensitivity
Edit `config/config.yaml`:
```yaml
wake_word:
  keyword: "porcupine"  # or bumblebee, americano, etc.
  sensitivity: 0.7  # 0.0 to 1.0 (higher = more sensitive)
```

### Testing Audio Devices
Run the audio configuration utility to list devices and test microphone/speakers:
```bash
./venv/bin/python tools/audio_config.py
```

Update device IDs in `config/config.yaml` under `audio.input_device` and `audio.output_device`.

### Debugging
- Logs: `logs/watkins.log`
- Set log level in `config/config.yaml`: `logging.level: "DEBUG"`
- Each module has standalone test code in `if __name__ == "__main__"` blocks

## Important Constraints

- **Never commit** `.env` file (contains API keys)
- **chunk_size must be 512** at 16kHz for VAD/wake word compatibility
- Recording must be started before calling `listen_for_speech()` (managed by main loop)
- Audio queue should be cleared after wake word detection to avoid processing buffered audio
- System prompt in LLM config should emphasize brevity for voice responses

## Project Structure Notes

- **src/**: All core modules (audio, VAD, STT, LLM, TTS, wake word, conversation)
- **config/**: YAML configuration
- **tools/**: Utility scripts (audio_config.py)
- **logs/**: Application logs (auto-created)
- **models/**: Model cache directory (auto-created)
- **watkins.py**: Main orchestrator that ties all components together

## Performance Considerations

- Smaller models = faster inference (tiny/base Whisper, tinyllama)
- Local mode eliminates network latency but slower inference
- Cloud mode (Claude) has faster inference but requires internet
- Hybrid mode balances both with automatic fallback
- Reduce `max_tokens` in LLM config for faster responses
