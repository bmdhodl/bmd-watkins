# Watkins Voice Assistant

A powerful, privacy-focused voice assistant built for Raspberry Pi 5, featuring local and cloud LLM support, wake word detection, and natural voice synthesis.

## Features

- **Privacy-First Memory**: Remembers all conversations locally on your device (never in the cloud)
- **Wake Word Detection**: Hands-free activation with "Porcupine" or other wake words
- **Speech-to-Text**: Fast, accurate transcription using Faster-Whisper
- **Hybrid LLM**: Switch between local (Ollama) and cloud (Anthropic Claude) models
- **Natural TTS**: High-quality voice synthesis with Piper
- **Voice Activity Detection**: Intelligent speech boundary detection with Silero VAD
- **Conversation Context**: Maintains conversation history for natural interactions
- **Privacy-Focused**: Can run completely offline with local models

## Privacy-First Memory

Unlike Alexa and other cloud-based assistants, Watkins stores your conversations **locally on your Raspberry Pi**. Your privacy is protected:

- ✅ **Local Storage Only**: All conversations saved to your device, never uploaded to the cloud
- ✅ **Persistent Memory**: Watkins remembers previous conversations across sessions
- ✅ **Intelligent Summaries**: Automatically summarizes conversations for easy reference
- ✅ **Configurable Retention**: Keep conversations for 30 days, 90 days, or forever
- ✅ **Full Control**: View, search, and export your conversation history anytime
- ✅ **Open Source**: You control your data, no corporate surveillance

### Memory Features

**Automatic Conversation Persistence:**
- All conversations are automatically saved and loaded on startup
- Watkins can reference past discussions: "As we talked about before..."
- Conversation summaries generated using local LLM (to save costs)

**Browse Your History:**
```bash
# View recent conversations
./tools/memory_viewer.py list

# View specific conversation
./tools/memory_viewer.py view 5

# Search for keywords
./tools/memory_viewer.py search "weather"

# Export conversation to file
./tools/memory_viewer.py export 3 conversation.txt

# View statistics
./tools/memory_viewer.py stats
```

**Privacy Comparison:**

| Feature | Watkins | Alexa |
|---------|---------|-------|
| Storage Location | Local (your Pi) | Amazon's cloud servers |
| Data Access | You only | Amazon + partners |
| Retention Control | You decide | Amazon decides |
| Conversation Export | Yes, anytime | Limited/difficult |
| Open Source | Yes | No |
| Cloud Surveillance | Never | Always |

All conversation data is stored in `logs/conversation_full_history.jsonl` on your device. See [PRIVACY.md](PRIVACY.md) for full details.

## Architecture

```
┌─────────────────────────────────────────┐
│          Watkins Voice Assistant         │
├─────────────────────────────────────────┤
│  Audio I/O → VAD → STT → LLM → TTS     │
│      ↓                                   │
│  Wake Word Detection (Optional)          │
└─────────────────────────────────────────┘
```

### Components

- **AudioManager**: Handles microphone input and speaker output
- **VADProcessor**: Detects speech vs silence (Silero VAD)
- **STTEngine**: Speech-to-text (Faster-Whisper)
- **LLMClient**: Language model inference (Anthropic/Ollama)
- **TTSEngine**: Text-to-speech (Piper)
- **WakeWordDetector**: Wake word detection (Picovoice Porcupine)
- **ConversationManager**: Manages conversation history and context

## Hardware Requirements

- **Raspberry Pi 5** (8GB RAM recommended, 4GB minimum)
- **USB Microphone** or compatible audio HAT
- **Speaker** (3.5mm, HDMI audio, or USB speaker)
- **Internet connection** (optional, for cloud LLM)

## Software Requirements

- Raspberry Pi OS (64-bit, Bookworm or later)
- Python 3.9+
- 10GB+ free disk space (for models)

## Installation

### 1. System Dependencies

The system dependencies are already installed! They include:
- portaudio19-dev
- alsa-utils
- python3-venv
- python3-dev
- build-essential

### 2. Python Environment

The Python virtual environment is already set up in `venv/`:

```bash
# Activate the environment (for manual use)
source venv/bin/activate
```

### 3. Configure API Keys

Edit the `.env` file and add your API keys:

```bash
nano .env
```

Add your keys:
```
# For cloud LLM (optional)
ANTHROPIC_API_KEY=your_anthropic_key_here

# For wake word detection (required for hands-free mode)
PORCUPINE_ACCESS_KEY=your_porcupine_key_here
```

**Get API Keys:**
- Anthropic Claude: https://console.anthropic.com/
- Picovoice Porcupine: https://console.picovoice.ai/ (free tier available)

### 4. Configure Settings

Edit `config/config.yaml` to customize Watkins:

```yaml
# Example: Switch to local-only mode
llm:
  mode: "local"  # Options: cloud, local, hybrid
```

## Audio Setup

### Test Your Audio Devices

Run the audio configuration utility:

```bash
./venv/bin/python tools/audio_config.py
```

This will:
1. List all available audio devices
2. Test your microphone
3. Test your speakers

### Troubleshooting Audio

If your microphone isn't detected:
1. Check USB connection
2. Run `arecord -l` to list recording devices
3. Run `aplay -l` to list playback devices
4. Update device IDs in `config/config.yaml` if needed

## Usage

### Running Watkins

Start Watkins:

```bash
./watkins.py
```

Or with the virtual environment:

```bash
./venv/bin/python watkins.py
```

### Interaction Modes

**With Wake Word (Hands-Free)**:
1. Say the wake word: "Porcupine" (or configured word)
2. Wait for confirmation
3. Speak your question
4. Listen to Watkins' response

**Without Wake Word (Push-to-Talk)**:
- Watkins will continuously listen and respond
- Press Ctrl+C to exit

### Example Conversation

```
You: Porcupine... What's the weather like today?
Watkins: I'm sorry, I don't have access to real-time weather data,
         but I can help you with other questions!

You: Tell me a joke
Watkins: Why did the robot go to therapy? It had too many bugs!
```

## Configuration

### Key Configuration Options

**Audio Settings** (`config/config.yaml`):
```yaml
audio:
  sample_rate: 16000    # Audio sample rate
  input_device: null    # null = default, or device ID
  output_device: null   # null = default, or device ID
```

**LLM Mode**:
```yaml
llm:
  mode: "hybrid"  # cloud, local, or hybrid

  cloud:
    model: "claude-3-5-sonnet-20241022"
    max_tokens: 150

  local:
    model: "phi3.5"
    host: "http://localhost:11434"
```

**Wake Word**:
```yaml
wake_word:
  enabled: true
  keyword: "porcupine"  # porcupine, bumblebee, americano, etc.
  sensitivity: 0.5      # 0.0 to 1.0
```

## Models

### Pre-installed Models

- **Ollama**: Phi-3.5 (3.8B parameters)
- **Faster-Whisper**: Base model (auto-downloads on first use)
- **Piper TTS**: Will download voice model on first synthesis

### Changing Models

**Switch Whisper Model**:
```yaml
stt:
  model_size: "small"  # tiny, base, small, medium
```

**Switch Ollama Model**:
```bash
ollama pull tinyllama  # Download a different model
```

Then update `config/config.yaml`:
```yaml
llm:
  local:
    model: "tinyllama"
```

## Performance

### Expected Latency (8GB Pi 5)

- Wake word detection: <50ms
- VAD response: <20ms
- STT (3-5 seconds of speech): 100-300ms
- Local LLM inference: 200ms-2s
- TTS synthesis: 100-300ms
- **Total (local mode)**: ~500ms-3s

### Optimization Tips

1. **Use smaller models**: Tiny/Base Whisper, TinyLlama
2. **Reduce max_tokens**: Shorter responses = faster
3. **Disable wake word**: If you don't need hands-free
4. **Use cloud LLM**: Claude is much faster than local

## Troubleshooting

### Common Issues

**"No PORCUPINE_ACCESS_KEY found"**
- Get a free key at https://console.picovoice.ai/
- Add it to `.env` file

**"Could not connect to Ollama"**
- Check if Ollama is running: `ollama list`
- Restart Ollama: `systemctl restart ollama`
- Or run in cloud-only mode

**"Failed to load Whisper model"**
- Models download automatically on first use
- Check disk space: `df -h`
- Check internet connection

**"No audio devices detected"**
- Plug in your USB microphone
- Run: `./venv/bin/python tools/audio_config.py`
- Check `arecord -l` and `aplay -l`

**High CPU usage**
- Use smaller models (tiny whisper, tinyllama)
- Reduce `max_tokens` in config
- Use cloud LLM instead of local

## Project Structure

```
Watkins/
├── watkins.py              # Main application
├── config/
│   └── config.yaml         # Configuration file
├── src/                    # Core modules
│   ├── audio_manager.py
│   ├── vad_processor.py
│   ├── stt_engine.py
│   ├── llm_client.py
│   ├── tts_engine.py
│   ├── wake_word_detector.py
│   └── conversation_manager.py
├── tools/
│   └── audio_config.py     # Audio testing utility
├── models/                 # Model cache (auto-created)
├── logs/                   # Application logs
├── .env                    # API keys (DO NOT COMMIT)
├── .env.template           # API key template
└── requirements.txt        # Python dependencies
```

## Development

### Testing Individual Modules

Each module can be tested independently:

```bash
# Test audio
./venv/bin/python src/audio_manager.py

# Test VAD
./venv/bin/python src/vad_processor.py

# Test STT
./venv/bin/python src/stt_engine.py

# Test LLM
./venv/bin/python src/llm_client.py

# Test TTS
./venv/bin/python src/tts_engine.py
```

### Adding Custom Features

The modular architecture makes it easy to extend Watkins:

1. **Custom Wake Words**: Get Picovoice custom wake word license
2. **New LLM Providers**: Add to `llm_client.py`
3. **Enhanced Audio Processing**: Modify `audio_manager.py`
4. **Skills/Commands**: Extend `conversation_manager.py`

## Contributing

Contributions are welcome! Areas for improvement:

- Add more LLM providers (OpenAI, local models)
- Implement custom skills/commands
- Add multi-language support
- Improve error handling
- Add unit tests
- Create web interface

## License

MIT License - feel free to modify and distribute!

## Acknowledgments

Built with:
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Picovoice Porcupine](https://picovoice.ai/)
- [Piper TTS](https://github.com/rhasspy/piper)
- [Ollama](https://ollama.ai/)
- [Anthropic Claude](https://www.anthropic.com/)

## Support

For issues and questions:
- Check the troubleshooting section
- Review configuration in `config/config.yaml`
- Check logs in `logs/watkins.log`

---

**Enjoy using Watkins! 🤖**
