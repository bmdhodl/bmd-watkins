# Watkins Setup Checklist

## ✅ Completed

- [x] System dependencies installed
- [x] Python virtual environment created
- [x] All Python packages installed
- [x] Ollama installed with Phi-3.5 model
- [x] Project structure created
- [x] Configuration files setup
- [x] **Porcupine API key configured** ✅

## 📋 Required Before Running

### 1. Hardware Setup
- [ ] **USB Microphone connected**
  - Run `arecord -l` to verify detection
  - Test with: `./venv/bin/python tools/audio_config.py`

- [ ] **Speaker connected** (when it arrives)
  - Run `aplay -l` to verify detection
  - Test audio output with the audio config tool

### 2. API Keys Status

#### ✅ Porcupine Wake Word (CONFIGURED)
- **Status**: ✅ Key added to `.env`
- **Required for**: Wake word detection ("Porcupine", "Bumblebee", etc.)
- **Works offline**: Yes, after initial validation

#### ⚠️ Anthropic Claude (OPTIONAL)
- **Status**: ⚠️ Not configured (optional)
- **Required for**: Cloud LLM mode (better responses)
- **Alternative**: Use local Ollama model (already installed)
- **Get key at**: https://console.anthropic.com/
- **Add to**: `.env` file → `ANTHROPIC_API_KEY=your_key_here`

## 🚀 Ready to Test!

### Mode 1: Local-Only (No additional API keys needed)
```bash
# Edit config to use local mode
nano config/config.yaml
# Change: llm.mode: "local"

# Run Watkins
./watkins.py
```

### Mode 2: Hybrid (Recommended, but needs Claude key)
```bash
# Add Claude API key to .env first
nano .env

# Run Watkins with default config
./watkins.py
```

### Mode 3: Test Without Wake Word First
```bash
# Edit config
nano config/config.yaml
# Change: wake_word.enabled: false

# Run Watkins (push-to-talk mode)
./watkins.py
```

## 🔧 Pre-Flight Checklist

Before running Watkins:
1. [ ] Microphone plugged in and working
2. [ ] Tested audio with: `./venv/bin/python tools/audio_config.py`
3. [ ] Decided on LLM mode (local vs cloud)
4. [ ] Added Claude key if using cloud/hybrid mode
5. [ ] Verified Ollama is running: `ollama list`

## 📝 API Key Summary

| Service | Status | Required? | Purpose | Get Key |
|---------|--------|-----------|---------|---------|
| **Porcupine** | ✅ Configured | Yes (for wake word) | Wake word detection | https://console.picovoice.ai/ |
| **Anthropic** | ⚠️ Optional | No (use local instead) | Better LLM responses | https://console.anthropic.com/ |

## 🎯 Next Steps

1. **Test your microphone**:
   ```bash
   ./venv/bin/python tools/audio_config.py
   ```

2. **Choose your mode**:
   - **Full Local**: Edit `config/config.yaml` → set `llm.mode: "local"`
   - **Cloud/Hybrid**: Add Claude key to `.env`

3. **Run Watkins**:
   ```bash
   ./watkins.py
   ```

4. **Say the wake word**: "Porcupine" (then ask a question!)

## ⚠️ Known Blockers

None! Everything is ready to run with:
- ✅ Wake word detection (Porcupine key configured)
- ✅ Local LLM (Ollama/Phi-3.5 installed)
- ⏳ Just need: Microphone connected + audio tested

**Optional enhancement**:
- Add Claude API key for better responses (but not required!)
