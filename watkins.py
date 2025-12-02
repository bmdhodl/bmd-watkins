#!/home/admin/Desktop/Repos/Watkins/venv/bin/python3
"""
Watkins Voice Assistant
Main application orchestrator
"""

import sys
import os
import logging
import yaml
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from audio_manager import AudioManager
from vad_processor import VADProcessor
from stt_engine import STTEngine
from llm_client import LLMClient
from tts_engine import TTSEngine
from wake_word_detector import ContinuousWakeWordDetector
from conversation_manager import ConversationManager


class Watkins:
    """Watkins Voice Assistant"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize Watkins

        Args:
            config_path: Path to configuration file
        """
        # Load environment variables
        load_dotenv()

        # Load configuration
        self.config = self._load_config(config_path)

        # Setup logging
        self._setup_logging()

        self.logger = logging.getLogger(__name__)
        self.logger.info("=" * 60)
        self.logger.info(" WATKINS VOICE ASSISTANT")
        self.logger.info("=" * 60)

        # Initialize components
        self.logger.info("Initializing components...")
        self._initialize_components()

        self.is_running = False
        self.logger.info("Watkins initialized successfully!")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default configuration")
            return self._default_config()

    def _default_config(self) -> dict:
        """Return default configuration"""
        return {
            "audio": {"sample_rate": 16000, "channels": 1, "chunk_size": 1024},
            "wake_word": {"enabled": True, "keyword": "porcupine", "sensitivity": 0.5},
            "vad": {"enabled": True, "threshold": 0.5},
            "stt": {"model_size": "base", "device": "cpu", "compute_type": "int8"},
            "llm": {"mode": "hybrid", "cloud": {}, "local": {}},
            "tts": {"model": "en_US-lessac-medium"},
            "conversation": {"max_history": 10, "timeout_seconds": 30},
            "logging": {"level": "INFO", "console": True}
        }

    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))

        # Create logs directory
        os.makedirs("logs", exist_ok=True)

        # Configure logging
        handlers = []

        if log_config.get("console", True):
            handlers.append(logging.StreamHandler())

        if log_config.get("file"):
            handlers.append(logging.FileHandler(log_config["file"]))

        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=handlers
        )

    def _initialize_components(self):
        """Initialize all Watkins components"""
        # Audio Manager
        audio_config = self.config.get("audio", {})
        self.audio = AudioManager(
            sample_rate=audio_config.get("sample_rate", 16000),
            channels=audio_config.get("channels", 1),
            chunk_size=audio_config.get("chunk_size", 1024),
            input_device=audio_config.get("input_device"),
            output_device=audio_config.get("output_device")
        )

        # VAD Processor
        vad_config = self.config.get("vad", {})
        self.vad = VADProcessor(
            threshold=vad_config.get("threshold", 0.5),
            min_speech_duration_ms=vad_config.get("min_speech_duration_ms", 250),
            max_speech_duration_s=vad_config.get("max_speech_duration_s", 15),
            sample_rate=audio_config.get("sample_rate", 16000)
        )

        # STT Engine
        stt_config = self.config.get("stt", {})
        self.stt = STTEngine(
            model_size=stt_config.get("model_size", "base"),
            device=stt_config.get("device", "cpu"),
            compute_type=stt_config.get("compute_type", "int8"),
            language=stt_config.get("language", "en")
        )

        # LLM Client
        llm_config = self.config.get("llm", {})
        cloud_config = llm_config.get("cloud", {})
        local_config = llm_config.get("local", {})

        self.llm = LLMClient(
            mode=llm_config.get("mode", "hybrid"),
            cloud_provider=cloud_config.get("provider", "anthropic"),
            cloud_model=cloud_config.get("model", "claude-3-5-sonnet-20241022"),
            cloud_api_key=os.getenv("ANTHROPIC_API_KEY"),
            local_host=local_config.get("host", "http://localhost:11434"),
            local_model=local_config.get("model", "phi3.5"),
            max_tokens=cloud_config.get("max_tokens", 150),
            temperature=cloud_config.get("temperature", 0.7),
            system_prompt=llm_config.get("system_prompt")
        )

        # TTS Engine
        tts_config = self.config.get("tts", {})
        self.tts = TTSEngine(
            model=tts_config.get("model", "en_US-lessac-medium"),
            sample_rate=tts_config.get("sample_rate", 22050),
            speed=tts_config.get("speed", 1.0)
        )

        # Wake Word Detector
        wake_config = self.config.get("wake_word", {})
        if wake_config.get("enabled", True):
            porcupine_key = os.getenv("PORCUPINE_ACCESS_KEY")
            if porcupine_key:
                self.wake_detector = ContinuousWakeWordDetector(
                    access_key=porcupine_key,
                    keyword=wake_config.get("keyword", "porcupine"),
                    sensitivity=wake_config.get("sensitivity", 0.5)
                )
            else:
                self.logger.warning("No Porcupine access key, wake word detection disabled")
                self.wake_detector = None
        else:
            self.wake_detector = None

        # Conversation Manager
        conv_config = self.config.get("conversation", {})
        self.conversation = ConversationManager(
            max_history=conv_config.get("max_history", 10),
            timeout_seconds=conv_config.get("timeout_seconds", 30),
            save_history=conv_config.get("save_history", False)
        )

    def listen_for_speech(self) -> str:
        """Listen for speech and transcribe it

        Note: Recording must already be active when this is called.
        Recording state is managed by the main loop (run_with_wake_word or run_continuous).
        """
        self.logger.info("Listening...")

        # Collect audio until silence
        audio_chunks = []
        silence_count = 0
        max_silence_chunks = 10  # ~0.6s of silence after speech ends
        speech_started = False

        # Recording should already be active - just consume chunks
        # Two-phase approach:
        # 1. Wait for initial speech (no timeout)
        # 2. After speech starts, count silence to detect end
        while True:
            chunk = self.audio.get_audio_chunk(timeout=0.5)

            if chunk is None:
                continue

            audio_chunks.append(chunk)

            # Check for speech
            if self.vad.is_speech(chunk):
                speech_started = True
                silence_count = 0
            else:
                # Only count silence AFTER we've heard speech
                if speech_started:
                    silence_count += 1

            # Exit conditions:
            # 1. If speech started and we've had enough silence, stop
            if speech_started and silence_count >= max_silence_chunks:
                break

            # 2. Stop after reasonable amount of audio (timeout)
            if len(audio_chunks) > 300:  # ~20s max
                break

        # Concatenate audio
        if audio_chunks:
            audio = np.concatenate(audio_chunks)

            # Extract speech segments
            speech_audio = self.vad.extract_speech(audio)

            if speech_audio is not None and len(speech_audio) > 0:
                # Transcribe
                text = self.stt.transcribe(speech_audio)
                return text

        return ""

    def process_query(self, user_input: str) -> str:
        """Process user query and generate response"""
        if not user_input:
            return ""

        # Add to conversation
        self.conversation.add_user_message(user_input)

        # Get response from LLM
        history = self.conversation.get_recent_context(num_turns=5)
        response = self.llm.generate_response(user_input, history)

        # Add assistant response to conversation
        self.conversation.add_assistant_message(response)

        return response

    def speak(self, text: str):
        """Speak text using TTS"""
        if not text:
            return

        self.logger.info(f"Speaking: {text}")

        # Synthesize speech
        audio = self.tts.synthesize(text)

        if len(audio) > 0:
            # Play audio
            self.audio.play_audio(audio, sample_rate=self.tts.sample_rate)

    def handle_interaction(self):
        """Handle a single voice interaction"""
        self.logger.info("\n--- Listening for your question ---")

        # Listen for speech
        user_input = self.listen_for_speech()

        if user_input:
            self.logger.info(f"You said: {user_input}")

            # Process and respond
            response = self.process_query(user_input)

            if response:
                self.logger.info(f"Watkins: {response}")
                self.speak(response)
        else:
            self.logger.info("No speech detected")

    def run(self):
        """Run Watkins main loop"""
        self.is_running = True

        try:
            if self.wake_detector:
                self.logger.info("\nWatkins is ready! Say the wake word to activate.")
                self.logger.info(f"Wake word: '{self.config['wake_word']['keyword']}'")
                self.run_with_wake_word()
            else:
                self.logger.info("\nWatkins is ready! (Push-to-talk mode)")
                self.run_continuous()

        except KeyboardInterrupt:
            self.logger.info("\nShutting down...")
        finally:
            self.cleanup()

    def run_with_wake_word(self):
        """Run with wake word detection"""
        self.audio.start_recording()

        try:
            while self.is_running:
                # Get audio chunk
                chunk = self.audio.get_audio_chunk(timeout=0.5)

                if chunk is None:
                    continue

                # Check for wake word
                if self.wake_detector.add_audio(chunk):
                    self.logger.info("Wake word detected!")

                    # Clear audio buffer
                    self.audio.clear_audio_queue()

                    # Enter conversation mode
                    conversation_active = True
                    conversation_timeout = 5  # seconds

                    while conversation_active and self.is_running:
                        # Handle the current interaction
                        self.handle_interaction()

                        # Wait for follow-up speech (5-second window)
                        self.logger.info("Listening for follow-up (5s window)...")
                        import time
                        start_time = time.time()
                        heard_speech = False
                        audio_buffer = []

                        # Collect audio for up to 5 seconds, checking for speech
                        while time.time() - start_time < conversation_timeout:
                            chunk = self.audio.get_audio_chunk(timeout=0.5)

                            if chunk is None:
                                continue

                            audio_buffer.append(chunk)

                            # Check if speech detected
                            if self.vad.is_speech(chunk):
                                heard_speech = True
                                break

                        if heard_speech:
                            # User spoke within 5 seconds, continue conversation
                            self.logger.info("Follow-up detected, continuing conversation...")
                            # Clear the audio buffer to prepare for next question
                            self.audio.clear_audio_queue()
                            # Loop continues to handle_interaction
                        else:
                            # 5 seconds passed with no speech, exit conversation mode
                            self.logger.info("No follow-up detected, returning to wake word mode")
                            conversation_active = False

        finally:
            self.audio.stop_recording()

    def run_continuous(self):
        """Run in continuous mode (no wake word)"""
        while self.is_running:
            # Start recording for this interaction
            self.audio.start_recording()

            try:
                self.handle_interaction()
            finally:
                # Stop recording after interaction
                self.audio.stop_recording()

            # Small delay
            import time
            time.sleep(0.5)

    def cleanup(self):
        """Clean up resources"""
        self.logger.info("Cleaning up...")

        if hasattr(self, 'audio'):
            self.audio.cleanup()

        if hasattr(self, 'wake_detector') and self.wake_detector:
            self.wake_detector.cleanup()

        self.logger.info("Watkins shut down successfully")


def main():
    """Main entry point"""
    print("\nWatkins Voice Assistant")
    print("========================\n")

    # Check for config file
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}")
        print("Using default configuration\n")

    # Create and run Watkins
    watkins = Watkins(config_path)
    watkins.run()


if __name__ == "__main__":
    main()
