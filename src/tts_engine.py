"""
Text-to-Speech Engine for Watkins
Uses Piper for fast, natural-sounding voice synthesis
"""

import numpy as np
import logging
from typing import Optional
import subprocess
import tempfile
import os
import wave
from pathlib import Path
from piper import PiperVoice


class TTSEngine:
    """Text-to-Speech using Piper"""

    def __init__(
        self,
        model: str = "en_US-lessac-medium",
        sample_rate: int = 22050,
        speed: float = 1.0,
        speaker: Optional[int] = None,
        models_dir: str = "models"
    ):
        """
        Initialize TTS Engine

        Args:
            model: Voice model name (e.g., "en_US-lessac-medium")
            sample_rate: Output sample rate
            speed: Speech speed (1.0 = normal)
            speaker: Speaker ID for multi-speaker models
            models_dir: Directory containing voice model files
        """
        self.model = model
        self.sample_rate = sample_rate
        self.speed = speed
        self.speaker = speaker
        self.models_dir = Path(models_dir)

        self.logger = logging.getLogger(__name__)

        # Convert model name to full path
        model_path = self.models_dir / f"{model}.onnx"

        # Initialize Piper voice
        try:
            if not model_path.exists():
                self.logger.warning(f"Model file not found: {model_path}")
                self.logger.info("Attempting to download voice model...")
                self._download_model(model)

            self.voice = PiperVoice.load(str(model_path))
            self.logger.info(f"Piper TTS initialized: {model} from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load Piper voice: {e}")
            if not model_path.exists():
                self.logger.info("Attempting to download voice model...")
                try:
                    # Try to download the model
                    self._download_model(model)
                    self.voice = PiperVoice.load(str(model_path))
                    self.logger.info(f"Voice model downloaded and loaded: {model}")
                except Exception as e2:
                    self.logger.error(f"Failed to download/load voice: {e2}")
                    raise
            else:
                raise

    def _download_model(self, model: str):
        """Download Piper voice model using piper.download_voices"""
        try:
            import sys
            from subprocess import run, CalledProcessError

            # Create models directory if it doesn't exist
            self.models_dir.mkdir(parents=True, exist_ok=True)

            # Use Piper's download utility
            self.logger.info(f"Downloading voice model: {model}")
            result = run(
                [sys.executable, "-m", "piper.download_voices", model, "--download-dir", str(self.models_dir)],
                capture_output=True,
                text=True,
                check=True
            )
            self.logger.info(f"Download successful: {model}")

        except CalledProcessError as e:
            self.logger.error(f"Failed to download model: {e.stderr}")
            raise Exception(f"Could not download voice model '{model}'. Run: python -m piper.download_voices {model} --download-dir {self.models_dir}")
        except Exception as e:
            self.logger.error(f"Download error: {e}")
            raise

    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize speech from text

        Args:
            text: Text to synthesize

        Returns:
            Audio data as numpy array
        """
        if not text or not text.strip():
            self.logger.warning("Empty text provided for synthesis")
            return np.array([], dtype=np.float32)

        try:
            # Clean text for speech
            text = text.strip()

            # Synthesize - returns iterable of AudioChunk objects
            audio_chunks = []

            for audio_chunk in self.voice.synthesize(text):
                # Extract the float array from the AudioChunk
                audio_chunks.append(audio_chunk.audio_float_array)

            # Combine chunks
            if audio_chunks:
                audio = np.concatenate(audio_chunks)
                audio = audio.astype(np.float32)

                # Normalize to [-1, 1]
                if audio.max() > 0:
                    audio = audio / np.abs(audio).max()

                self.logger.info(
                    f"Synthesized {len(text)} chars to {len(audio)/self.sample_rate:.2f}s audio"
                )
                self.logger.debug(f"Text: '{text}'")

                return audio
            else:
                self.logger.warning("No audio generated from synthesis")
                return np.array([], dtype=np.float32)

        except Exception as e:
            self.logger.error(f"Synthesis failed: {e}")
            return np.array([], dtype=np.float32)

    def synthesize_to_file(self, text: str, output_path: str):
        """
        Synthesize speech and save to WAV file

        Args:
            text: Text to synthesize
            output_path: Output WAV file path
        """
        audio = self.synthesize(text)

        if len(audio) == 0:
            self.logger.error("No audio to save")
            return

        try:
            # Convert to int16 for WAV
            audio_int16 = (audio * 32767).astype(np.int16)

            # Write WAV file
            with wave.open(output_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 2 bytes for int16
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())

            self.logger.info(f"Audio saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to save audio: {e}")
            raise

    def estimate_duration(self, text: str) -> float:
        """
        Estimate speech duration for text

        Args:
            text: Input text

        Returns:
            Estimated duration in seconds
        """
        # Rough estimate: ~150 words per minute, ~5 chars per word
        chars_per_second = (150 * 5) / 60
        duration = len(text) / chars_per_second
        return duration

    def split_long_text(self, text: str, max_length: int = 500) -> list:
        """
        Split long text into chunks for synthesis

        Args:
            text: Input text
            max_length: Maximum length per chunk

        Returns:
            List of text chunks
        """
        if len(text) <= max_length:
            return [text]

        # Split by sentences
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def synthesize_long_text(self, text: str, max_length: int = 500) -> np.ndarray:
        """
        Synthesize long text by splitting into chunks

        Args:
            text: Text to synthesize
            max_length: Maximum length per chunk

        Returns:
            Combined audio data
        """
        chunks = self.split_long_text(text, max_length)

        audio_segments = []
        for chunk in chunks:
            audio = self.synthesize(chunk)
            if len(audio) > 0:
                audio_segments.append(audio)

                # Add small pause between chunks
                pause = np.zeros(int(0.2 * self.sample_rate), dtype=np.float32)
                audio_segments.append(pause)

        if audio_segments:
            return np.concatenate(audio_segments)
        else:
            return np.array([], dtype=np.float32)


if __name__ == "__main__":
    # Test the TTS Engine
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        tts = TTSEngine()

        test_text = "Hello! I am Watkins, your voice assistant. How can I help you today?"
        logger.info(f"Synthesizing: '{test_text}'")

        audio = tts.synthesize(test_text)
        logger.info(f"Generated {len(audio)} samples ({len(audio)/tts.sample_rate:.2f}s)")

        # Save to file
        output_file = "test_tts.wav"
        tts.synthesize_to_file(test_text, output_file)
        logger.info(f"Saved to {output_file}")

    except Exception as e:
        logger.error(f"TTS test failed: {e}")
        logger.info("Piper TTS may need voice models to be downloaded first")
