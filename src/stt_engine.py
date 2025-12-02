"""
Speech-to-Text Engine for Watkins
Uses Faster-Whisper for efficient speech recognition
"""

import numpy as np
import logging
from typing import Optional, List, Dict
from faster_whisper import WhisperModel
import time


class STTEngine:
    """Speech-to-Text using Faster-Whisper"""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
        beam_size: int = 5,
        vad_filter: bool = True,
        sample_rate: int = 16000
    ):
        """
        Initialize STT Engine

        Args:
            model_size: Model size (tiny, base, small, medium, large)
            device: Device to run on (cpu, cuda)
            compute_type: Computation type (int8, float16, float32)
            language: Language code (en, es, fr, etc.)
            beam_size: Beam size for decoding
            vad_filter: Use VAD to filter silence
            sample_rate: Audio sample rate
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.sample_rate = sample_rate

        self.logger = logging.getLogger(__name__)

        # Load Whisper model
        self.logger.info(f"Loading Faster-Whisper model: {model_size}")
        start_time = time.time()

        try:
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type
            )
            load_time = time.time() - start_time
            self.logger.info(f"Whisper model loaded in {load_time:.2f}s")

        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> str:
        """
        Transcribe audio to text

        Args:
            audio: Audio data as numpy array
            language: Language code (None to use default)
            task: Task type (transcribe or translate)

        Returns:
            Transcribed text
        """
        if language is None:
            language = self.language

        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        try:
            start_time = time.time()

            # Transcribe
            segments, info = self.model.transcribe(
                audio,
                language=language,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                task=task
            )

            # Combine segments
            text = " ".join([segment.text for segment in segments])
            text = text.strip()

            transcribe_time = time.time() - start_time
            audio_duration = len(audio) / self.sample_rate

            self.logger.info(
                f"Transcribed {audio_duration:.2f}s audio in {transcribe_time:.2f}s "
                f"(RTF: {transcribe_time/audio_duration:.2f}x)"
            )
            self.logger.debug(f"Transcription: '{text}'")

            return text

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            return ""

    def transcribe_with_timestamps(
        self,
        audio: np.ndarray,
        language: Optional[str] = None
    ) -> List[Dict]:
        """
        Transcribe audio with word-level timestamps

        Args:
            audio: Audio data as numpy array
            language: Language code (None to use default)

        Returns:
            List of segments with text and timestamps
        """
        if language is None:
            language = self.language

        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        try:
            segments, info = self.model.transcribe(
                audio,
                language=language,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                word_timestamps=True
            )

            results = []
            for segment in segments:
                results.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text
                })

            return results

        except Exception as e:
            self.logger.error(f"Transcription with timestamps failed: {e}")
            return []

    def detect_language(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Detect language of audio

        Args:
            audio: Audio data as numpy array

        Returns:
            Dict of language codes and probabilities
        """
        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        try:
            segments, info = self.model.transcribe(audio)

            language_probs = {}
            if hasattr(info, 'language'):
                language_probs[info.language] = info.language_probability

            self.logger.debug(f"Detected language: {language_probs}")
            return language_probs

        except Exception as e:
            self.logger.error(f"Language detection failed: {e}")
            return {}

    def is_speech_present(self, audio: np.ndarray, threshold: float = 0.5) -> bool:
        """
        Check if audio contains intelligible speech

        Args:
            audio: Audio data as numpy array
            threshold: Confidence threshold

        Returns:
            True if speech detected above threshold
        """
        text = self.transcribe(audio)
        return len(text.strip()) > 0


if __name__ == "__main__":
    # Test the STT Engine
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize with tiny model for testing
    stt = STTEngine(model_size="tiny")

    # Create test audio (2 seconds of random noise as placeholder)
    sample_rate = 16000
    test_audio = np.random.randn(sample_rate * 2).astype(np.float32) * 0.01

    logger.info("Testing STT on sample audio...")
    text = stt.transcribe(test_audio)
    logger.info(f"Result: '{text}'")

    # Test with timestamps
    segments = stt.transcribe_with_timestamps(test_audio)
    logger.info(f"Segments with timestamps: {len(segments)}")
