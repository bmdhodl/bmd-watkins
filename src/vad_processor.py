"""
Voice Activity Detection (VAD) Processor for Watkins
Uses Silero VAD for detecting speech in audio
"""

import torch
import numpy as np
import logging
from typing import List, Tuple, Optional
from silero_vad import load_silero_vad, get_speech_timestamps


class VADProcessor:
    """Voice Activity Detection using Silero VAD"""

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: int = 15,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 30,
        sample_rate: int = 16000
    ):
        """
        Initialize VAD Processor

        Args:
            threshold: Speech threshold (0.0 to 1.0)
            min_speech_duration_ms: Minimum speech duration to detect
            max_speech_duration_s: Maximum speech duration
            min_silence_duration_ms: Minimum silence to end speech
            speech_pad_ms: Padding around speech segments
            sample_rate: Audio sample rate (8000 or 16000)
        """
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.max_speech_duration_s = max_speech_duration_s
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.sample_rate = sample_rate

        self.logger = logging.getLogger(__name__)

        # Load Silero VAD model
        try:
            self.model = load_silero_vad()
            self.logger.info("Silero VAD model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load Silero VAD: {e}")
            raise

    def is_speech(self, audio: np.ndarray) -> bool:
        """
        Check if audio contains speech

        Args:
            audio: Audio data as numpy array

        Returns:
            True if speech detected, False otherwise
        """
        # Convert to torch tensor
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio

        # Get speech probability
        try:
            speech_prob = self.model(audio_tensor, self.sample_rate).item()
            return speech_prob > self.threshold
        except Exception as e:
            self.logger.error(f"Error in speech detection: {e}")
            return False

    def get_speech_timestamps(
        self,
        audio: np.ndarray,
        return_seconds: bool = False
    ) -> List[dict]:
        """
        Get timestamps of speech segments in audio

        Args:
            audio: Audio data as numpy array
            return_seconds: Return timestamps in seconds instead of samples

        Returns:
            List of dicts with 'start' and 'end' timestamps
        """
        # Convert to torch tensor
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio

        try:
            timestamps = get_speech_timestamps(
                audio_tensor,
                self.model,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_duration_ms,
                max_speech_duration_s=self.max_speech_duration_s,
                min_silence_duration_ms=self.min_silence_duration_ms,
                speech_pad_ms=self.speech_pad_ms,
                return_seconds=return_seconds,
                sampling_rate=self.sample_rate
            )

            self.logger.debug(f"Found {len(timestamps)} speech segments")
            return timestamps

        except Exception as e:
            self.logger.error(f"Error getting speech timestamps: {e}")
            return []

    def extract_speech(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract speech segments from audio

        Args:
            audio: Audio data as numpy array

        Returns:
            Concatenated speech segments, or None if no speech detected
        """
        timestamps = self.get_speech_timestamps(audio, return_seconds=False)

        if not timestamps:
            self.logger.debug("No speech detected in audio")
            return None

        # Extract and concatenate speech segments
        speech_segments = []
        for ts in timestamps:
            start = ts['start']
            end = ts['end']
            segment = audio[start:end]
            speech_segments.append(segment)

        if speech_segments:
            speech_audio = np.concatenate(speech_segments)
            self.logger.debug(
                f"Extracted {len(speech_audio)/self.sample_rate:.2f}s of speech"
            )
            return speech_audio

        return None

    def collect_speech_until_silence(
        self,
        audio_generator,
        max_duration: Optional[float] = None
    ) -> List[np.ndarray]:
        """
        Collect audio chunks from generator until silence detected

        Args:
            audio_generator: Generator yielding audio chunks
            max_duration: Maximum duration in seconds (None for unlimited)

        Returns:
            List of audio chunks containing speech
        """
        if max_duration is None:
            max_duration = self.max_speech_duration_s

        chunks = []
        total_samples = 0
        silence_samples = 0
        silence_threshold_samples = int(
            self.min_silence_duration_ms * self.sample_rate / 1000
        )
        max_samples = int(max_duration * self.sample_rate)

        for chunk in audio_generator:
            if self.is_speech(chunk):
                chunks.append(chunk)
                silence_samples = 0
            else:
                silence_samples += len(chunk)
                if silence_samples >= silence_threshold_samples:
                    # Sufficient silence detected, stop collecting
                    break

            total_samples += len(chunk)
            if total_samples >= max_samples:
                # Maximum duration reached
                break

        self.logger.debug(f"Collected {len(chunks)} audio chunks")
        return chunks

    def get_speech_probability(self, audio: np.ndarray) -> float:
        """
        Get speech probability for audio chunk

        Args:
            audio: Audio data as numpy array

        Returns:
            Speech probability (0.0 to 1.0)
        """
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio

        try:
            prob = self.model(audio_tensor, self.sample_rate).item()
            return float(prob)
        except Exception as e:
            self.logger.error(f"Error getting speech probability: {e}")
            return 0.0


if __name__ == "__main__":
    # Test the VAD Processor
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    vad = VADProcessor()

    # Create test audio: 1 second of silence, 2 seconds of simulated speech
    sample_rate = 16000
    silence = np.zeros(sample_rate)
    speech = np.random.randn(sample_rate * 2) * 0.1  # Simulated speech
    test_audio = np.concatenate([silence, speech, silence])

    logger.info(f"Testing VAD on {len(test_audio)/sample_rate:.1f}s audio")

    # Test speech detection
    timestamps = vad.get_speech_timestamps(test_audio, return_seconds=True)
    logger.info(f"Speech timestamps: {timestamps}")

    # Test speech extraction
    extracted = vad.extract_speech(test_audio)
    if extracted is not None:
        logger.info(f"Extracted {len(extracted)/sample_rate:.2f}s of speech")
    else:
        logger.info("No speech extracted")
