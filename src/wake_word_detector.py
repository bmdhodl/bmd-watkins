"""
Wake Word Detector for Watkins
Uses Picovoice Porcupine for efficient wake word detection
"""

import numpy as np
import logging
from typing import Optional, Callable
import struct
import pvporcupine


class WakeWordDetector:
    """Wake Word Detection using Picovoice Porcupine"""

    def __init__(
        self,
        access_key: str,
        keyword: str = "porcupine",
        sensitivity: float = 0.5,
        sample_rate: int = 16000
    ):
        """
        Initialize Wake Word Detector

        Args:
            access_key: Picovoice access key
            keyword: Wake word keyword (porcupine, bumblebee, etc.)
            sensitivity: Detection sensitivity (0.0 to 1.0)
            sample_rate: Audio sample rate (must be 16000 for Porcupine)
        """
        self.access_key = access_key
        self.keyword = keyword
        self.sensitivity = sensitivity
        self.sample_rate = sample_rate

        self.logger = logging.getLogger(__name__)

        # Porcupine requires 16kHz
        if sample_rate != 16000:
            self.logger.warning(
                f"Porcupine requires 16kHz sample rate, got {sample_rate}. "
                "Audio will need resampling."
            )

        # Initialize Porcupine
        try:
            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=[keyword],
                sensitivities=[sensitivity]
            )

            self.frame_length = self.porcupine.frame_length
            self.logger.info(
                f"Porcupine initialized: keyword='{keyword}', "
                f"sensitivity={sensitivity}, frame_length={self.frame_length}"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize Porcupine: {e}")
            self.logger.info(
                "Make sure you have a valid Picovoice access key. "
                "Get one free at https://console.picovoice.ai/"
            )
            raise

    def process_frame(self, audio_frame: np.ndarray) -> bool:
        """
        Process audio frame for wake word detection

        Args:
            audio_frame: Audio frame (must be frame_length samples)

        Returns:
            True if wake word detected, False otherwise
        """
        # Ensure correct frame length
        if len(audio_frame) != self.frame_length:
            self.logger.warning(
                f"Frame length mismatch: expected {self.frame_length}, "
                f"got {len(audio_frame)}"
            )
            return False

        try:
            # Convert float32 to int16 with proper clipping
            if audio_frame.dtype == np.float32:
                # Clip to [-1.0, 1.0] range and convert to int16
                audio_clipped = np.clip(audio_frame, -1.0, 1.0)
                audio_int16 = (audio_clipped * 32767).astype(np.int16)
            else:
                audio_int16 = audio_frame.astype(np.int16)

            # Debug: Log audio level
            frame_level = np.abs(audio_int16).max()
            self.logger.debug(f"Processing frame: level={frame_level}")

            # Process frame
            keyword_index = self.porcupine.process(audio_int16)

            if keyword_index >= 0:
                self.logger.info(f"Wake word '{self.keyword}' detected!")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return False

    def process_audio_stream(
        self,
        audio_generator,
        on_wake_word: Optional[Callable] = None,
        max_iterations: Optional[int] = None
    ):
        """
        Process audio stream for wake word detection

        Args:
            audio_generator: Generator yielding audio chunks
            on_wake_word: Callback function when wake word detected
            max_iterations: Maximum iterations (None for unlimited)
        """
        buffer = np.array([], dtype=np.float32)
        iterations = 0

        for audio_chunk in audio_generator:
            # Add to buffer
            buffer = np.concatenate([buffer, audio_chunk])

            # Process complete frames
            while len(buffer) >= self.frame_length:
                frame = buffer[:self.frame_length]
                buffer = buffer[self.frame_length:]

                if self.process_frame(frame):
                    if on_wake_word:
                        on_wake_word()
                    return True

            iterations += 1
            if max_iterations and iterations >= max_iterations:
                break

        return False

    def get_frame_length(self) -> int:
        """Get required frame length for processing"""
        return self.frame_length

    def get_sample_rate(self) -> int:
        """Get required sample rate"""
        return 16000  # Porcupine always uses 16kHz

    def cleanup(self):
        """Clean up Porcupine resources"""
        if hasattr(self, 'porcupine') and self.porcupine:
            self.porcupine.delete()
            self.logger.info("Porcupine cleaned up")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()


class ContinuousWakeWordDetector:
    """Continuous wake word detection with buffering"""

    def __init__(
        self,
        access_key: str,
        keyword: str = "porcupine",
        sensitivity: float = 0.5
    ):
        """
        Initialize continuous wake word detector

        Args:
            access_key: Picovoice access key
            keyword: Wake word keyword
            sensitivity: Detection sensitivity
        """
        self.detector = WakeWordDetector(
            access_key=access_key,
            keyword=keyword,
            sensitivity=sensitivity
        )

        self.buffer = np.array([], dtype=np.float32)
        self.frame_length = self.detector.get_frame_length()
        self.is_running = False

        self.logger = logging.getLogger(__name__)

    def add_audio(self, audio: np.ndarray) -> bool:
        """
        Add audio and check for wake word

        Args:
            audio: Audio samples

        Returns:
            True if wake word detected
        """
        # Debug: Check audio level
        audio_level = np.abs(audio).max()
        if audio_level > 0.01:  # Only log if audio is non-trivial
            self.logger.debug(f"Audio chunk received: {len(audio)} samples, level={audio_level:.4f}")

        # Add to buffer
        self.buffer = np.concatenate([self.buffer, audio])

        # Process complete frames
        detected = False
        frames_processed = 0
        while len(self.buffer) >= self.frame_length:
            frame = self.buffer[:self.frame_length]
            self.buffer = self.buffer[self.frame_length:]
            frames_processed += 1

            if self.detector.process_frame(frame):
                detected = True

        if frames_processed > 0:
            self.logger.debug(f"Processed {frames_processed} frame(s)")

        return detected

    def reset_buffer(self):
        """Clear audio buffer"""
        self.buffer = np.array([], dtype=np.float32)

    def cleanup(self):
        """Clean up resources"""
        self.detector.cleanup()


if __name__ == "__main__":
    # Test the Wake Word Detector
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    import os
    from dotenv import load_dotenv

    load_dotenv()

    access_key = os.getenv("PORCUPINE_ACCESS_KEY")

    if not access_key:
        logger.error("No PORCUPINE_ACCESS_KEY found in environment")
        logger.info("Get a free access key at https://console.picovoice.ai/")
    else:
        try:
            detector = WakeWordDetector(
                access_key=access_key,
                keyword="porcupine"
            )

            logger.info(f"Frame length: {detector.get_frame_length()}")
            logger.info(f"Sample rate: {detector.get_sample_rate()}")

            # Create test audio frame
            frame_length = detector.get_frame_length()
            test_frame = np.random.randn(frame_length).astype(np.float32) * 0.1

            logger.info("Testing wake word detection on random audio...")
            detected = detector.process_frame(test_frame)
            logger.info(f"Detection result: {detected}")

            detector.cleanup()

        except Exception as e:
            logger.error(f"Test failed: {e}")
