"""
Audio Manager Module for Watkins Voice Assistant
Handles microphone input and speaker output
"""

import sounddevice as sd
import numpy as np
import logging
from typing import Optional, Callable
import queue
import threading


class AudioManager:
    """Manages audio input/output streams"""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None
    ):
        """
        Initialize Audio Manager

        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1=mono, 2=stereo)
            chunk_size: Size of audio chunks
            input_device: Input device ID (None for default)
            output_device: Output device ID (None for default)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.input_device = input_device
        self.output_device = output_device

        self.logger = logging.getLogger(__name__)

        # Audio buffer queue
        self.audio_queue = queue.Queue()

        # Stream control
        self.input_stream = None
        self.output_stream = None
        self.is_recording = False

        self.logger.info(
            f"AudioManager initialized: {sample_rate}Hz, {channels}ch, chunk={chunk_size}"
        )

    def list_devices(self):
        """List all available audio devices"""
        devices = sd.query_devices()
        self.logger.info("Available audio devices:")
        for idx, device in enumerate(devices):
            device_type = []
            if device['max_input_channels'] > 0:
                device_type.append("INPUT")
            if device['max_output_channels'] > 0:
                device_type.append("OUTPUT")
            self.logger.info(
                f"  [{idx}] {device['name']} ({' & '.join(device_type)})"
            )
        return devices

    def _audio_callback(self, indata, frames, time, status):
        """Callback function for audio input stream"""
        if status:
            self.logger.warning(f"Audio callback status: {status}")

        if self.is_recording:
            # Convert to float32 and put in queue
            audio_data = indata.copy().flatten()
            self.audio_queue.put(audio_data)

    def start_recording(self):
        """Start recording audio from microphone"""
        if self.is_recording:
            self.logger.warning("Already recording")
            return

        self.is_recording = True

        try:
            self.input_stream = sd.InputStream(
                device=self.input_device,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=self._audio_callback,
                dtype=np.float32
            )
            self.input_stream.start()
            self.logger.info("Recording started")

        except Exception as e:
            self.logger.error(f"Failed to start recording: {e}")
            self.is_recording = False
            raise

    def stop_recording(self):
        """Stop recording audio"""
        if not self.is_recording:
            return

        self.is_recording = False

        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None

        self.logger.info("Recording stopped")

    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get next audio chunk from queue

        Args:
            timeout: Timeout in seconds

        Returns:
            Audio chunk as numpy array, or None if timeout
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_audio_queue(self):
        """Clear all audio chunks from queue"""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def record_audio(self, duration: float) -> np.ndarray:
        """
        Record audio for a specific duration

        Args:
            duration: Recording duration in seconds

        Returns:
            Recorded audio as numpy array
        """
        self.logger.debug(f"Recording {duration}s of audio")

        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            device=self.input_device,
            dtype=np.float32
        )
        sd.wait()

        return recording.flatten()

    def play_audio(self, audio: np.ndarray, sample_rate: Optional[int] = None):
        """
        Play audio through speaker

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate (uses manager's rate if None)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        try:
            sd.play(audio, sample_rate, device=self.output_device)
            sd.wait()
            self.logger.debug(f"Played {len(audio)/sample_rate:.2f}s of audio")

        except Exception as e:
            self.logger.error(f"Failed to play audio: {e}")
            raise

    def play_audio_async(self, audio: np.ndarray, sample_rate: Optional[int] = None):
        """
        Play audio asynchronously (non-blocking)

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate (uses manager's rate if None)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        try:
            sd.play(audio, sample_rate, device=self.output_device)
            self.logger.debug(f"Started async playback of {len(audio)/sample_rate:.2f}s")

        except Exception as e:
            self.logger.error(f"Failed to play audio async: {e}")
            raise

    def stop_playback(self):
        """Stop any ongoing audio playback"""
        sd.stop()
        self.logger.debug("Playback stopped")

    def get_input_level(self, duration: float = 0.1) -> float:
        """
        Get current microphone input level

        Args:
            duration: Duration to measure in seconds

        Returns:
            Average input level (0.0 to 1.0)
        """
        audio = self.record_audio(duration)
        level = np.abs(audio).mean()
        return float(level)

    def cleanup(self):
        """Clean up audio resources"""
        self.stop_recording()
        self.stop_playback()
        self.logger.info("AudioManager cleanup complete")


if __name__ == "__main__":
    # Test the AudioManager
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    am = AudioManager()
    am.list_devices()

    # Test microphone
    logger.info("Testing microphone for 3 seconds...")
    audio = am.record_audio(3.0)
    logger.info(f"Recorded {len(audio)/am.sample_rate:.2f}s, level: {np.abs(audio).mean():.4f}")

    # Test playback
    logger.info("Playing test tone...")
    t = np.linspace(0, 2, int(2 * am.sample_rate))
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    am.play_audio(tone)

    am.cleanup()
