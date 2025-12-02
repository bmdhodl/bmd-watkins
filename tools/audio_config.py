#!/usr/bin/env python3
"""
Audio Configuration Utility for Watkins Voice Assistant
Detects and tests audio input/output devices
"""

import sounddevice as sd
import numpy as np
import sys
import time


def list_audio_devices():
    """List all available audio devices"""
    print("\n" + "="*70)
    print(" AUDIO DEVICES DETECTED")
    print("="*70 + "\n")

    devices = sd.query_devices()

    print(f"Default Input Device:  {sd.default.device[0]}")
    print(f"Default Output Device: {sd.default.device[1]}\n")

    for idx, device in enumerate(devices):
        device_type = []
        if device['max_input_channels'] > 0:
            device_type.append("INPUT")
        if device['max_output_channels'] > 0:
            device_type.append("OUTPUT")

        print(f"[{idx}] {device['name']}")
        print(f"    Type: {' & '.join(device_type)}")
        print(f"    Channels: In={device['max_input_channels']}, Out={device['max_output_channels']}")
        print(f"    Sample Rate: {device['default_samplerate']} Hz")
        print()

    return devices


def get_input_devices():
    """Get list of input devices"""
    devices = sd.query_devices()
    input_devices = []

    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append((idx, device['name']))

    return input_devices


def get_output_devices():
    """Get list of output devices"""
    devices = sd.query_devices()
    output_devices = []

    for idx, device in enumerate(devices):
        if device['max_output_channels'] > 0:
            output_devices.append((idx, device['name']))

    return output_devices


def test_microphone(device_id=None, duration=3):
    """Test microphone by recording and showing audio levels"""
    print("\n" + "="*70)
    print(" MICROPHONE TEST")
    print("="*70 + "\n")

    if device_id is None:
        device_id = sd.default.device[0]

    device_info = sd.query_devices(device_id)
    print(f"Testing: {device_info['name']}")
    print(f"Recording for {duration} seconds...")
    print("Speak into the microphone!\n")

    sample_rate = int(device_info['default_samplerate'])

    try:
        # Record audio
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            device=device_id,
            dtype=np.float32
        )

        # Show real-time levels
        for i in range(duration):
            sd.wait(1000)  # Wait 1 second
            chunk = recording[i*sample_rate:(i+1)*sample_rate]
            if len(chunk) > 0:
                level = np.abs(chunk).mean()
                bars = int(level * 50)
                print(f"Level: [{'=' * bars}{' ' * (50-bars)}] {level:.4f}")

        sd.wait()  # Wait for recording to complete

        # Calculate statistics
        max_level = np.abs(recording).max()
        avg_level = np.abs(recording).mean()

        print(f"\nResults:")
        print(f"  Max Level: {max_level:.4f}")
        print(f"  Avg Level: {avg_level:.4f}")

        if avg_level < 0.001:
            print(f"  Status: ⚠️  WARNING - Very low input! Check microphone connection.")
        elif avg_level < 0.01:
            print(f"  Status: ⚠️  Low input - consider adjusting gain")
        else:
            print(f"  Status: ✓ Good signal detected!")

        return True

    except Exception as e:
        print(f"❌ Error testing microphone: {e}")
        return False


def test_speaker(device_id=None):
    """Test speaker by playing a tone"""
    print("\n" + "="*70)
    print(" SPEAKER TEST")
    print("="*70 + "\n")

    if device_id is None:
        device_id = sd.default.device[1]

    device_info = sd.query_devices(device_id)
    print(f"Testing: {device_info['name']}")
    print("Playing test tone (440 Hz for 2 seconds)...\n")

    sample_rate = int(device_info['default_samplerate'])
    duration = 2.0
    frequency = 440.0  # A4 note

    try:
        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration))
        tone = 0.3 * np.sin(2 * np.pi * frequency * t)

        # Play tone
        sd.play(tone, sample_rate, device=device_id)
        sd.wait()

        print("✓ Tone played successfully!")
        response = input("Did you hear the tone? (y/n): ").strip().lower()

        if response == 'y':
            print("✓ Speaker working correctly!")
            return True
        else:
            print("⚠️  Check speaker connection or volume")
            return False

    except Exception as e:
        print(f"❌ Error testing speaker: {e}")
        return False


def interactive_mode():
    """Interactive device selection and testing"""
    print("\n" + "="*70)
    print(" WATKINS AUDIO CONFIGURATION TOOL")
    print("="*70)

    # List devices
    devices = list_audio_devices()

    # Check for input devices
    input_devices = get_input_devices()
    if not input_devices:
        print("⚠️  WARNING: No input devices detected!")
        print("   Please connect your USB microphone and run this tool again.\n")
        return

    # Test microphone
    print("\n" + "-"*70)
    print("Select microphone to test:")
    for idx, name in input_devices:
        print(f"  [{idx}] {name}")

    try:
        mic_choice = input(f"\nEnter device number (default: {input_devices[0][0]}): ").strip()
        mic_id = int(mic_choice) if mic_choice else input_devices[0][0]
        test_microphone(mic_id)
    except (ValueError, KeyboardInterrupt):
        print("\nSkipping microphone test")

    # Test speaker
    output_devices = get_output_devices()
    if output_devices:
        print("\n" + "-"*70)
        print("Select speaker to test:")
        for idx, name in output_devices:
            print(f"  [{idx}] {name}")

        try:
            spk_choice = input(f"\nEnter device number (default: {output_devices[0][0]}): ").strip()
            spk_id = int(spk_choice) if spk_choice else output_devices[0][0]
            test_speaker(spk_id)
        except (ValueError, KeyboardInterrupt):
            print("\nSkipping speaker test")

    print("\n" + "="*70)
    print("Configuration complete!")
    print("="*70 + "\n")


def main():
    """Main function"""
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "list":
            list_audio_devices()
        elif command == "test-mic":
            device_id = int(sys.argv[2]) if len(sys.argv) > 2 else None
            test_microphone(device_id)
        elif command == "test-speaker":
            device_id = int(sys.argv[2]) if len(sys.argv) > 2 else None
            test_speaker(device_id)
        else:
            print(f"Unknown command: {command}")
            print("\nUsage:")
            print("  python audio_config.py              # Interactive mode")
            print("  python audio_config.py list         # List all devices")
            print("  python audio_config.py test-mic [device_id]")
            print("  python audio_config.py test-speaker [device_id]")
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
