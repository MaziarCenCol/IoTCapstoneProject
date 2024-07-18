import subprocess
import requests
import time
import uuid


def list_audio_devices():
    result = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
    print(result.stdout)


def record_audio(filename, duration, card, device):
    command = [
        'arecord',
        '-D', f'plughw:{card},{device}',
        '-f', 'cd',
        '-t', 'wav',
        '-d', str(duration),
        '-r', '44100',
        filename
    ]
    subprocess.run(command)
    print(f"Saved to {filename}")


def amplify_audio(input_filename, output_filename, gain):
    command = [
        'sox', input_filename, output_filename, 'gain', str(gain)
    ]
    subprocess.run(command)
    print(f"Amplified audio saved to {output_filename}")


def send_audio_file(url, filepath):
    with open(filepath, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
        return response


def main():
    print("Available audio devices:")
    list_audio_devices()

    card = 3
    device = 0
    duration = 5  # seconds
    url = "http://10.0.0.226:5000/upload"  # Replace with your server URL

    while True:
        raw_filename = f"raw_recorded_audio_{uuid.uuid4().hex}.wav"
        amplified_filename = f"recorded_audio_{uuid.uuid4().hex}.wav"

        record_audio(raw_filename, duration, card, device)
        amplify_audio(raw_filename, amplified_filename, gain=10)  # Adjust gain as needed

        print("Playing recorded audio...")
        subprocess.run(['aplay', amplified_filename])

        response = send_audio_file(url, amplified_filename)
        print(f"Response from server: {response.status_code}, {response.text}")

        # Wait for 5 seconds before recording again
        time.sleep(5)


if __name__ == "__main__":
    main()
