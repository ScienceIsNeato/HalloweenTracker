import os
import random
import threading
import time  # Import time module for sleep
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio

class SoundPlayer:
    def __init__(self, sound_directory='sound_clips'):
        self.sound_directory = sound_directory
        self.sound_files = self._get_sound_files()
        self.currently_playing = False
        self.lock = threading.Lock()

    def _get_sound_files(self):
        # Get list of all files in the directory
        sound_files = [
            f for f in os.listdir(self.sound_directory)
            if os.path.isfile(os.path.join(self.sound_directory, f))
        ]
        # Filter for sound files (e.g., wav, mp3, ogg, flac)
        sound_files = [
            f for f in sound_files
            if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))
        ]
        return sound_files

    def play_random_sound(self):
        with self.lock:
            if self.currently_playing:
                print("A sound is already playing or in buffer time.")
                return
            else:
                self.currently_playing = True

        if not self.sound_files:
            print("No sound files found in the 'sound_clips' directory.")
            with self.lock:
                self.currently_playing = False
            return

        # Choose a random sound file
        sound_file = random.choice(self.sound_files)
        sound_path = os.path.join(self.sound_directory, sound_file)
        print(f"Playing sound: {sound_file}")
        # Load the sound file
        audio = AudioSegment.from_file(sound_path)

        # Play the audio in a separate thread
        threading.Thread(target=self._play_audio, args=(audio,), daemon=True).start()

    def _play_audio(self, audio):
        # Use _play_with_simpleaudio to get a PlayObject
        play_obj = _play_with_simpleaudio(audio)
        play_obj.wait_done()  # Wait until playback is finished
        # Wait for the buffer time
        buffer_time = 30  # Buffer time in seconds
        print(f"Buffering for {buffer_time} seconds before allowing next sound.")
        time.sleep(buffer_time)
        with self.lock:
            self.currently_playing = False
