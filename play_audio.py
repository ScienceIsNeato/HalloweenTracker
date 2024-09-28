import os
import random
from pydub import AudioSegment
from pydub.playback import play

def play_random_sound():
    sound_directory = 'sound_clips'
    # Get list of all files in the directory
    sound_files = [
        f for f in os.listdir(sound_directory)
        if os.path.isfile(os.path.join(sound_directory, f))
    ]
    # Filter for sound files (e.g., wav, mp3, ogg)
    sound_files = [
        f for f in sound_files
        if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))
    ]
    if not sound_files:
        print("No sound files found in the 'sound_clips' directory.")
        return
    # Choose a random sound file
    sound_file = random.choice(sound_files)
    sound_path = os.path.join(sound_directory, sound_file)
    print(f"Playing sound: {sound_file}")
    # Load and play the sound file
    audio = AudioSegment.from_file(sound_path)
    play(audio)

if __name__ == '__main__':
    play_random_sound()
