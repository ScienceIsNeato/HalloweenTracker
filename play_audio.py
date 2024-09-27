import winsound
import os
import random

clip = random.choice(os.listdir("sound_clips"))  # Grab a random clip from the directory
print(clip)
winsound.PlaySound('sound_clips/' + clip, winsound.SND_FILENAME)
