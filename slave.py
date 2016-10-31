import winsound
import os, random

clip = random.choice(os.listdir("sound_clips")) #grab random clip
print clip
winsound.PlaySound('sound_clips/' + clip, winsound.SND_FILENAME)


