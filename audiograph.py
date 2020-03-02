import pyaudio
import struct
import numpy as np
import wave
import audioop
import time
import PIL
import cv2
from random import randint
import threading 
import queue
Chunk = 1024
q = queue.Queue()

def imageGenerator():
    rndI = randint(0, 19)
    cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("image",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    while True:
        if q.empty() == True:
            rndJ = randint(0, 19)
            cvimage = cv2.imread('images/new%d_morphed%d' % (rndI, rndJ))
            cv2.imshow("image", cvimage)
            cv2.waitKey(10)
        else:
            rndI = randint(0, 19)
            print("changed!")
            q.get()


p = pyaudio.PyAudio()
p.get_default_input_device_info()
wf = wave.open('audiofiles/inst16b.wav','rb')

stream = p.open(
    format = p.get_format_from_width(wf.getsampwidth()),
    channels = wf.getnchannels(),
    rate = wf.getframerate(),
    output = True
    )

start = time.time()
startAudio = time.time()
audioData = wf.readframes(Chunk)
rndI = randint(0, 19)
rndJ = randint(0, 19)

x = threading.Thread(target=imageGenerator)
x.start()

while audioData != '':    
    stream.write(audioData)
    audioData = wf.readframes(Chunk)
    #print (audioData)
    amplitude = audioop.rms(audioData, 2)

    if (amplitude > 15000) & (time.time() - start > 0.3):
        start = time.time()
        #put to queue
        q.put(1)
        print("bang")