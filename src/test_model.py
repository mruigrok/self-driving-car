import cv2
import numpy as np
import random
import time
from helpers.grabscreen import grab_screen
from helpers.getkeys import key_check
from helpers.directkeys import PressKey, ReleaseKey, W, A, S, D
from models.alexnet import alexnet_2

#Model type for formatting the saved model
type_model = 'alexnetv2'

WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 5
MODEL_NAME = 'pygta5-car-{}-{}-{}-epochs.model'.format(LR, type_model, EPOCHS)

#Possible key press combinations
w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,0]

#Key presses
def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)

def left():
    PressKey(A)
    PressKey(W)
    ReleaseKey(D)
    ReleaseKey(S)
    time.sleep(1)

def right():
    PressKey(D)
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    time.sleep(1)

def reverse():
    PressKey(S)
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def straight_left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
    time.sleep(1)

def straight_right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    time.sleep(1)

def reverse_left():
    PressKey(S)
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)

def reverse_right():
    PressKey(S)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(W)
    
def no_keys():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)

def countdown():
    for i in range(0,3,-1):
        print(i+1)
        time.sleep(1)
        
#Load the model
model = alexnet_2(WIDTH, HEIGHT, LR, output=9)
model.load(MODEL_NAME)

def run():
    countdown()
    last_time = time.time()
    paused = False

    while True:
        if not paused:
            screen = grab_screen(region=(0,40,800,640))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (HEIGHT,WIDTH))
            print('Frame took {} seconds'.format(time.time()-last_time))
            last_time = time.time()

            #Get the model prediction from the model
            prediction = model.predict([screen.reshape(WIDTH,HEIGHT,1)])[0]
            moves = list(np.around(prediction))
            print(moves, prediction)

            if moves == w:
                straight()
            elif moves == s:
                reverse()
            elif moves == a:
                left()
            elif moves == d:
                right()
            elif moves == wa:
                straight_left()
            elif moves == wd:
                straight_right()
            elif moves == sd:
                reverse_right()
            elif moves == sa:
                reverse_right()
            elif moves == nk:
                no_keys()

        #To pause and unpause the game use 'Z'        
        keys = key_check()
        if 'Z' in keys:
            if paused:
                paused = False
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)     
                
            time.sleep(1)

if __name__ == "__main__":
    run()
