import cv2
import numpy as np
import random
import sys
import time
from helpers.grabscreen import grab_screen
from helpers.getkeys import key_check
from helpers.directkeys import PressKey, ReleaseKey, W, A, S, D
from models.alexnet import inception_v3 as googlenet

#Model type for formatting the saved model
type_model = 'googlenet-color'

WIDTH = 200
HEIGHT = 100
LR = 1e-3
EPOCHS = 5
MODEL_NAME = 'pygta5-car-{}-{}-{}-epochs.model'.format(LR, type_model, EPOCHS)

#screen region
region = (0,40,800,640)

#Possible key press combinations
key_map = {
    "W" : [1,0,0,0,0,0,0,0,0],
    "S" : [0,1,0,0,0,0,0,0,0],
    "A" : [0,0,1,0,0,0,0,0,0],
    "D" : [0,0,0,1,0,0,0,0,0],
    "WA" : [0,0,0,0,1,0,0,0,0],
    "WD" : [0,0,0,0,0,1,0,0,0],
    "SA" : [0,0,0,0,0,0,1,0,0],
    "SD" : [0,0,0,0,0,0,0,1,0],
    "NK" : [0,0,0,0,0,0,0,0,1]
}

#Key press functions
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

def countdown(seconds):
    for i in range(0,seconds)[::-1]:
        print(i+1)
        time.sleep(1)
        

def run(screen_region):
    print("Getting ready to test the model, loading model")
    time.sleep(1)
    #Load the model
    try:
        model = googlenet(WIDTH, HEIGHT, 3, LR, output=9, model_name=MODEL_NAME)
        model.load(MODEL_NAME)
    except Exception as e:
        print(str(e))
        print("Can't find the model, exiting........")
        exit()

    countdown(3)
    loop_time = time.time()
    done = False
    paused = False

    while not done:
        if not paused:
            screen = grab_screen(screen_region)
            #screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (HEIGHT,WIDTH))
            print('Frame took {} seconds'.format(time.time()-loop_time))
            loop_time = time.time()

            #Get the model prediction from the model
            prediction = model.predict([screen.reshape(WIDTH,HEIGHT,3)])[0]
            moves = list(np.around(prediction))
            print(moves, prediction)

            if moves == key_map["W"]:
                straight()
            elif moves == key_map["S"]:
                reverse()
            elif moves == key_map["A"]:
                left()
            elif moves == key_map["D"]:
                right()
            elif moves == key_map["WA"]:
                straight_left()
            elif moves == key_map["WD"]:
                straight_right()
            elif moves == key_map["SD"]:
                reverse_right()
            elif moves == key_map["SA"]:
                reverse_right()
            elif moves == key_map["NK"]:
                no_keys()

        #To pause and unpause the game use 'P'        
        keys = key_check()
        if 'P' in keys:
            if paused:
                paused = False
            else:
                paused = True
                no_keys()
                response = input("Would you like to exit? 'Y' to exit")
                if response == 'y' or response == 'Y':
                    done = True
                else:
                    print('Continue')   
            time.sleep(1)

if __name__ == "__main__":
    run(region)
