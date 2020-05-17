import cv2
import numpy as np
import time
import os
from helpers.grabscreen import grab_screen
from helpers.getkeys import key_check
from helpers.balance_data import balance_data

#Script to get the screen and keyboard input from the user. Use 'P' to pause collecting the data 

#These are the key press combinations we care about
key_map = {
    "W" : [1,0,0,0,0,0,0,0,0],
    "S" : [0,1,0,0,0,0,0,0,0],
    "A" : [0,0,1,0,0,0,0,0,0],
    "D" : [0,0,0,1,0,0,0,0,0],
    "WA" : [0,0,0,0,1,0,0,0,0],
    "WD" : [0,0,0,0,0,1,0,0,0],
    "SA" : [0,0,0,0,0,0,1,0,0],
    "SD" : [0,0,0,0,0,0,0,1,0],
    "NK" : [0,0,0,0,0,0,0,0,0]
}

def keys_to_output(keys):
    keys_pressed = ''.join(keys)
    if keys_pressed in key_map:
        return key_map[keys_pressed]
    else:
        return key_map['NK']

def countdown():
    for i in range(0,3,-1):
        print(i+1)
        time.sleep(1)

def roi(img, vertices):
    #blank mask in the image region then filling pixels inside the vertices with the fill colour
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    
    #Return the image only where mask pixels are nonzero
    masked = cv2.bitwise_and(img, mask)
    return masked

def get_model_data(file_name, screen_region):
    if os.path.isfile(file_name):
        print('File exists, loading previous data!')
        training_data = list(np.load(file_name, allow_pickle=True))
    else:
        print('Does not exits, creating new')
        training_data = []

    countdown()
    last_time = time.time()
    done = False
    vertices = np.array([[10, 500], [10, 300], [300, 200], [500, 200], [800, 300], [800, 500]])

    while not done:
        #Grab the screen in the given region, get the keys and 
        #add it to the training data and then update the time
        screen = grab_screen(region=screen_region)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        #screen = roi(screen, [vertices])
        screen = cv2.resize(screen, (160,120))
        output_keys = keys_to_output(key_check())
        training_data.append([screen, output_keys])
        
        print('Frame took {} seconds'.format(time.time()-last_time))
        print(output_keys)
        
        last_time = time.time()

        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.save(file_name, training_data)

        #Hit P to pause collecting data and start balancing phase
        keys = key_check()
        if 'P' in keys:
            response = input("Are you sure you want to stop getting data? (Enter Y to proceed): ")
            if response == "Y" or response == "y":
                done = True
            else:
                print("Continue collection...")
                countdown()


#filename for saving the data and screen region
file_name = 'training_data2.npy'
region = (0,40,800,640)


#Get model and then balance the data before training
if __name__ == "__main__":
    print("Getting the Data for training!")
    countdown()
    get_model_data(file_name, region)
    balance_data(file_name)

