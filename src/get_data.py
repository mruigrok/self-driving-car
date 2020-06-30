import cv2
import numpy as np
import time
import os
from helpers.grabscreen import grab_screen
from helpers.getkeys import key_check
from helpers.balance_data import balance_data

#Script to get the screen and keyboard input from the user. Use 'P' to pause collecting the data 

#Variables for image resize, filename for saving the data and screen region 
HEIGHT = 200
WIDTH = 100
#file_name = 'training_data_roi.npy'
region = (0,40,800,640)

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
    "NK" : [0,0,0,0,0,0,0,0,1]
}

def keys_to_output(keys):
    keys_pressed = ''.join(keys)
    if keys_pressed in key_map:
        return key_map[keys_pressed]
    else:
        return key_map['NK']

def countdown(seconds):
    for i in range(0,seconds)[::-1]:
        print(i+1)
        time.sleep(1)

def roi(img, vertices):
    #blank mask in the image region then filling pixels inside the vertices with the fill colour
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    
    #Return the image only where mask pixels are nonzero
    masked = cv2.bitwise_and(img, mask)
    return masked


def get_model_data(screen_region, start_val, roi=True):
    #Find where to start saving the data
    while True:
        file_name = 'training_data-{0}.npy'.format(start_val)
        if os.path.isfile(file_name):
            print('File exists')
            start_val += 1
        else:
            print('File does not exist')
            break
    
    countdown(3)
    loop_time = time.time()
    done = False
    #vertices = np.array([[10, 500], [10, 300], [300, 200], [500, 200], [800, 300], [800, 500]])
    training_data = []

    while not done:
        #Grab the screen in the given region, get the keys and 
        #add it to the training data and then update the time
        screen = grab_screen(region=screen_region)
<<<<<<< HEAD
<<<<<<< HEAD
    
        #Focus on the road and less on surroundings, then re-size, re-colour and save to data
        if roi == True:
            screen = screen[200:500, 100:700]
    
        screen = cv2.resize(screen, (HEIGHT,WIDTH))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        output_keys = keys_to_output(key_check())
=======
=======
>>>>>>> 19dcb836d3bfe2fea430c1e9be46d877818eb7de
        output_keys = keys_to_output(key_check())
        
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        #screen = roi(screen, [vertices])
        screen = cv2.resize(screen, (160,120))
<<<<<<< HEAD
>>>>>>> 19dcb836d3bfe2fea430c1e9be46d877818eb7de
=======
>>>>>>> 19dcb836d3bfe2fea430c1e9be46d877818eb7de
        training_data.append([screen, output_keys])
        
        print('Frame took {} seconds'.format(time.time()-loop_time))
        print(output_keys)
        
        loop_time = time.time()

        if len(training_data) % 500 == 0:
            print(len(training_data))
            file_name = 'training_data-{0}.npy'.format(start_val)
            np.save(file_name, training_data)
            start_val += 1
            training_data = []

        #Hit P to pause collecting data and start balancing phase
        keys = key_check()
        if 'P' in keys:
            response = input("Are you sure you want to stop getting data? (Enter Y to proceed): ")
            if response == "Y" or response == "y":
                done = True
            else:
                print("Continue collection...")
                countdown()

#Get model data and then balance
if __name__ == "__main__":
    start_val = 1
    print("Getting the Data for training!")
    time.sleep(1)
    get_model_data(region, start_val, roi=True)
    
    #Go through the data and balance data if not done
    while True:
        file_name = 'training_data-{0}.npy'.format(start_val)
        file_name_balanced = 'training_data-{0}_balanced.npy'.format(start_val)

        if os.path.isfile(file_name) and os.path.isfile(file_name_balanced):
            print('File already balanced')
        elif os.path.isfile(file_name) and not os.path.isfile(file_name_balanced):
            print(file_name)
            balance_data(file_name)
        else:
            break

        start_val += 1

    print("---------------- Done collecting and balancing ------------------")


