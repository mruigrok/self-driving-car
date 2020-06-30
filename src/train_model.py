import numpy as np
import pandas as pd
import time
import os
from random import shuffle
from models.alexnet import inception_v3 as googlenet

#The last data that was used to train
last_trained = 1
type_model = 'googlenet-color'
load_previous = True

WIDTH = 200
HEIGHT = 100
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'pygta5-car-{}-{}-{}-epochs.model'.format(LR, type_model, EPOCHS)

#Using the inception_v3 model (200x100 images with colour, reason for 3)
model = googlenet(WIDTH, HEIGHT, 3, LR, output=9, model_name=MODEL_NAME)

#Load the model
if load_previous:
    print("\nLoading previous model\n")
    try:
        model.load(MODEL_NAME)
    except:
        print("\nDidn't find previous model... training new one!\n")

#Find how many data sets there are to train
while True:
    data_end = last_trained
    file_name = 'training_data-{0}_balanced.npy'.format(data_end)
    if os.path.isfile(file_name):
        data_end += 1
    else:
        break

#Train the model
for epoch in range(EPOCHS):
    order = [num for num in range(last_trained, data_end + 1)]
    shuffle(order)

    for count, set_num in enumerate(order):
        file_name = 'training_data-{0}_balanced.npy'.format(set_num)
        try:
            train_data = np.load(file_name, allow_pickle=True)
            df = pd.DataFrame(train_data)
            train_data = df.values.tolist()
            print('Number of samples: {}'.format(len(train_data)))

            #Split into training and testing data, use 15% to validate
            pivot = int(len(train_data)*0.15)
            train = train_data[:-pivot]
            test = train_data[-pivot:]

            #Get the train and test data
            X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 3)
            Y = [i[1] for i in train]
            test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 3)
            test_y = [i[1] for i in test]

            #Fit the data to the model
            model.fit({'input': X}, {'targets': Y}, n_epoch=1,
                    validation_set=({'input':test_x},{'targets':test_y}),
                    snapshot_step=1000, show_metric=True, run_id=MODEL_NAME)

            #Save every 5 sets of data
            if count%5 == 0:
                model.save(MODEL_NAME)
                
        except Exception as e:
            print(e)
            print("Error with data set...going to the next one")
            time.sleep(1)

print("\n\n----------------- Training Completed ----------------- ")

'''
try:
    model.save(MODEL_NAME)
except Exception as e:
    print(e)
    print("Was not able to save the model....")
'''

#tensorboard --logdir=foo:C:\PythonScripts\gta5 project\log
