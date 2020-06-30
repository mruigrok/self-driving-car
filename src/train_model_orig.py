import pandas as pd
import numpy as np
import os
from models.alexnet import alexnet_2

WIDTH = 160
HEIGHT = 80
LR = 1e-3
model = alexnet_2(WIDTH, HEIGHT, LR, output=9)

def train_model(file_name, model_name, load_previous=True):
    #Find the data and model
    if os.path.isfile(file_name):
        print('Located training data')
        train_data = np.load(file_name, allow_pickle=True)
        if load_previous:
            try:
                model.load(model_name)
            except:
                print('Cannot find existing model...exiting')
                return
        else:
            print('Creating new model...')
    else:
        print('Cannot find file...exiting')
        return

    #Data and model are loaded..start training
    df = pd.DataFrame(train_data)
    train_data = df.values.tolist()

    #Split into training and testing data, use 15% to validate
    pivot = int(len(train_data)*0.15)
    train = train_data[:-pivot]
    test = train_data[-pivot:]

    #Get the train and test data
    X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
    Y = [i[1] for i in train]
    test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
    test_y = [i[1] for i in test]

    #Fit the data to the model
    model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS,
            validation_set=({'input':test_x},{'targets':test_y}),
            snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    #tensorboard --logdir=foo:C:\PythonScripts\gta5 project\log
    model.save(MODEL_NAME)
    return