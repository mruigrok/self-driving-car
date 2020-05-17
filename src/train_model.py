import numpy as np
import pandas as pd
import os
from models.alexnet import alexnet_2

WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 5
MODEL_NAME = 'pygta5-car-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2', EPOCHS)
file_name = 'training_data2_balanced.npy'
load_previous = True

model = alexnet_2(WIDTH, HEIGHT, LR, output=9)
if load_previous:
    print("Loading previous model")
    try:
        model.load(MODEL_NAME)
    except:
        print("Didn't find previous model... training new one!")

train_data = np.load(file_name, allow_pickle=True)
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
