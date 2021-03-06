import pandas as pd
import numpy as np
import cv2

filename = '../training_data-2_balanced.npy'
train_data = np.load(filename, allow_pickle = True)
images = [train_data[i][0] for i in range(len(train_data))]

img = images[0]

cv2.imshow('window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
