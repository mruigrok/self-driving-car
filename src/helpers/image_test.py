import pandas as pd
import numpy as np
import cv2

filename = 'training_data3.npy'

train_data = np.load(filename, allow_pickle = True)
images = []
for i in range(len(train_data)):
    screen = train_data[i][0]
    images.append(screen)
    
    
cv2.imshow('window', screen)
cv2.waitKey(0)
cv2.destroyAllWindows()

A = [1,2,3]
print(A[:5])


