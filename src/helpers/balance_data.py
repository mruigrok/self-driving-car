import cv2
import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

def balance_data(file_name):
    try:
        #Load in the data corresponding to the file_name
        train_data = np.load(file_name, allow_pickle=True)
        df = pd.DataFrame(train_data)
        print(Counter(df[1].apply(str)))

        #Key combinations
        w = [1,0,0,0,0,0,0,0,0]
        s = [0,1,0,0,0,0,0,0,0]
        a = [0,0,1,0,0,0,0,0,0]
        d = [0,0,0,1,0,0,0,0,0]
        wa = [0,0,0,0,1,0,0,0,0]
        wd = [0,0,0,0,0,1,0,0,0]
        sa = [0,0,0,0,0,0,1,0,0]
        sd = [0,0,0,0,0,0,0,1,0]
        nk = [0,0,0,0,0,0,0,0,0]

        forwards = []
        lefts = []
        rights = []
        no_keys = []

        print('Training length: {}'.format(len(train_data)))

        for data in train_data:
            img = data[0]
            choice = data[1]
            
            #Group the forwards, lefts and rights together to remove bias of driving straight
            if (choice == w) or (choice == wa) or (choice == wd):
                forwards.append([img, choice])
            elif choice == a:
                lefts.append([img, choice])
            elif choice == d:
                rights.append([img, choice])
            elif choice == nk:
                no_keys.append([img, choice])

        print('Forwards: {}'.format(len(forwards)))
        print('Lefts: {}'.format(len(lefts)))
        print('Rights: {}'.format(len(rights)))
        print('No keys: {}'.format(len(no_keys)))

        min_length = min(len(forwards), len(lefts), len(rights), len(no_keys))
        forwards = forwards[:min_length]
        lefts = lefts[:min_length]
        rights = rights[:min_length]
        no_keys = no_keys[:int(min_length/4)]
        final_data = forwards + lefts + rights + no_keys

        #NN is not looking at the previous frame for training so shuffle the training
        #data so it doesn't bias the training data
        shuffle(final_data)

        balanced_data_filename = file_name.split('.npy')[0] + '_balanced.npy'
        np.save(balanced_data_filename, final_data)

    except Exception as e:
        print(str(e))


    '''
        min_length = min(len(forwards), len(forward_rights), len(forward_lefts), 
                    len(backwards), len(backwards_lefts), len(backwards_rights), len(lefts), len(rights))
        
        if not (min_length == 0):
            forwards = forwards[:min_length]
            backwards = backwards[:min_length]
            lefts = lefts[:min_length]
            rights = rights[:min_length]
            forward_lefts = forward_lefts[:min_length]
            forward_rights = forward_rights[:min_length]
            backwards_lefts = backwards_lefts[:min_length]
            backwards_rights = backwards_rights[:min_length]

            #Add all of the balanced data back 
            final_data = forwards + backwards + lefts + rights + forward_lefts + forward_rights + backwards_lefts + backwards_rights 
            shuffle(final_data)
            print(len(final_data))
            balanced_data_filename = file_name.split('.npy')[0] + '_balanced.npy'
            np.save(balanced_data_filename, final_data)
            
        else:
            raise ValueError
    '''