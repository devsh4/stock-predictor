# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 18:52:04 2018

@author: Dev
"""
import numpy as np

def calc_directional_accuracy(test, pred, pred_length):
    
    test = np.asarray(test)

    pred = np.asarray(pred)
    pred = np.concatenate(pred)

    test_new=[] 
    pred_new=[]
    
    temp = pred_length - 1
    for x in range(0, 500, pred_length):
        test_new.append(test[x])
        test_new.append(test[x+temp])

    for x in range(0, 500, pred_length):
        pred_new.append(pred[x])
        pred_new.append(pred[x+temp])

    pred_new = np.array(pred_new)
    test_new = np.array(test_new)

    test_direction = []
    pred_direction = []

    for x in range(0, len(test_new), 2):
        if(test_new[x] > test_new[x+1]):
            test_direction.append('down')
        else:
            test_direction.append('up')

    print('>>>>>>>>')

    for x in range(0, len(pred_new), 2):
        if(pred_new[x] > pred_new[x+1]):
            pred_direction.append('down')
        else:
            pred_direction.append('up')


    print("True:", test_direction)
    print("Length", len(test_direction))
    print("Predicted:", pred_direction)
    print("Length", len(pred_direction))

    results = []
    for x in range(len(pred_direction)):
        if(pred_direction[x] == test_direction[x]):
            results.append(1)
        else:
            results.append(0)

    results = np.array(results)
    count = 0

    for x in range(len(results)):
        if(results[x] == 1):
            count = count+1
        
    da = count/(len(results))
    print('Directional Accuracy:',da)