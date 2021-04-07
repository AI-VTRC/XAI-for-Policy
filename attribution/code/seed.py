import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import torch

def generate_data(num_input=2):
    """Define function to compute the minimum class label of C 16-bit integers. Return the corresponding oracle"""
    #define params
    min_value = np.iinfo(np.int16).min
    max_value = np.iinfo(np.int16).max
    C = num_input #number of input variable
    
    data = np.random.randint(low=min_value, high=max_value, size=(int(1e5), C))
    oracle = []
    for num in data:
        val = [0]*C
        val[np.argmax(num)] = 1
        oracle.append(val)
    
    return data, np.array(oracle).reshape(len(data), C)

def seed_data(num_input):
    X, y = generate_data(num_input) #gather data
    #split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    ### Create tensors from np.ndarry main data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Enable cuda if available

    X_train = torch.FloatTensor(X_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    y_test = torch.LongTensor(y_test).to(device)
    
    return X_train, X_test, y_train, y_test

def gather_loans():
    """ Load tensors for loans dataset """
    X_train, y_train = torch.load("../data/loans/tensor_train.pt")
    X_test, y_test = torch.load("../data/loans/tensor_test.pt")
    return X_train, X_test, y_train, y_test

def gather_internet():
    """ Load tensors for internet usage (UCI) """
    X_train, y_train = torch.load("../data/internet_usage/tensor_train.pt")
    X_test, y_test = torch.load("../data/internet_usage/tensor_test.pt")
    return X_train, X_test, y_train, y_test