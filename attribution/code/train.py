import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import seed
from model import *

#generate seed data
X_train, X_test, y_train, y_test = seed.gather_laws()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Enable cuda if available

# seed the model for reproducibility (ideally across all nodes, later...)
torch.manual_seed(0)

dimensions = X_train.shape[1], y_train.shape[1]
model = LawsNetwork(*dimensions).to(device)

# implement backprop
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

def train(epochs=int(5e3), epsilon=0.1):
    """
    Train the model. Assumes access to global variables X_train, X_test, y_train, y_test, loss function & optim.
    """
    start_time = time.time()
    losses = []
    
    for i in tqdm(range(epochs)):
        y_pred = model(X_train)
        loss = loss_function(y_pred, y_train)
        losses.append(loss)
        
        if loss.item() < epsilon:
            print(f"Model Converged at epoch {i + 1}, loss = {loss.item()}")
            break
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Total training time (sec): {time.time() - start_time}, loss - {loss.item()}")
    
    return losses

def save_model(PATH):
    torch.save(model.state_dict(), "../model/" + PATH)
    
if __name__ == '__main__':
    cost = train()
    #graph cost
    plt.plot(cost)
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.title("Cost over epoch")
    plt.grid()
    plt.show()
    
    save_model("laws.pth")