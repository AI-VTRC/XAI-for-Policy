import torch
import torch.nn as nn
import torch.nn.functional as F

### List of models for attribution generation

class SeedNetwork(nn.Module):
    """3 layer ANN for prediction of minimum number among C numbers"""
    def __init__(self, C=5, fc1=16, fc2=8):
        """ C represents number of classes """
        super(SeedNetwork, self).__init__()
        self.fc1 = nn.Linear(C, fc1)
        self.dp1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(fc1, fc2)
        self.dp2 = nn.Dropout(p=0.3)
        self.output = nn.Linear(fc2, C)
    
    def forward(self, x):
        """ Forward pass with input x of shape N x C """
        #Layer 1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dp1(x)
        #Layer 2
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dp2(x)
        #Output Layer
        x = torch.sigmoid(self.output(x))
        return x
    
#main model for loan prediction (try out different values for hidden layers to improve baseline)
class LoanNetwork(nn.Module):
    def __init__(self, input_features, layer1=30, layer2=20, out_features=2):
        """Initialize the model for loan prediction"""
        super().__init__()
        self.fc1 = nn.Linear(input_features, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.out = nn.Linear(layer2, out_features)
        
    def forward(self, x):
        """Forward pass with 11 input features"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
    
class LawsNetwork(nn.Module):
    """ 4 layer ANN for Laws regression  """
    def __init__(self, input_features, output_features, fc1=32, fc2=16, fc3=8):
        """ C represents number of classes """
        super(LawsNetwork, self).__init__()
        self.fc1 = nn.Linear(input_features, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, fc3)
        self.output = nn.Linear(fc3, output_features)
    
    def forward(self, x):
        """ Forward pass with input x of shape N x C """
        #Layer 1
        x = self.fc1(x)
        x = F.relu(x)
        #Layer 2
        x = self.fc2(x)
        x = F.relu(x)
        #Layer 3
        x = self.fc3(x)
        x = F.relu(x)
        #Output Layer
        x = self.output(x)
        return x