from captum.attr import (
            IntegratedGradients, LayerActivation, 
            LayerConductance, DeepLift, 
            Occlusion, NeuronConductance,LayerDeepLiftShap,LayerGradientShap,
            InputXGradient, ShapleyValueSampling)

import plotly.express as px
import torch
import numpy as np

from model import *
from seed import *

import time

start_time = time.time()

# defining datasets
X_train, X_test, y_train, y_test = gather_laws_DP()#seed_data(5)
n_features = X_train.shape[1]
out_features = 3#y_train.shape[1]

# defining number of examples
N = 100

# defining baselines for each input tensor
baseline = torch.zeros(N, n_features)

# defining and laoding existing model
model = LawsNetwork(input_features=n_features, output_features=out_features)
model.load_state_dict(torch.load("../model/laws.pth"), strict=True)
model.eval()

# defining sample input
indices = np.random.choice(np.arange(len(X_test)), N)
input = X_test[indices].reshape(N, n_features)#torch.rand(N, n_features)

# defining and applying integrated gradients on SeedModel
ig = IntegratedGradients(model)
attributions, approximation_error = ig.attribute(inputs=input,
                                                 baselines=baseline,
                                                 return_convergence_delta=True,
                                                 target=[0]*N,
                                                 n_steps=10)

# defining and applying Layer Conducatance to see the importance of neurons for a layer and given input.
# lc = LayerConductance(model, model.fc1)
# attributions, approximation_error = lc.attribute(inputs=input,
#                                                  baselines=baseline,
#                                                  return_convergence_delta=True,
#                                                  target=0)

# defining and applying Input * Gradient to see the importance of neurons for a layer and given input.
lgs = InputXGradient(model)
attributions = lgs.attribute(input, target=0)

# # defining and applying DeepLIFT on SeedModel
# dl = DeepLift(model)
# attributions, approximation_error = dl.attribute(inputs=input,
#                                                  baselines=baseline,
#                                                  return_convergence_delta=True,
#                                                  target=0)

if __name__ == "__main__":
    # print("Attributions\n", attributions.detach().numpy())
    end_time = time.time()
    
    print("Total time (sec)", end_time - start_time)
    # print("Error", np.linalg.norm(approximation_error.detach().numpy()))
    fig = px.imshow(attributions.detach().numpy(), 
                    labels=dict(x='Feature Inputs', y='Number of Inputs', color="Attributions"))
    # fig = px.imshow(attributions.detach().numpy(), 
    #                 labels=dict(x='Feature Inputs', y='Number of Inputs', color="Attributions"),
    #                 x=['Gender', 'Married', 'Dependents', 'Education', 'Self Employed', 'Applicant Income', 'Coapplicant Income',
    #                    'Loan Amount', 'Loan Amount Term', 'Credit History', 'Property Area'])
    
    fig.show()