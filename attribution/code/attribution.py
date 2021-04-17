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
X_train, X_test, y_train, y_test = gather_internet()#seed_data(5)
n_features = X_train.shape[1]

# defining number of examples
N = 100

# defining baselines for each input tensor
baseline = torch.zeros(N, n_features)

# defining and laoding existing model
model = LoanNetwork(input_features=n_features)
model.load_state_dict(torch.load("../model/internet_usage_model.pth"), strict=True)
model.eval()

# defining sample input
indices = np.random.choice(np.arange(len(X_test)), N)
input = X_test[indices].reshape(N, n_features)#torch.rand(N, n_features)

# defining and applying integrated gradients on SeedModel
# ig = IntegratedGradients(model)
# attributions, approximation_error = ig.attribute(inputs=input,
#                                                  baselines=baseline,
#                                                  return_convergence_delta=True,
#                                                  target=0,
#                                                  n_steps=50)

# defining and applying Layer Conducatance to see the importance of neurons for a layer and given input.
lc = LayerConductance(model, model.fc1)
attributions, approximation_error = lc.attribute(inputs=input,
                                                 baselines=baseline,
                                                 return_convergence_delta=True,
                                                 target=0)

# defining and applying Input * Gradient to see the importance of neurons for a layer and given input.
# lgs = InputXGradient(model)
# attributions = lgs.attribute(input,
#                                                  target=0)

# # defining and applying DeepLIFT on SeedModel
# dl = DeepLift(model)
# attributions, approximation_error = dl.attribute(inputs=input,
#                                                  baselines=baseline,
#                                                  return_convergence_delta=True,
#                                                  target=0)

# defining and applying Occlusion on SeedModel
ablator = Occlusion(model)
### TODO: fix error from sliding_window_shapes. !-- perturbation causes param mismatch --!
# attributions, approximation_error = ablator.attribute(inputs=input,
#                                                  sliding_window_shapes=(1,1),
#                                                  target=1)
column = ['Age', 'Community Building', 'Community Membership_Family',
'Community Membership_Hobbies', 'Community Membership_None',
'Community Membership_Other', 'Community Membership_Political',
'Community Membership_Professional',
'Community Membership_Religious', 'Community Membership_Support',
'Country\tDisability_Cognitive', 'Disability_Hearing',
'Disability_Motor', 'Disability_Not Impaired',
'Disability_Not Say', 'Disability_Vision', 'Education Attainment',
'Falsification of Information', 'Gender', 'Household Income',
'How You Heard About Survey_Banner',
'How You Heard About Survey_Friend',
'How You Heard About Survey_Mailing List',
'How You Heard About Survey_Others',
'How You Heard About Survey_Printed Media',
'How You Heard About Survey_Remebered',
'How You Heard About Survey_Search Engine',
'How You Heard About Survey_Usenet News',
'How You Heard About Survey_WWW Page',
'Major Geographical Location', 'Major Occupation',
'Marital Status', 'Most Import Issue Facing the Internet',
'Opinions on Censorship', 'Primary Computing Platform',
'Primary Language', 'Primary Place of WWW Access', 'Race',
'Not Purchasing_Bad experience', 'Not Purchasing_Bad press',
'Not Purchasing_Cant find', 'Not Purchasing_Company policy',
'Not Purchasing_Easier locally', 'Not Purchasing_Enough info',
'Not Purchasing_Judge quality', 'Not Purchasing_Never tried',
'Not Purchasing_No credit', 'Not Purchasing_Not applicable',
'Not Purchasing_Not option', 'Not Purchasing_Other',
'Not Purchasing_Prefer people', 'Not Purchasing_Privacy',
'Not Purchasing_Receipt', 'Not Purchasing_Security',
'Not Purchasing_Too complicated', 'Not Purchasing_Uncomfortable',
'Not Purchasing_Unfamiliar vendor', 'Registered to Vote',
'Sexual Preference', 'Web Ordering', 'Web Page Creation',
"Who Pays for Access_Don't Know", 'Who Pays for Access_Other',
'Who Pays for Access_Parents', 'Who Pays for Access_School',
'Who Pays for Access_Self', 'Who Pays for Access_Work',
'Willingness to Pay Fees', 'Years on Internet']

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