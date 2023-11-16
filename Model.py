"""
     This is the file that will hold the neural net model using pytorch
"""

####################
# TODO find out what the fuck is going on with this and how to make it work...
## this is in progress, I want to chat more with Hutt about it but I think I can get most of the way there
# TODO add more commenting and the preamble
####################

import torch
import torch.nn as nn # import the neural network package
import torch.optim as optim # import the optimizer function
import torch.nn.functional as F # import this helper function
import numpy as np


class linear_model(nn.Module):
     """
          This is the code to create the linear neural network model using pytorch.
          This code is largely taken from the pytorch website. 
          https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
     """

     def __init__(self, in_features: int, Hn: int, out_features: int):
          super().__init__() # call the parent init fucntion
          # TODO add the rest of the init of the model here

     def forward(self, x): 
          """
               This function will act as a helper to pass information forward to the further layers
               :param x: this is the data that will pass through the neural netword
          """
          pass