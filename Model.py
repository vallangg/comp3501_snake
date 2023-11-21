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
from Snake import Game

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class Brain(nn.Module):
     """
          This is the code to create the linear neural network model using pytorch.
          This code is largely taken from the pytorch website. 
          https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
     """

     def __init__(self):
          input_size = 100
          hidden_size = 512
          output_size = 4
          super().__init__() # call the parent init fucntion
          self.flatten = nn.Flatten()

          self.linear_relu_stack = nn.Sequential(
               nn.Linear(input_size, hidden_size),   # 10X10 grid with hidden layers
               nn.ReLU(),
               nn.Linear(hidden_size, hidden_size),   #Hidden layer
               nn.ReLU(),
               nn.Linear(hidden_size, hidden_size), # second hidden layer
               nn.ReLU(), 
               nn.Linear(hidden_size, output_size),  #Hidden Layers and 4 directions to move in
          )

     def forward(self, x): 
          """
               This function will act as a helper to pass information forward to the further layers
               :param x: this is the data that will pass through the neural netword
          """
          # print(f"Original x shape: {x.shape}")  # Debugging: Check the original shape of x
          # x = self.flatten(x)
          # print(f"Flattened x shape: {x.shape}")  # Debugging: Check the shape after flattening
          logits = self.linear_relu_stack(x)
          return logits
     


     


# b = Brain()

# game = Game()

# # Testing the code to ensure the output is what we want.
# stat = game.return_state()
# ten_stat = torch.tensor(stat, device=device).float()
# # ten_stat = ten_stat.view(-1, 100) 
# # print(ten_stat)
# # X = torch.rand(1, 28, 28, device=device)
# logits = b(ten_stat)
# print("Logits: ",logits)



# pred_probab = nn.Softmax(dim=1)(logits)
# y_pred = pred_probab.argmax(1)
# print(f"Predicted class: {y_pred}")
