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
          super().__init__() # call the parent init fucntion
          # TODO add the rest of the init of the model here
          self.flatten = nn.Flatten()
          self.linear_relu_stack = nn.Sequential(
               nn.Linear(10*10, 4),   # 10X10 grid with hidden layers
               nn.ReLU(),
               nn.Linear(4, 4),   #Hidden layers
               nn.ReLU(),
               nn.Linear(4, 4),  #Hidden Layers and 4 directions to move in
          )

     def forward(self, x): 
          """
               This function will act as a helper to pass information forward to the further layers
               :param x: this is the data that will pass through the neural netword
          """
          x = self.flatten(x)
          print(x)
          logits = self.linear_relu_stack(x)
          return logits
     


b = Brain()

game = Game()
stat = game.get_state()
ten_stat = torch.tensor(stat, device=device).float()
# print(ten_stat)
# X = torch.rand(1, 28, 28, device=device)
logits = b(ten_stat)
print("Logits: ",logits)

pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# data = [[int(i) for i in range(0,10)] for i in range(0,10)]
# tensor = torch.tensor(data)
# t = b.forward(tensor)
# print(t)
# untimeError: mat1 and mat2 must have the same dtype, but got Long and Float