"""
     Stuff
"""

import torch as T
import torch.nn as nn # import the neural network package
import torch.optim as optim # import the optimizer function
import torch.nn.functional as F # import this helper function
import numpy as np
from Snake import Game

class trainer:
     """
     This does the training and running of the game
     The code for this is based on several different sources:
     https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
     https://www.youtube.com/watch?v=L8ypSXwyBds&t=4355s
     """

     def __init__(self, model, learning_rate, gamma):
          self.lr = learning_rate
          self.gamma = gamma
          self.model = model
          self.optimizer = optim.Adam(self.model.parameters(), self.lr)# this is the optimizer function
          self.loss_function = nn.MSELoss() # define the loss fucntion
 
     def train_step(self, state, action, next_state, score, game_over):
          # print(f"State in Trainer: {state}")
          state = T.tensor(state, dtype=T.float)
          action = T.tensor(action, dtype=T.float)
          next_state = T.tensor(next_state, dtype=T.float)
          score = T.tensor(score, dtype=T.float)

          if len(state.shape) == 1: # if there is only one value in the tensor
               state = T.unsqueeze(state, 0) # this appends one dimension to the tensor shape
               action = T.unsqueeze(action, 0)
               next_state = T.unsqueeze(next_state, 0) 
               score = T.unsqueeze(score, 0) 
               game_over = (game_over, ) # put the bool into a tuple with only itself
          
          pred = self.model(state) # generate the predicted Q values from the state

          target = pred.clone() 

          for ii_pred in range(len(game_over)):
               # print("Trainer in Trainer.py. target: ", target[ii_pred])
               Q_new = score[ii_pred]
               if not game_over[ii_pred]:
                    Q_new = score[ii_pred] + self.gamma * T.max(self.model(next_state[ii_pred])) # compute the Q value
               
               target[ii_pred][T.argmax(action).item()] = Q_new # update the Q table using the new value of Q that was just caluclated
          
          self.optimizer.zero_grad() # empty the gradient 
          loss = self.loss_function(target, pred) # comapre the new Q and old one
          loss.backward() 
          
          self.optimizer.step()

