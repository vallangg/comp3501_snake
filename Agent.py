"""
     This is the file for the agent that will learn
"""

import torch as T
from Model import Brain
from Snake import Game
from collections import deque # use this package to act as the memory
import numpy as np
import random
from Trainer import trainer

####################
# TODO: We need to figure out what exactly we need from the snake game to give to the agent
## I personally think a list of the locations of the snake segments as well as the location of the food, but looking online I have seen either feeding the whole frame into the agent
# i think this is pretty costly computation wise. I have also seen people give the snake a certain amount of vision. We should talk to Hutt about it
# TODO add more commenting and the preamble
####################\


class Agent:
     """
          This is the agent that will be doing the actual learning. 
          The code for this is based on several different sources:
          https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
          https://www.youtube.com/watch?v=L8ypSXwyBds&t=4355s
          https://www.youtube.com/watch?v=wc-FxNENg9U
     """

     def __init__(self, gamma, epislon, learning_rate):
          self.num_games = 0 # keep track ofthe number of games that has been run
          self.epsilon = epislon # measure of randomness in decision making
          self.gamma = gamma # weighting for rewards
          self.learning_rate = learning_rate # how fast will the agent learn
          self.memory = deque(maxlen=100000) # create the memory of the model with a maximum length of 100,000 so it does not get too big
          self.model =  Brain()
          self.trainer = trainer(self.model, self.learning_rate, self.gamma)

     def get_state(self):
          """
               This is the function of the agent that will retrun the state of the snake at any moment. 
               :param None:
               :return State: What do we want this to be? the entire image? just the location of the snake and the location of the food?
          """
          return Game.get_state()

     def cache(self, state, action, next_state, score, game_over):
          """
               append a state containing the parameters to the memeory deque
               :param state: the state of the game
               :param action: the action that is taken
               :param next_state: the state after the action is taken
               :param score: the score of the game after the action is taken
               :param game_over: is the game over (bool)
          """
          self.memory.append(state, action, next_state, score, game_over) 


     def act(self, current_state):
          """
               This is function that will tell the game what to do based on the current state of the game.
               :param State: this is the state of the game that will be used to feed into the model
          """
          final_move = [0,0,0,0] # this is what the snake will do 
          if np.random.random() > self.epsilon: # if the random value is greater than the episolon than the model can act. EXPLOITATION
               state = T.tensor(current_state, dtype=T.float) # change the state into a tensor for the NN to use
               actions = self.model(state) # store the actions that the NN gives back
               action = T.argmax(actions).item() # take the value that is the highest from the NN
               final_move[action] = 1
          else:
               action = np.random.choice(self.action) # choose a random action from the action space. EXPLORATION
               final_move[action] = 1
               self.epsilon -= 1/self.num_games # if we choose a random move, decrement the epsilon value
          
          return final_move # return the action 


     def train_long(self):
          if len(self.memory) > 1000: # if the length of the memory excedes 1,000 then we can pull a large batch of data from the memory to train from
               mini_sample = random.sample(self.memory, 100_000) # take a sample from memory
          else:
               mini_sample = self.memory # if the length is less than 1,00 just  pull all of the memory

          state, action, next_state, score, game_over = zip(*mini_sample) # agregate the data

          self.trainer.train_step(state, action, next_state, score, game_over) # train on the data

     def train_short(self):
          pass

     
def Trainer():
     """
          This does the training and running of the game
          The code for this is based on several different sources:
          https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
          https://www.youtube.com/watch?v=L8ypSXwyBds&t=4355s
     """

     game = Game() # bring in the snake game
     agent = Agent() # create the agent

     running_scores = [] # create a list that will contain all of the scores the model acheives
     highest_score = 0 # store the highest score the model has acheived






