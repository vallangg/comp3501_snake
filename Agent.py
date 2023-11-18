"""
     This is the file for the agent that will learn
"""

import torch
from Model import Brain
from Snake import Game
from collections import deque # use this package to act as the memory

####################
# TODO: We need to figure out what exactly we need from the snake game to give to the agent
## I personally think a list of the locations of the snake segments as well as the location of the food, but looking online I have seen either feeding the whole frame into the agent
# i think this is pretty costly computation wise. I have also seen people give the snake a certain amount of vision. We should talk to Hutt about it
# TODO add more commenting and the preamble
####################
class Agent:
     """
          This is the agent that will be doing the actual learning. 
          The code for this is based on several different sources:
          https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
          https://www.youtube.com/watch?v=L8ypSXwyBds&t=4355s
     """

     def __init__(self):
          self.num_games = 0 # keep track ofthe number of games that has been run
          self.epsilon = 0 # measure of randomness in decision making
          self.gamma = 0 # learning rate
          self.memory = deque(maxlen=10000) # create the memory of the model with a maximum length of 10,000 so it does not get too big
          self.model =  Brain

     def get_state(self):
          """
               This is the function of the agent that will retrun the state of the snake at any moment. 
               :param None:
               :return State: What do we want this to be? the entire image? just the location of the snake and the location of the food?
          """
          return Game.get_state()

     def remember(self, state, action, next_state, score, game_over):
          """
               append a state containing the parameters to the memeory deque
               :param state: the state of the game
               :param action: the action that is taken
               :param next_state: the state after the action is taken
               :param score: the score of the game after the action is taken
               :param game_over: is the game over (bool)
          """
          self.memory.append(state, action, next_state, score, game_over) 


     def act(self, State):
          """
               This is function that will tell the game what to do based on the current state of the game.
               :param State: this is the state of the game that will be used to feed into the model
          """
          sel.fpytorch thing -> feed forward State.substate

     def train_long(self):
          pass

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






