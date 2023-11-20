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
import matplotlib.pyplot as plt
from IPython import display

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
          self.game = Game()

     def get_state(self):
          """
               This is the function of the agent that will retrun the state of the snake at any moment. 
               :param None:
               :return State: What do we want this to be? the entire image? just the location of the snake and the location of the food?
          """
          return self.game.return_state()

     def cache(self, state, action, next_state, score, game_over):
          """
               append a state containing the parameters to the memeory deque
               :param state: the state of the game
               :param action: the action that is taken
               :param next_state: the state after the action is taken
               :param score: the score of the game after the action is taken
               :param game_over: is the game over (bool)
          """
          # print(f"cache in Agent.py: state: {state}")
          self.memory.append((state, action, next_state, score, game_over))


     def act(self, current_state):
          """
               This is function that will tell the game what to do based on the current state of the game.
               :param State: this is the state of the game that will be used to feed into the model
          """
          random_choice = random.random()
          # print(f"episolon {self.epsilon} \t rando {random_choice}")
          final_move_set = [1, 2, 3, 4] # this is what the snake will do 
          if random_choice > self.epsilon: # if the random value is greater than the episolon than the model can act. EXPLOITATION
               # print('exploit')
               state = T.tensor(current_state, dtype=T.float) # change the state into a tensor for the NN to use
               actions = self.model(state) # store the actions that the NN gives back
               choice = T.argmax(actions).item() # take the value that is the highest from the NN
               final_move = final_move_set[choice]
          else:
               # print('explore')
               final_move = random.randint(1, 4) # chose a random direction do move
               self.epsilon -= 0.001 * (1/(self.num_games+1)) # if we choose a random move, decrement the epsilon value
          
          return final_move # return the action 


     def train_long(self):
          if len(self.memory) > 1000: # if the length of the memory excedes 1,000 then we can pull a large batch of data from the memory to train from
               sample = random.sample(self.memory, 1000) # take a sample from memory
          else:
               sample = self.memory # if the length is less than 1,00 just  pull all of the memory

          state, action, next_state, score, game_over = zip(*sample) # agregate the data
          # print(f"train_long in Agent.py. state: {state}")
          self.trainer.train_step(state, action, next_state, score, game_over) # train on the data

     def train_short(self, state, action, next_state, score, game_over):
          self.trainer.train_step(state, action, next_state, score, game_over)

     
def Train(gamma:float=0.9, epsilon:float=0.5, learning_rate:float = 0.5):
     
     # define all the nodes that you need
     agent = Agent(gamma, epsilon, learning_rate)
     game = Game()

     running_score = []

     while True: # loop over this as long as you want to train
          state0 = agent.get_state() # get the state of the agent
          # print(f"Original state shape: {len(state0)}, {len(state0)} (should be 10x10)")
          # print(f"Converted state tensor: {state0.shape} (should be [100])")

          move0 = agent.act(state0) # get the action that the agent will take
          
          game_over, score = game.step(move0) # run a single step of the snake game and pull the score, game_over, and new state of the step
          state1 = game.return_state()
          # print(f"Train in Agent.py. state0: {state0}")

          agent.cache(state0, move0, state1, score, game_over) # store the data into the memory

          agent.train_long()
          # agent.train_short(state0, move0, state1, score, game_over)


          if game_over: # if the game is over
               game.new_game() # start a new game
               agent.num_games += 1
               agent.train_long()
               running_score.append(score)
               Plot_Results(running_score)


def Plot_Results(scores: list): # make a function to plot the scores of the game
     """
     This code was taken from:
     https://www.youtube.com/watch?v=wc-FxNENg9U
     """
     display.clear_output(wait=True)
     display.display(plt.gcf())
     plt.clf()
     plt.title('Training...')
     plt.xlabel('Number of Games')
     plt.ylabel('Score')
     plt.plot(scores)
     #     plt.plot(mean_scores)
     plt.ylim(ymin=-10)
     plt.text(len(scores)-1, scores[-1], str(scores[-1]))
     #     plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
     plt.show(block=False)
     plt.pause(.1)





Train(.3,0.9,1)