"""
     This is the file for the agent that will learn
"""

####################
# TODO: I we need to figure out what exactly we need from the snake game to give to the agent
## I personally think a list of the locations of the snake segments as well as the location of the food, but looking online I have seen either feeding the whole frame into the agent
# i think this is pretty costly computation wise. I have also seen people give the snake a certain amount of vision. We should talk to Hutt about it
# TODO add more commenting and the preamble
####################
class agent:
     """
          This is the agent that will be doing the actual learning. 
          The code for this is based on several different sources:
          https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
     """

     def __init__(self):
          self.num_games = 0 # keep track ofthe number of games that has been run

     def get_state(self):
          """
               This is the function of the agent that will retrun the state of the snake at any moment. 
               :param None:
               :return State: What do we want this to be? the entire image? just the location of the snake and the location of the food?
          """
          pass

     def remember(self):
          pass

     def act(self, State):
          """
               This is function that will tell the game what to do based on the current state of the game.
               :param State: this is the state of the game that will be used to feed into the model
          """

