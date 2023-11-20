"""
    This code was taken from the github linked here: https://gist.github.com/StanislavPetrovV/bb36787efbc30cd0921f0cbaa05244f1
    Author: StanislavPetrovV
    All edits that have been made by this team are commented as such
"""

####################
# TODO add the preamble to this and make it better. 
## I dont think that we need to do much commenting on the code that isn't ours
# TODO figure out what is needed for the agent and find a way to implement it
## I think this requires a better understanding of what exactly the agent needs and then doing some experimenting. obviosuly it would need the score for training purposes but
# i don't know what
####################



import pygame as pg
import sys
from random import randrange
import numpy as np

vec2 = pg.math.Vector2


class Snake:
    def __init__(self, game):
        self.game = game
        self.size = game.TILE_SIZE
        self.rect = pg.rect.Rect([0, 0, game.TILE_SIZE - 2, game.TILE_SIZE - 2])
        self.range = self.size // 2, self.game.WINDOW_SIZE - self.size // 2, self.size
        self.rect.center = self.get_random_position()
        self.direction = vec2(0, 0)
        self.step_delay = 100  # milliseconds
        self.time = 0
        self.length = 1
        self.segments = []
        self.directions = {pg.K_w: 1, pg.K_s: 1, pg.K_a: 1, pg.K_d: 1}
        self.score = 0 # CUSTOM CODE. keep track of the score of the snake
        self.game_done = False # CUSTOM CODE. if the game is done this will be true
        
    def control(self, event):
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_w and self.directions[pg.K_w]:
                self.direction = vec2(0, -self.size)
                self.directions = {pg.K_w: 1, pg.K_s: 0, pg.K_a: 1, pg.K_d: 1}

            if event.key == pg.K_s and self.directions[pg.K_s]:
                self.direction = vec2(0, self.size)
                self.directions = {pg.K_w: 0, pg.K_s: 1, pg.K_a: 1, pg.K_d: 1}

            if event.key == pg.K_a and self.directions[pg.K_a]:
                self.direction = vec2(-self.size, 0)
                self.directions = {pg.K_w: 1, pg.K_s: 1, pg.K_a: 1, pg.K_d: 0}

            if event.key == pg.K_d and self.directions[pg.K_d]:
                self.direction = vec2(self.size, 0)
                self.directions = {pg.K_w: 1, pg.K_s: 1, pg.K_a: 0, pg.K_d: 1}

    def neural_control(self, neural_direction):
            if neural_direction == 1:
                self.direction = vec2(0, -self.size)
                self.directions = {pg.K_w: 1, pg.K_s: 0, pg.K_a: 1, pg.K_d: 1}

            if neural_direction == 2:
                self.direction = vec2(0, self.size)
                self.directions = {pg.K_w: 0, pg.K_s: 1, pg.K_a: 1, pg.K_d: 1}

            if neural_direction == 3:
                self.direction = vec2(-self.size, 0)
                self.directions = {pg.K_w: 1, pg.K_s: 1, pg.K_a: 1, pg.K_d: 0}

            if neural_direction == 4:
                self.direction = vec2(self.size, 0)
                self.directions = {pg.K_w: 1, pg.K_s: 1, pg.K_a: 0, pg.K_d: 1}

    def delta_time(self):
        time_now = pg.time.get_ticks()
        if time_now - self.time > self.step_delay:
            self.time = time_now
            return True
        return False

    def get_random_position(self):
        return [randrange(*self.range), randrange(*self.range)]

    def check_borders(self):
        if self.rect.left < 0 or self.rect.right > self.game.WINDOW_SIZE:
            # print("hit a border")
            self.game_done = self.game.game_over()
        if self.rect.top < 0 or self.rect.bottom > self.game.WINDOW_SIZE:
            # print("hit a border")
            self.game_done = self.game.game_over()

    def check_food(self):
        if self.rect.center == self.game.food.rect.center:
            self.game.food.rect.center = self.get_random_position()
            self.length += 1
            self.score += 20 # CUSTOM CODE. if the snake eats the food add 10 points to the score
            # print("FOOD WAS EATEN")

    def check_selfeating(self):
        if len(self.segments) != len(set(segment.center for segment in self.segments)):
            # print("hit itself")
            self.game_done = self.game.game_over()

    def move(self):
        if self.delta_time():
            self.rect.move_ip(self.direction)
            self.segments.append(self.rect.copy())
            self.segments = self.segments[-self.length:]

    def update(self):
        self.check_selfeating()
        self.check_borders()
        self.check_food()
        self.move()

    def draw(self):
        [pg.draw.rect(self.game.screen, 'green', segment) for segment in self.segments]
    
    def return_score(self)->int:
        return self.score # CUSTOM CODE. return the score of the current game of snake


class Food:
    def __init__(self, game):
        self.game = game
        self.size = game.TILE_SIZE
        self.rect = pg.rect.Rect([0, 0, game.TILE_SIZE - 2, game.TILE_SIZE - 2])
        self.rect.center = self.game.snake.get_random_position()

    def draw(self):
        pg.draw.rect(self.game.screen, 'red', self.rect)


class Game:
    def __init__(self):
        pg.init()
        self.WINDOW_SIZE = 500
        self.TILE_SIZE = 50
        self.screen = pg.display.set_mode([self.WINDOW_SIZE] * 2)
        self.clock = pg.time.Clock()
        self.new_game()

    def draw_grid(self):
        [pg.draw.line(self.screen, [50] * 3, (x, 0), (x, self.WINDOW_SIZE))
                                             for x in range(0, self.WINDOW_SIZE, self.TILE_SIZE)]
        [pg.draw.line(self.screen, [50] * 3, (0, y), (self.WINDOW_SIZE, y))
                                             for y in range(0, self.WINDOW_SIZE, self.TILE_SIZE)]

    def new_game(self):
        # print("GAME ENDED")
        pg.init()
        self.snake = Snake(self)
        self.food = Food(self)
        # self.snake.game_done = False
    
    def game_over(self): # CUTSOM CODE. if the gameww is over return the final score
        self.snake.score -= 10 # decrement the code if the game ends
        # print(self.snake.score)
        return True

    def update(self):
        self.snake.update()
        pg.display.flip()
        self.clock.tick(60)

    def draw(self):
        self.screen.fill('black')
        self.draw_grid()
        self.food.draw()
        self.snake.draw()

    def check_event(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            # snake control
            self.snake.control(event)
        

    def step(self, neural_direction):
        """
            CUSTOM CODE
            This code will run a single step in the game of snake based on the Neural Networks input
        """
        self.snake.neural_control(neural_direction)
        self.check_event()
        self.update()
        self.draw()

        # print(f"direction chosen: {neural_direction}")

        return  self.snake.game_done, self.snake.score

    def run(self):
        while not self.snake.game_done:
            self.check_event()
            self.update()
            self.draw()
        else:
            self.new_game()
    
    def return_state(self)->list:
        """
            This code will return a list to represent the board
            0=nothing, 1=snake, 2=head, 3=food
            :param None:
            :return list: the array that represents the game board
        """

        grid_size = 10  # for a 10 square grid
        return_list = [[0 for ii in range(grid_size)] for jj in range(grid_size)]  # Corrected list initialization

        # print(self.snake.segments)

        for segment in self.snake.segments:
            x_index = int(segment.x / self.TILE_SIZE)
            y_index = int(segment.y / self.TILE_SIZE)
            # print(f"Segment position: ({segment.x}, {segment.y}), Grid index: ({x_index}, {y_index})")
            if 0 <= x_index < grid_size and 0 <= y_index < grid_size:
                return_list[y_index][x_index] = 1  # Place a 1 where there is a snake segment
        
        if self.snake.segments:
            head_segment = self.snake.segments[0]
            head_x = int(head_segment.x / self.TILE_SIZE)
            head_y = int(head_segment.y / self.TILE_SIZE)
            if 0 <= head_x < grid_size and 0 <= head_y < grid_size:
                return_list[head_y][head_x] = 2  # Mark as snake head
        

        food_x_index = int(self.food.rect.centerx / self.TILE_SIZE)
        food_y_index = int(self.food.rect.centery / self.TILE_SIZE)
        if 0 <= food_x_index < grid_size and 0 <= food_y_index < grid_size:
            return_list[food_y_index][food_x_index] = 3  # Place a 2 where there is food
        

        # print(f"return_state in Snake.py. state: {return_list}")

        return_list = np.array(return_list).flatten()  # Flatten the 2D list into a 1D numpy array
        return return_list


        



# game = Game()
# game.run()