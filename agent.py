import torch
import random
import numpy as np
from snake_game import SnakeGameAI, Direction, Point
from collections import deque
from model import QTrainer, Linear_QNet
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 #learning rate

class Agent:
    def __init__(self):
        self.number_games = 0
        self.eps = 0 #entropy
        self.gamma = 0.9 #discount rate
        self.memory = deque(maxlen=MAX_MEMORY) #if we exceed memory size, popleft
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        #points next to the head in all directions
        #we will use them to check whether there is a danger or not
        point_r = Point(head.x + 20, head.y) #20 is block size
        point_l = Point(head.x - 20, head.y)
        point_d = Point(head.x, head.y + 20)
        point_u = Point(head.x, head.y - 20)

        dir_is_r = game.direction == Direction.RIGHT
        dir_is_l = game.direction == Direction.LEFT
        dir_is_u = game.direction == Direction.UP
        dir_is_d = game.direction == Direction.DOWN

        state = [
            #danger is straight (without changing direction)
            (dir_is_r and game.is_collision(point_r)) or
            (dir_is_l and game.is_collision(point_l)) or
            (dir_is_u and game.is_collision(point_u)) or
            (dir_is_d and game.is_collision(point_d)),
            #danger is right (clockwise)
            (dir_is_r and game.is_collision(point_d)) or
            (dir_is_l and game.is_collision(point_u)) or
            (dir_is_u and game.is_collision(point_r)) or
            (dir_is_d and game.is_collision(point_l)),
            #danger is left (inverse clockwise)
            (dir_is_r and game.is_collision(point_u)) or
            (dir_is_l and game.is_collision(point_d)) or
            (dir_is_u and game.is_collision(point_l)) or
            (dir_is_d and game.is_collision(point_r)),

            #moving directions : only one can be true
            dir_is_l,
            dir_is_r,
            dir_is_u,
            dir_is_d,

            #where is the food
            game.food.x < game.head.x, #left
            game.food.x > game.head.x, #right
            game.food.y < game.head.y, #up
            game.food.y > game.head.y, #down
            ]

        return np.array(state, dtype=int)


    def remember(self, state, action, reward, next_state, game_over):
        #we just save everything in the memory
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE) #returns a list tuples of (state, action, reward, next_state, game_over)
        else:
            sample = self.memory

        #put all the states together, actions together,....
        states, actions, rewards, next_states, game_overs = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        #we train it only for one game step
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        #random moves 
        #tradoff between exploration and exploitation
        self.eps = 80 - self.number_games
        move = [0,0,0]
        #when epsilon gets smaller, we don't get much random moves : no more exploration
        if random.randint(0, 200) < self.eps:
            move_idx = random.randint(0, 2)
            move[move_idx] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) #returns an array like [6.0, 5.4, 9.2]
            move_idx = torch.argmax(prediction).item() #returns the max value
            move[move_idx] = 1

        return move


#end of Agent class

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        #get old state of the game
        old_state = agent.get_state(game)
        #get move from the agent
        move = agent.get_action(old_state)
        #play this move and get new state
        reward, game_over, score = game.play_step(move)
        new_state = agent.get_state(game)
        #train short memory (only for one step)
        agent.train_short_memory(old_state, move, reward, new_state, game_over)
        agent.remember(old_state, move, reward, new_state, game_over) #save that in memory

        if game_over:
            #train long memory
            game.reset()
            agent.number_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            print('Game number : ', agent.number_games, 'Score : ', score, 'Record : ', record)

            #plotting
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.number_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            

if __name__ == '__main__':
    train()