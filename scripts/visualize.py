import pygame
import time
from environment import MazeEnv
from agent import QLearningAgent

CELL_SIZE = 80 
ROWS = 5    
COLS = 5 
# COLORS
WHITE = (255,255,255)
BLACK = (0,0,0)
BROWN = (139,69,19)
BLUE = (50,150,255)
GREEN = (0,255,0)
robot = pygame.image.load("robot.png")
robot = pygame.transform.scale(robot,(CELL_SIZE,CELL_SIZE))

pygame.init()
screen = pygame.display.set_mode((COLS*CELL_SIZE,ROWS*CELL_SIZE))
pygame.display.set_caption("RL Maze Solver")

env = MazeEnv()
agent = QLearningAgent(env.state_space,env.action_space)
agent.q_table = np.load("q_table.npy")

def draw():
    screen.fill(WHITE)
    for r in range(env.size):
        for c in range(env.size):
            rect = pygame.Rect(c*CELL_SIZE,r*CELL_SIZE,CELL_SIZE,CELL_SIZE)

            # walls
            if [r,c] in env.walls:
                pygame.draw.rect(screen,BROWN,rect)

            # goal
            elif [r,c]== [env.size-1,env.size-1]:
                pygame.draw.circle(screen,GREEN,rect)
                pygame.draw.rect(screen,(0,200,0),rect.inflate(10,10),3)

            # agent
            elif [r,c]== env.agent_pos:
                pygame.draw.circle(screen,BLUE,rect.center,CELL_SIZE//3)
                screen.blit(robot,rect)


            else:
                pygame.draw.rect(screen,WHITE,rect)
            
            pygame.draw.rect(screen,BLACK,rect,1)
    pygame.display.update()