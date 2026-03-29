import pygame
import time
from scripts.environment import MazeEnv
from scripts.agent import QLearningAgent
import numpy as np

# pygame setup
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

# environment and agent

env = MazeEnv()
agent = QLearningAgent(env.state_space,env.action_space)
agent.q_table = np.load("q_table.npy")

# draw maze function

def draw():
    screen.fill(WHITE)
    for r in range(env.size):
        for c in range(env.size):
            rect = pygame.Rect(c*CELL_SIZE,r*CELL_SIZE,CELL_SIZE,CELL_SIZE)
            pygame.draw.rect(screen,WHITE,rect)
            # walls
            if [r,c] in env.walls:
                pygame.draw.rect(screen,BROWN,rect)

            # goal
            if [r,c]== [env.size-1,env.size-1]:
                cx,cy = rect.center
                size = CELL_SIZE//3
                points = [
                    (cx,cy -size),#top

                    (cx+size ,cy),#right
                    (cx,cy+size), # bottom
                    (cx-size,cy) # left
                ]
                pygame.draw.polygon(screen,GREEN,points)
                pygame.draw.polygon(screen,(0,200,0),points,3)
                

            # agent
            if [r,c]== env.agent_pos:
                screen.blit(robot,rect)
            pygame.draw.rect(screen,BLACK,rect,1)
    pygame.display.update()



moves = ['Up','Down','Left','Right']
while True:

    state = env.reset()
    done = False
    steps = 0
    
    print("\n Starting new run!")

    while not done and steps<25:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        action = np.argmax(agent.q_table[state])
        state,reward,done = env.step(action)
        print(f"Step {steps+1}: Agent move {moves[action]}")
        draw()
        time.sleep(0.5) #small delay to see movemts
        steps +=1
    if done :
        print("Goal reached in", steps, "steps! ")
    else:
        print("Agent got stuck.")
    time.sleep(1)
 