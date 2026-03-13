import numpy as np


class MazeEnv:
    def __init__(self,size=5):
        self.size = size
        self.state_space = size*size
        self.action_space = 4 # 0:up,1:down,2:left,3:right
        self.walls = [[1,1],[1,2],[3,0],[3,1],[3,3]]
        self.reset()
    
    def reset(self):
        #put the agent back at the start (0,0)
        self.agent_pos = [0,0]
        return 0

        
    
    def step(self, action):
        """Actions- 0=up,
                    1=down,
                    2=left,
                    3=right
           Returns- (next_state,reward,done)"""
        old_pos = list(self.agent_pos)
        row,col = self.agent_pos

        #movement logic
        if action ==0 and row >0: row-=1 #up
        elif action == 1 and row < self.size-1: row += 1 #down
        elif action == 2 and col> 0: col-=1 #left
        elif action == 3 and col < self.size-1: col+=1 #right

        #wall check: if new position is a wall,stay at old  position
        if [row, col] in self.walls:
            row,col = old_pos
            reward = -5 # penalty for hitting a wall
        else:
            reward = -1 # standard step cost

        self.agent_pos = [row,col]
        next_state = row*self.size + col

        # goal check

        done = False
        if self.agent_pos == [self.size-1,self.size-1]:
            reward = 20 #big reward for winning 
            done = True
        return next_state,reward ,done

    def visualize_maze(env,agent_pos):
        grid = np.full((env.size,env.size),".",dtype=str)
        

        # mark the walls
        for r,c in env.walls:
            grid[r,c] = "X"

        # mark the goal
        grid[env.size-1,env.size-1] = "G"

        # mark the agent's current position

        grid[agent_pos[0],agent_pos[1]] = "A"

        for row in grid:
            print("".join(row))
        print("-"*10)