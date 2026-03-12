import numpy as np 

class QLearningAgent:
    def __init__(self,state_size,action_size,lr=0.1,gamma = 0.9,epsilon=1.0):
        self.q_table = np.zeros((state_size,action_size))
        self.lr = lr #learning rate(alpha)
        self.gamma = gamma # Discount Factor 
        self.epsilon = epsilon #Exploration Rate

    def choose_action(self,state):
        #Epsilon-greedy logic
        if np.random.uniform(0,1) < self.epsilon:
            return np.random.randint(0,4) # Explore:Random move
        else:
            return np.argmax(self.q_table[state]) # exploit:best known move,return the index of the largest value of q table
        
    def learn(self,state,action,reward, next_state):
        # the bellman equation implementation
        old_value = self.q_table[state,action]
        next_max = np.max(self.q_table[next_state])

        #update Q-value
        new_value = old_value + self.lr*(reward + self.gamma*next_max-old_value)
        self.q_table[state,action]=new_value