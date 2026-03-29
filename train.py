from scripts.environment import MazeEnv
from scripts.agent import QLearningAgent
import numpy as np

env = MazeEnv(size=5)
agent = QLearningAgent(env.state_space,env.action_space)

episodes = 1000
rewards_history = []

for e in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state,reward ,done = env.step(action)

        agent.learn(state,action,reward,next_state)

        state = next_state
        total_reward += reward
    #gradually reduce exploration as the agent gets smarter
    agent.epsilon = max(0.01,agent.epsilon*0.995)
    #here we use max function by this choose max number between these two numbers ,so epsilon never goes below 0.01
    #max(0.01, 0.2) = 0.2
    #max(0.01, 0.005) = 0.01
    rewards_history.append(total_reward)

    if(e+1)%100 == 0:
        print(f"Episode {e+1}: Total Reward = {total_reward}")
        # print episodes in 100s total reward

print("Training Finished!")

np.save("q_table.npy", agent.q_table)
print("Q-table saved as q_table.npy")
 
# test the agent
steps = 0


state = env.reset()
done = False

print("Starting Position:", env.agent_pos)
env.visualize_maze(env.agent_pos)
moves = ['Up','Down','Left','Right']
while not done  and steps< 25 :
    action = np.argmax(agent.q_table[state])#This is equivalent to epsilon = 0 (pure exploitation),the agent always chooses the best known action.
    move = moves[action]
    state,reward,done = env.step(action)
    print(f"Step {steps+1}: Agent move {move}")
    env.visualize_maze(env.agent_pos)
    steps+=1
if done:
    print("Goal Reached in ", steps,'steps!👾🎉')
else:
    print("Agent got lost or stuck.😒")
        
