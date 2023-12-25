import gym
import random
import numpy as np
import time
from collections import deque
import pickle
from collections import defaultdict

print(gym.__version__)
print(np.__version__)


EPISODES =  20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0

if __name__ == "__main__":

    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v1")
    obs = env.reset(seed=1)

    # You will need to update the Q_table in your iteration
    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()

        ##########################################################
        # YOU DO NOT NEED TO CHANGE ANYTHING ABOVE THIS LINE
        # TODO: Replace the following with Q-Learning

        while (not done):
            
            # generate random variable between 0 and 1
            random_variable = random.uniform(0,1)

            # Exploration with a probability of EPSILON: choose a random action
            if random_variable < EPSILON:
                action = env.action_space.sample() # currently only performs a random action.
            
            # Exploitation with a probabilty of 1 - EPSILON: choose a best known action
            else:
                possible_action = np.array([Q_table[(obs, i)] for i in range(env.action_space.n)])
                action =  np.argmax(possible_action)
            
            # take the action and get next state and reward    
            next_obs, reward, done, info = env.step(action)    
            
            # find all the possible Q value and compute the new max Q value
            possible_action = np.array([Q_table[(next_obs,i)] for i in range(env.action_space.n)])
            max_value = np.amax(possible_action)
            
            # Bellman Equation for new max Q value
            Q_table[(obs, action)] = (1 - LEARNING_RATE) * Q_table[(obs, action)] + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_value)
            
            # update episode reward and the state
            episode_reward += reward # update episode reward
            obs = next_obs

        # update epsilon after value iteration
        EPSILON = EPSILON * EPSILON_DECAY
        
        # END of TODO
        # YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE
        ##########################################################

        # record the reward for this episode
        episode_reward_record.append(episode_reward) 
     
        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    
    #### DO NOT MODIFY ######
    model_file = open('Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()
    #########################
