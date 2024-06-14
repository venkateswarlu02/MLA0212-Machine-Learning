import numpy as np


num_states = 10  
num_actions = 4  
initial_state = 0  
goal_state = 9 
max_episodes = 1000  
max_steps_per_episode = 100  


Q = np.zeros((num_states, num_actions))


learning_rate = 0.1
discount_rate = 0.99
exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.001


def reward(state):
    return 1 if state == goal_state else 0


for episode in range(max_episodes):
    state = initial_state
    done = False
    total_reward = 0
    
    for step in range(max_steps_per_episode):
     
        exploration_threshold = np.random.uniform(0, 1)
        if exploration_threshold > exploration_rate:
            action = np.argmax(Q[state, :])
        else:
            action = np.random.randint(0, num_actions)
        
        
        new_state = min(max(0, state + action - 1), num_states - 1)  # Ensure new_state stays within bounds
        
        
        Q[state, action] = Q[state, action] * (1 - learning_rate) + \
                            learning_rate * (reward(new_state) + discount_rate * np.max(Q[new_state, :]))
        
       
        total_reward += reward(new_state)
        
     
        state = new_state
        
      
        if state == goal_state:
            done = True
            break
    
 
    exploration_rate = min_exploration_rate + \
                        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
    
    
    print("Episode:", episode+1, "Total Reward:", total_reward)


print("Learned Q-table:")
print(Q)
