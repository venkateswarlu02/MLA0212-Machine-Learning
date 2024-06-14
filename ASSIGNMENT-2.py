import numpy as np


num_products = 3  
num_competitors = 2 
num_prices = 5  
num_states = num_products * num_prices 
num_actions = num_prices  


Q = np.zeros((num_states, num_actions))

learning_rate = 0.1
discount_rate = 0.99
exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.001
max_episodes = 1000  


def reward(sales, revenue):
    return sales * revenue


def state_index(product, price):
    return product * num_prices + price


for episode in range(max_episodes):
    
    competitor_prices = np.random.randint(1, num_prices+1, size=num_products)
    
   
    sales = np.random.randint(10, 100, size=num_products)
    
    total_reward = 0
    
    for product in range(num_products):
        state = state_index(product, competitor_prices[product] - 1)  
        
        
        exploration_threshold = np.random.uniform(0, 1)
        if exploration_threshold > exploration_rate:
            action = np.argmax(Q[state, :])
        else:
            action = np.random.randint(0, num_actions)
        
        
        price = action + 1
        revenue = price * sales[product]
        
      
        Q[state, action] = Q[state, action] * (1 - learning_rate) + \
                            learning_rate * (reward(sales[product], revenue) + \
                                            discount_rate * np.max(Q[state, :]))
        
        
        total_reward += reward(sales[product], revenue)
    
  
    exploration_rate = min_exploration_rate + \
                        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
    
    
    print("Episode:", episode+1, "Total Reward:", total_reward)


print("Learned Q-table:")
print(Q)
