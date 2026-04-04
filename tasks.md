“RL Fails in Interesting Ways”

- If for example you did this with snake, train the model and condition it to associate fruits and with being good by rewarding it. Then, change the rules to make it so that in the short run, getting fruits is actually bad. After a certain point, in the middle of the game, it actually becomes good though, but long run getting fruits becomes bad. Observe how it behaves and adapts while still prioritizing score and visualize it, because it will try to end the game early, then keep going, and then going back to ending it early.


- Failure Case 1: Reward Hacking
  - loop near the goal forever
  - farm “close distance” reward instead of finishing
  - +3 for moving closer to goal 
  - +5 for reaching goal

- Failure Case 2: Lazy Agent
  - small penalty per step
  - small reward for goal
  - "Doing nothing is safer"

- Failure Case 3: Risk Avoidance Gone Wrong
  - massive penalty for dying with small reward for living
  - agent refuses to take optimal paths
  - plays overly safe and inefficiently

- Failure Case 4: Exploiting Bugs
  - reward tied to a metric you didn’t think through



rl_project/  
│  
├─ env.py # SnakeEnv or any other environment  
├─ agent.py # RL agent / neural network  
├─ evaluate.py # Evaluation of trained agent  
├─ config.py # Hyperparameters  
└─ utils.py # Logging, plotting, seeds, etc.

