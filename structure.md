“RL Fails in Interesting Ways”

- Train a model on a normal snake environment, then change the rule

- Failure Case 1: 
  - Change the rules to make it so that in the short run, getting fruits is actually bad. 
  - After a certain point, in the middle of the game, it actually becomes good though, but long run getting fruits becomes bad. 
  - It will try to end the game early, then keep going, and then going back to ending it early.

- Failure Case 2: Unexpected bug
  - reward tied to a metric you didn’t think through
  - original bug in stepping caused snake to not get longer. 
  - tail would delete always but head was never removed.
  - Fixing the issue with this older model caused very interesting issues as it was learning to adapt to a newer 
environment. 
  - Bug was kept due to unintended results which make it a good train/test environment to see how behavior 
adapts.
  
- Failure Case 3: Risk Avoidance Gone Wrong
  - massive penalty for dying with small reward for living
  - agent refuses to take optimal paths
  - plays overly safe and inefficiently

- Failure Case 4: Lazy Agent
  - small penalty per step
  - small reward for goal
  - "Doing nothing is safer"

- Failure Case 5: Reward Hacking
  - loop near the goal forever
  - farm “close distance” reward instead of finishing
  - +3 for moving closer to goal 
  - +5 for reaching goal


rl_project/  
│  
├─ env.py # SnakeEnv or any other environment  
├─ agent.py # RL agent / neural network  
├─ evaluate.py # Evaluation of trained agent  
├─ config.py # Hyperparameters  
└─ utils.py # Logging, plotting, seeds, etc.

