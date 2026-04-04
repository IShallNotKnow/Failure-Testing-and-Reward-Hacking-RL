def evaluate(agent, env, episodes):
    avg_rewards = []
    episode_lengths = []
    original_epsilon = agent.exploration_rate
    agent.exploration_rate = 0

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            env.render()
            action = agent.choose_action(state, env.get_actions())
            next_state, reward, done, score = agent.step(action)
            total_reward += reward
            state = next_state

        avg_rewards.append(total_reward/env.timesteps)
        episode_lengths.append(env.timesteps)

    agent.exploration_rate = original_epsilon
    return avg_rewards, episode_lengths