def evaluate(agent, env, episodes=100):
    avg_rewards = []
    episode_lengths = []
    original_epsilon = agent.exploration_rate
    agent.exploration_rate = 0

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = agent.choose_action(state, env.get_actions())
            next_state, reward, done, score = agent.step(action)
            steps += 1
            total_reward += reward
            state = next_state

        avg_rewards.append(total_reward/steps)
        episode_lengths.append(steps)

    agent.exploration_rate = original_epsilon
    return avg_rewards, episode_lengths
