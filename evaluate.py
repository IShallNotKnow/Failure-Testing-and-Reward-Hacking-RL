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
            next_state, reward, done, score = env.step(action)
            total_reward += reward
            state = next_state

        avg_reward = total_reward/env.timestep
        avg_rewards.append(avg_reward)
        episode_lengths.append(env.timestep)

        if (episode+1) % 100 == 0:
            print(f"Episode {episode+1} | Avg Reward: {avg_rewards[-1]:.2f} | Length: {env.timestep}")

    agent.exploration_rate = original_epsilon
    return avg_rewards, episode_lengths