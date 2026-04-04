import matplotlib
from evaluate import evaluate

def eval_graphs(agent, env, episodes):
    avg_rewards, ep_length = evaluate(agent, env, episodes)
