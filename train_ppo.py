import pygame
import torch

from env import PingPongEnv
from ppo_agent import PPOAgent


def train():
    env = PingPongEnv()
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    agent = PPOAgent(input_dim, output_dim)

    max_episodes = 1000
    max_timesteps = 500
    render_frequency = 10  # Render every 10 episodes for visualization

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0

        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        dones = []

        for t in range(max_timesteps):
            # Ensure Pygame events are handled
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

            if episode % render_frequency == 0:
                env.render()
                pygame.time.delay(30)  # Delay to make the rendering visible

            action, log_prob = agent.select_action(state)
            value = agent.value_net(torch.FloatTensor(state)).item()
            next_state, reward, done, _, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob.item())
            values.append(value)
            dones.append(done)

            episode_reward += reward
            state = next_state

            if done:
                break

        # Compute advantages and returns
        advantages, returns = agent.compute_advantages(rewards, values, dones)

        # Update the PPO agent
        agent.update(states, actions, log_probs, returns, advantages)

        print(f"Episode {episode + 1}: Reward = {episode_reward}")

        # Optional: Save the model every few episodes
        if (episode + 1) % 100 == 0:
            torch.save(agent.policy_net.state_dict(), f"models/policy_net_{episode+1}.pth")
            torch.save(agent.value_net.state_dict(), f"models/value_net_{episode+1}.pth")

    env.close()

if __name__ == "__main__":
    train()
