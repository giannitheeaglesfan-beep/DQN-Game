"""Train a DQN agent to play CreatureBattleEnv against its built-in heuristic opponent.

Usage:
    python train.py

Produces `dqn_battle_agent.pth` containing the trained policy network's
weights, which play_gui.py and app.py both load for inference.
"""

from __future__ import annotations

from battle_env import CreatureBattleEnv
from model import DQNAgent

NUM_EPISODES = 2000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 1800  # reach EPSILON_END by this episode, then hold
GAMMA = 0.99
BATCH_SIZE = 64
TARGET_SYNC_EVERY_STEPS = 500
BUFFER_CAPACITY = 50_000
LOG_EVERY = 100
MODEL_PATH = "dqn_battle_agent.pth"


def epsilon_for_episode(episode: int) -> float:
    """Linear decay from EPSILON_START to EPSILON_END over EPSILON_DECAY_EPISODES."""
    fraction = min(1.0, episode / EPSILON_DECAY_EPISODES)
    return EPSILON_START + fraction * (EPSILON_END - EPSILON_START)


def main() -> None:
    env = CreatureBattleEnv()
    agent = DQNAgent(buffer_capacity=BUFFER_CAPACITY)

    total_steps = 0
    window_rewards: list[float] = []
    window_wins = 0
    window_episodes = 0

    for episode in range(1, NUM_EPISODES + 1):
        epsilon = epsilon_for_episode(episode)
        state, _ = env.reset()
        episode_reward = 0.0
        terminated = False
        truncated = False
        won = False

        while not (terminated or truncated):
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)

            agent.replay_buffer.push(state, action, reward, next_state, terminated or truncated)
            agent.update(BATCH_SIZE, GAMMA)

            state = next_state
            episode_reward += reward
            total_steps += 1

            if total_steps % TARGET_SYNC_EVERY_STEPS == 0:
                agent.sync_target()

            if terminated and env.opponent.hp <= 0:
                won = True

        window_rewards.append(episode_reward)
        window_wins += int(won)
        window_episodes += 1

        if episode % LOG_EVERY == 0:
            avg_reward = sum(window_rewards) / len(window_rewards)
            win_rate = 100.0 * window_wins / window_episodes
            print(
                f"Episode {episode:5d} | epsilon {epsilon:.3f} | "
                f"avg reward (last {LOG_EVERY}): {avg_reward:6.2f} | "
                f"win rate: {win_rate:5.1f}%"
            )
            window_rewards.clear()
            window_wins = 0
            window_episodes = 0

    agent.save(MODEL_PATH)
    print(f"Training complete. Saved weights to {MODEL_PATH}")


if __name__ == "__main__":
    main()
