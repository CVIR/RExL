from gym.envs.registration import register

register(
    id='explainAI-v0',
    entry_point='explainableAI.envs:AttentionEnv',
    max_episode_steps=196,
)
