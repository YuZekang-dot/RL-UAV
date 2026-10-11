"""Stable constants expected by legacy platform configuration."""

from rl_uav.algorithms import PPOConfig
from rl_uav.config import ModelConfig, PlannerConfig


class Config:
    TASK_NAME = "ObstacleHover"
    RAW_OBS_DIM = 95
    SHARED_OBS_DIM = ModelConfig().shared_obs_dim
    TASK_OBS_DIM = ModelConfig().max_task_obs_dim
    ACTION_DIM = ModelConfig().action_dim
    LEARNING_RATE = PPOConfig().learning_rate
    GAMMA = PPOConfig().gamma
    LAM = PPOConfig().gae_lambda
    CLIP_PARAM = PPOConfig().clip_ratio
    ARENA_SIZE = PlannerConfig().arena_size

