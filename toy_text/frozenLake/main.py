from tqdm import tqdm
from toy_text.frozenLake.frozenLakeMaps import FrozenLakeMaps
from toy_text.frozenLake.frozenLakeAgent import FrozenLakeAgent
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import gymnasium as gym
import common.utils as utils
from typing import Dict, List
from common.loggerConfig import logger

def run(train: bool,
        test: bool,
        dataset: Dict[str, List[str]],
        episodes: int,
        render: bool,
        learning_rate: float,
        start_epsilon: float,
        final_epsilon: float,
        test_size: float = 0.2,
        model_save_path: str = None,
        model_load_path: str = None,
        plot: bool = True,
        is_slippery: bool = False,
        enable_dqn_dueling: bool = False,
        enable_dqn_double: bool = False,
        hidden_layer_dims: List[int] = [64, 16],
        max_episode_steps: int = 500):

    # 1) Split into training and testing maps
    train_maps, test_maps = utils.split_dict(dataset, test_size)
    # train_maps, test_maps = dataset, dataset
    # 2) Normalize paths
    save_path = Path(model_save_path) if model_save_path else None
    load_path = Path(model_load_path) if model_load_path else None

    # 3) TRAINING
    if train:
        logger.info(f"=== Starting training on {len(train_maps)} maps for {episodes} episodes each ===")
        agent = FrozenLakeAgent(
            maps=train_maps,
            learning_rate=learning_rate,
            initial_epsilon=start_epsilon,
            final_epsilon=final_epsilon,
            existing_dqn_path=load_path,
            save_dqn_path=save_path,
            is_slippery=is_slippery,
            enable_dqn_dueling=enable_dqn_dueling,
            enable_dqn_double=enable_dqn_double,
            hidden_layer_dims=hidden_layer_dims,
            max_episode_steps=max_episode_steps
        )
        agent.train(
            num_episodes=episodes,
            render=render
        )
        if plot:
            # the agent’s run method already calls save_graph periodically,
            # and leaves the final plot at agent.graph_file
            logger.info(f"Training complete. Graph saved to {agent.graph_file}")

    # 4) TESTING
    if test:
        # for testing, we always load the trained model (from save_path if provided)
        load_for_test = save_path or load_path
        if load_for_test is None or not load_for_test.exists():
            raise FileNotFoundError(f"No model file found at {load_for_test!r} to load for test run")

        logger.info(f"=== Starting evaluation on {len(test_maps)} maps for {episodes} episodes each ===")
        tester = FrozenLakeAgent(
            maps=test_maps,
            learning_rate=learning_rate,      # lr is unused in test but required by ctor
            initial_epsilon=0.00,    # start & final equal so no exploration
            final_epsilon=0.00,
            existing_dqn_path=load_for_test,
            save_dqn_path=save_path,
            is_slippery=is_slippery,
            enable_dqn_dueling=enable_dqn_dueling,
            enable_dqn_double=enable_dqn_double,
            hidden_layer_dims=hidden_layer_dims,
            max_episode_steps=max_episode_steps
        )
        tester.evaluate(
            num_episodes=episodes,
            render=render
        )
        logger.info("Testing complete.")