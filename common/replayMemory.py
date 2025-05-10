from collections import deque
from typing import Deque, Tuple, Any, List
import random


class ReplayMemory:
    """
    @class ReplayMemory
    @brief A fixed-size buffer to store experience tuples for DQN training.

    Stores transitions (state, action, next_state, reward, terminated) and 
    supports random sampling for mini-batch training.

    @param max_size Maximum number of transitions to store in memory.
    """

    def __init__(self, max_size: int = 10000):
        """
        @brief Constructor for ReplayMemory.

        @param max_size The maximum number of elements the buffer can hold.
        """
        self.max_size = max_size

        self.states: Deque[Any] = deque([], maxlen=max_size)
        self.actions: Deque[Any] = deque([], maxlen=max_size)
        self.next_states: Deque[Any] = deque([], maxlen=max_size)
        self.rewards: Deque[float] = deque([], maxlen=max_size)
        self.terminated: Deque[bool] = deque([], maxlen=max_size)

    def push(self, state: Any, action: Any, next_state: Any, reward: float, terminated: bool) -> None:
        """
        @brief Adds a transition to the replay buffer.

        @param state      The current state.
        @param action     The action taken.
        @param next_state The resulting next state.
        @param reward     The reward received.
        @param terminated Whether the episode terminated after this transition.
        """
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.terminated.append(terminated)

    def sample(self, sample_size: int) -> Tuple[List[Any], List[Any], List[Any], List[float], List[bool]]:
        """
        @brief Randomly samples a batch of transitions from the memory.

        @param sample_size Number of samples to return.
        @return A tuple containing lists of (states, actions, next_states, rewards, terminated_flags).
        """
        indices = random.sample(range(len(self)), sample_size)
        return (
            [self.states[i] for i in indices],
            [self.actions[i] for i in indices],
            [self.next_states[i] for i in indices],
            [self.rewards[i] for i in indices],
            [self.terminated[i] for i in indices],
        )

    def __len__(self) -> int:
        """
        @brief Returns the current number of elements in the memory.

        @return Number of stored transitions.
        """
        return len(self.states)
