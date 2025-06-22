from collections import deque
from typing import Deque, Tuple, Any, List
import random
from common.loggerConfig import logger

class ReplayMemory:
    """
    @class ReplayMemory
    @brief A fixed-size buffer to store experience tuples for DQN training.

    Stores transitions (state, action, next_state, reward, terminated) and
    supports random sampling for mini-batch training.
    Keeps a separate buffer for successful episodes.

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


        self.success_states: Deque[Any] = deque([], maxlen=max_size)
        self.success_actions: Deque[Any] = deque([], maxlen=max_size)
        self.success_next_states: Deque[Any] = deque([], maxlen=max_size)
        self.success_rewards: Deque[float] = deque([], maxlen=max_size)
        self.success_terminated: Deque[bool] = deque([], maxlen=max_size)

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

    def push_success(self, states: List | deque, actions: List | deque, next_states: List | deque, rewards: List | deque, terminateds: List | deque) -> None:
        """
        @brief Adds a transition to the replay buffer.

        @param state      List or Deque of the current state.
        @param action     List or Deque of the action taken.
        @param next_state List or Deque of the resulting next state.
        @param reward     List or Deque of the reward received.
        @param terminated List or Deque of whether the episode terminated after this transition.
        """
        self.success_states.extend(states)
        self.success_actions.extend(actions)
        self.success_next_states.extend(next_states)
        self.success_rewards.extend(rewards)
        self.success_terminated.extend(terminateds)

    def sample(self, sample_size: int, num_batches: int = 1, success_ratio: float = 0.50) -> Tuple[List[Any], List[Any], List[Any], List[float], List[bool]]:
        """
        @brief Randomly samples a batch of transitions from the memory, optionally including success transitions.

        @param sample_size   Number of samples to return.
        @param success_ratio Proportion of samples to draw from success memory (default: 0.5).
        @return A tuple containing lists of (states, actions, next_states, rewards, terminated_flags).
        """
        if num_batches <= 0:
            raise ValueError("Number of batches must be at least 1.")
        success_count = int(sample_size * num_batches * success_ratio)
        regular_count = sample_size * num_batches - success_count

        success_len = len(self.success_states)
        main_len = len(self)

        # TODO: Re-write all of this
        def _get_batch(indices_list: List[int]):
            return (
                    [self.states[i] for i in indices_list],
                    [self.actions[i] for i in indices_list],
                    [self.next_states[i] for i in indices_list],
                    [self.rewards[i] for i in indices_list],
                    [self.terminated[i] for i in indices_list],
                )
        # If not enough success samples, fall back to only regular memory
        if success_len < success_count:
            indices = random.sample(range(main_len), sample_size)
            if num_batches == 1:
                return _get_batch(indices)
            else:
                batch_indices = [indices[i::num_batches] for i in range(num_batches)]
                return [_get_batch(ind) for ind in batch_indices]


        # Otherwise, mix success and regular samples
        success_indices = random.sample(range(success_len), success_count)
        regular_indices = random.sample(range(main_len), regular_count)

        def _get_batch_with_success(suc_ind, reg_ind):
            states = [self.success_states[i] for i in suc_ind] + [self.states[i] for i in reg_ind]
            actions = [self.success_actions[i] for i in suc_ind] + [self.actions[i] for i in reg_ind]
            next_states = [self.success_next_states[i] for i in suc_ind] + [self.next_states[i] for i in reg_ind]
            rewards = [2.0*self.success_rewards[i] for i in suc_ind] + [self.rewards[i] for i in reg_ind]
            terminated_flags = [self.success_terminated[i] for i in suc_ind] + [self.terminated[i] for i in reg_ind]

            return states, actions, next_states, rewards, terminated_flags

        logger.debug(f"[Sampled] {len(success_indices)} from success, {len(regular_indices)} from regular")
        if num_batches == 1:
            return _get_batch_with_success(success_indices, regular_indices)
        else:
            success_split = [success_indices[i::num_batches] for i in range(num_batches)]
            regular_split = [regular_indices[i::num_batches] for i in range(num_batches)]
            return [_get_batch_with_success(s, r) for s, r in zip(success_split, regular_split)]

    def clear(self, include_success: bool = False) -> None:
        """
        @brief Clears the memory.
        """
        self.states.clear()
        self.actions.clear()
        self.next_states.clear()
        self.rewards.clear()
        self.terminated.clear()

        if include_success:
            self.success_states.clear()
            self.success_actions.clear()
            self.success_next_states.clear()
            self.success_rewards.clear()
            self.success_terminated.clear()

    def __len__(self) -> int:
        """
        @brief Returns the current number of elements in the memory.

        @return Number of stored transitions.
        """
        return len(self.states)

    def describe(self) -> str:
        return f"Main Memory: {len(self.states)}, Success Memory: {len(self.success_states)}"
