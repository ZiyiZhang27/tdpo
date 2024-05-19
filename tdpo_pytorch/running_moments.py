import numpy as np
from collections import deque


class PerPromptRunningMoments:
    def __init__(self, buffer_size, min_count, mode):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.stats = {"rewards": {}, "temporal_rewards": {}}
        self.mode = mode

    def state_dict(self):
        return self.stats

    def load_state_dict(self, state_dict):
        self.stats = state_dict

    def update_rewards(self, prompts, rewards):
        prompts = np.array(prompts)
        rewards = np.array(rewards)
        unique = np.unique(prompts)
        final_rewards = np.empty_like(rewards)
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats["rewards"]:
                self.stats["rewards"][prompt] = deque(maxlen=self.buffer_size)
            self.stats["rewards"][prompt].extend(prompt_rewards)

            if len(self.stats["rewards"][prompt]) < self.min_count:
                mean = np.mean(rewards)
                std = np.std(rewards)
            else:
                mean = np.mean(self.stats["rewards"][prompt])
                std = np.std(self.stats["rewards"][prompt])

            if self.mode == "norm":
                final_rewards[prompts == prompt] = (prompt_rewards - mean) / (std + 1e-6)
            if self.mode == "scaling":
                final_rewards[prompts == prompt] = prompt_rewards / (std + 1e-6)

        return final_rewards

    def update_temporal_rewards(self, prompts, temporal_rewards):
        prompts = np.array(prompts)
        temporal_rewards = np.array(temporal_rewards)
        step_mean = np.mean(temporal_rewards, axis=1)
        step_std = np.std(temporal_rewards, axis=1)
        unique = np.unique(prompts)
        final_temporal_rewards = np.empty_like(temporal_rewards)
        for prompt in unique:
            prompt_temporal_rewards = temporal_rewards[prompts == prompt]
            if prompt not in self.stats["temporal_rewards"]:
                self.stats["temporal_rewards"][prompt] = deque(maxlen=self.buffer_size)
            self.stats["temporal_rewards"][prompt].extend([step_mean, step_std])

            if len(self.stats["temporal_rewards"][prompt]) < self.min_count:
                mean = np.mean(temporal_rewards)
                std = np.std(temporal_rewards)
            else:
                mean = np.mean(self.stats["temporal_rewards"][prompt][0])
                std = np.sqrt(
                    np.mean(
                        np.square(self.stats["temporal_rewards"][prompt][0]) + np.square(self.stats["temporal_rewards"][prompt][1])
                    ) - np.square(mean)
                )
            final_temporal_rewards[prompts == prompt] = (prompt_temporal_rewards - mean) / (std + 1e-6)

        return final_temporal_rewards


class RunningMoments:
    def __init__(self):
        """
        Calculates the running mean and standard deviation of a data stream.
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24

    def update(self, xs):
        """Updates running moments from batch's moments computed across ranks"""
        xs_count = xs.shape[0]
        xs_mean, xs_var = np.mean(xs), np.var(xs)

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = np.sqrt(self.var * tot_count / (tot_count - 1))
        self.count = tot_count
