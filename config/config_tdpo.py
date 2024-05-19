import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base_tdpo.py"))


def aesthetic():
    config = base.get_config()

    config.pretrained.model = "CompVis/stable-diffusion-v1-4"

    config.num_epochs = 100
    config.use_lora = True
    config.save_freq = 1
    config.num_checkpoint_limit = 100000000

    # this corresponds to 8 * 8 * 4 = 256 samples per epoch when using 8 GPUs.
    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    # this corresponds to (8 * 4) / (1 * 16) = 2 gradient updates per epoch.
    config.train.batch_size = 1
    config.train.gradient_accumulation_steps = 16

    # reward function for training
    config.reward_fn1 = "aesthetic_score"
    config.reward_name1 = "aesthetic"
    # reward functions for cross-reward evaluation
    config.reward_fn2 = "ImageReward"
    config.reward_name2 = config.reward_fn2
    config.reward_fn3 = "PickScore"
    config.reward_name3 = config.reward_fn3
    config.reward_fn4 = "hpsv2"
    config.reward_name4 = config.reward_fn4

    config.prompt_fn = "simple_animals"
    config.per_prompt_running_moments = {
        "buffer_size": 32,
        "min_count": 16,
    }

    return config


def pickscore():
    config = aesthetic()

    config.num_epochs = 100

    # reward function for training
    config.reward_fn1 = "PickScore"
    config.reward_name1 = config.reward_fn1
    # reward functions for cross-reward evaluation
    config.reward_fn2 = "hpsv2"
    config.reward_name2 = config.reward_fn2
    config.reward_fn3 = "ImageReward"
    config.reward_name3 = config.reward_fn3
    config.reward_fn4 = "aesthetic_score"
    config.reward_name4 = "aesthetic"

    config.train.gradient_accumulation_steps = 16

    config.prompt_fn = "simple_animals"
    config.per_prompt_running_moments = {
        "buffer_size": 32,
        "min_count": 16,
    }

    return config


def hpsv2():
    config = aesthetic()

    config.num_epochs = 100

    # reward function for training
    config.reward_fn1 = "hpsv2"
    config.reward_name1 = config.reward_fn1
    # reward functions for cross-reward evaluation
    config.reward_fn2 = "PickScore"
    config.reward_name2 = config.reward_fn2
    config.reward_fn3 = "ImageReward"
    config.reward_name3 = config.reward_fn3
    config.reward_fn4 = "aesthetic_score"
    config.reward_name4 = "aesthetic"

    config.train.gradient_accumulation_steps = 16

    config.prompt_fn = "hpsv2_photo"
    config.per_prompt_running_moments = {
        "buffer_size": 32,
        "min_count": 16,
    }

    return config


def get_config(name):
    return globals()[name]()
