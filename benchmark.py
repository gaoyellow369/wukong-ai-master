# training.py
# coding=utf-8
import random
import numpy as np
import cv2
import time
import actions
from models.dqn import DeepQNetwork
from models.qn import QLearningNetwork
from models.resnet import ResnetClassification
import window
import judge
from context import Context
from log import log
import yaml
import csv
import base64

BENCHMARK_MODEL_EPISODES = [0, 200, 400, 600, 800, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
def reload_qn_model(episode):
    closest_episode = max(filter(lambda x: x <= episode, BENCHMARK_MODEL_EPISODES))
    file_name = f"model_weight/qn_model_{closest_episode}.pth"
    model = QLearningNetwork(
        context_dim=5,
        action_dim=5,
        config={},
        model_file=file_name
    )
    model.epsilon = 0
    log(f"Episode: {episode}, Model loaded from {file_name}.\n")
    return model

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_type = config['model']['type']
model_config = config['model'].get(model_type, {})
training_config = config['training']
env_config = config['environment']
model_file = config['model']['model_file']

# Dynamically import the model class using importlib
model_module = f"models.{model_type.lower()}"  # e.g., "models.dqn"
#Agent = getattr(importlib.import_module(model_module), model_type)

def get_boss_state(resnet_result, is_freeze):
    if is_freeze:
        return 2
    return resnet_result


if __name__ == "__main__":
    log(f"Agent load as models.{model_type.lower()}.\n")
    # Initialize Context and Agent
    log("Initializing Context and Agent.\n")
    ctx = Context()
    ctx.updateContext(0)

    classifier = ResnetClassification()
    classifier.load_from_file('model_weight/resnet_model.pth')

    context_dim = len(ctx.get_states())
    state_dim = (env_config['height'], env_config['width'])
    action_dim = env_config['action_size']

    ctx = actions.pause_game(ctx)
    ENABLE_TRAINING = False
    
    
    csv_file = open('game_data.csv', 'a', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['TimeStamp', 'Action', 'Index', 'Damage', 'Next State Image'])
    consecutive_attack = 0
    MAX_CONECUTIVE = 5
    for i in range(10):
        for e in BENCHMARK_MODEL_EPISODES:
            episode = e + i
            ctx.begin_time = time.time()
            agent = reload_qn_model(episode)
            # Get initial state
            state_image_origin = window.get_main_screen_window().color[:, :, :3]
            boss_freeze = False
            boss_state = get_boss_state(classifier.classify(state_image_origin), boss_freeze)
            state = agent.discretize_state(ctx.get_states())

            last_time = time.time()
            target_step = 0
            total_reward = 0
            ctx.done = 0

            N = 1  # Number of actions to accumulate before updating
            state_buffer = []
            cur_reward = 0
            last_freeze_time = 0
            ctx.medicine_nums = 5
            while True:
                initial_reward = 0
                last_time = time.time()
                if last_time - last_freeze_time < 5:
                    boss_freeze  = True
                else:
                    boss_freeze = False

                action = agent.choose_action(state)
                if ctx.next_boss_health < 10:
                    action = 0 # Do not kill the boss
                log(f"Boss state: {state[0]}, cooldown: {state[1]}, combo: {state[2]}, drink: {state[3]}, health:{state[4]}, action: {action}")

                if action == 3:
                    if not actions.is_skill_1_in_cd():
                        last_freeze_time = time.time()
                        boss_freeze = True
                

                ctx = actions.take_action(action, ctx)
                ctx.updateContext(action)
        
                log("Action took {:.2f} seconds".format(time.time() - last_time))
                screen_after  = window.get_main_screen_window()

                next_state_image_origin = screen_after.color[:, :, :3]

                boss_state = get_boss_state(classifier.classify(next_state_image_origin), boss_freeze)
                ctx.update_boss_state(boss_state)

                next_state = agent.discretize_state(ctx.get_next_states())

                ctx = judge.action_judge(ctx, initial_reward, episode=episode)

                if ctx.emergence_break == 100:
                    log("Emergency break activated.\n")
                    agent.save_model()
                    log("Model Saved.\n")
                    ctx.paused = True
                state = next_state  # Update the state
                state_image_origin = next_state_image_origin

                # Control pause
                before_pause = ctx.paused
                ctx = actions.pause_game(ctx, ctx.emergence_break)
                if before_pause and not ctx.paused:
                    ctx.emergence_break = 0

                if ctx.done == 1:
                    break