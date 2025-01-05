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
import importlib
import grabscreen

def capture_n_images(window, n=3, interval=0.05):
    state_images = []
    for i in range(n):
        time.sleep(interval)  # 50ms delay
        frame = grabscreen.grab_screen()
        window.update(frame)
        gray_image = cv2.cvtColor(window.color, cv2.COLOR_BGR2GRAY)
        state_images.append(gray_image)
    state_images_array = np.array(state_images)

    # Reshape the array to match the original state_image_array dimensions
    state_images_array = state_images_array[np.newaxis, :, :, :]
    return state_images_array

import csv
import base64


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

    agent = QLearningNetwork(
        context_dim=5,
        action_dim=5,
        config={},
        model_file='model_weight/qn_model.pth'
    )
    # agent = DeepQNetwork(
    #     context_dim=6,
    #     action_dim=5,
    #     config={},
    #     model_file='model_weight/dqn_model.pth'
    # )
    ctx = actions.pause_game(ctx)
    agent.epsilon = 0
    ENABLE_TRAINING = False

    csv_file = open('game_data.csv', 'a', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['TimeStamp', 'Action', 'Index', 'Damage', 'Next State Image'])
    consecutive_attack = 0
    MAX_CONECUTIVE = 5
    for episode in range(training_config['episodes']):
        ctx.begin_time = time.time()
        # Get initial state
        state_image_origin = window.get_main_screen_window().color[:, :, :3]
        state_image = cv2.resize(state_image_origin, state_dim[::-1])
        state_image_array = np.array(state_image).transpose(2, 0, 1)[np.newaxis, :, :, :]
        # state_image_array = capture_n_images(window.get_main_screen_window())
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
            next_state_image = cv2.resize(next_state_image_origin, state_dim[::-1])
            next_state_image_array = np.array(next_state_image).transpose(2, 0, 1)[np.newaxis, :, :, :]
            # next_state_image_array = capture_n_images(window.get_main_screen_window())

            boss_state = get_boss_state(classifier.classify(next_state_image_origin), boss_freeze)
            ctx.update_boss_state(boss_state)

            next_state = agent.discretize_state(ctx.get_next_states())

            # Encode and log screen after color and action
            _, buffer = cv2.imencode('.png', state_image_origin)
            screen_base64 = base64.b64encode(buffer).decode('utf-8')
            #damage = ctx.next_self_health - ctx.self_health
            #timestamp_ms = int(time.time() * 1000)
            #csv_writer.writerow([timestamp_ms, action, 0, damage, screen_base64])  # Writing to CSV

            state_buffer.append((state, action, next_state))
            ctx = judge.action_judge(ctx, initial_reward)

            if ctx.emergence_break == 100:
                log("Emergency break activated.\n")
                agent.save_model()
                log("Model Saved.\n")
                ctx.paused = True
            cur_reward += ctx.reward
            
            if (target_step + 1) % N == 0:
                total_reward += cur_reward
                log("Store reward for all past actions: " + str(cur_reward/3))
                for state, action, next_state in state_buffer:
                    if ENABLE_TRAINING:
                        agent.store_data(state, action, cur_reward/N, next_state, ctx.done)
                state_buffer = []
                cur_reward = 0

            target_step += 1
            if target_step % training_config['update_step'] == 0:
                agent.update_target_network()

            state = next_state  # Update the state
            state_image_origin = next_state_image_origin

            # Control pause
            before_pause = ctx.paused
            ctx = actions.pause_game(ctx, ctx.emergence_break)
            if before_pause and not ctx.paused:
                ctx.emergence_break = 0

            if ctx.done == 1:
                break

        log(f"episode: {episode} Evaluation Average Reward: { total_reward / target_step}")
