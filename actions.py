# coding=utf-8
import grabscreen
import keys
from getkeys import is_key_pressed
import window
import restart

import time
import threading
from context import Context
from log import log

keys = keys.Keys()


# Create an action executor class for managing the asynchronous execution of actions
class ActionExecutor:
    def __init__(self):
        # Used to stop the action execution thread
        self.running = True
        # Action queue
        self.action_queue = []
        # Create a thread lock to ensure thread safety
        self.lock = threading.Lock()
        # Start the action execution thread
        self.thread = threading.Thread(target=self._execute_actions)
        self.thread.daemon = True  # Set as a daemon thread
        self.thread.start()

    def add_action(self, action_func, *args, delay=0, **kwargs):
        # Add the action and its execution time to the queue
        execute_time = time.time() + delay
        with self.lock:
            # log('excute_time %d' % execute_time)
            # log(f'{args}')
            self.action_queue.append((execute_time, action_func, args, kwargs))

    def _execute_actions(self):
        while self.running:
            current_time = time.time()
            with self.lock:
                queue_copy = self.action_queue.copy()
            for item in queue_copy:
                execute_time, action_func, args, kwargs = item
                if current_time >= execute_time:
                    try:
                        action_func(*args, **kwargs)
                    except Exception as e:
                        log(f"Action execution exception: {e}")
                    with self.lock:
                        if item in self.action_queue:
                            self.action_queue.remove(item)
            # To prevent high CPU usage
            time.sleep(0.001)

    def stop(self):
        self.running = False
        self.thread.join()

    def wait(self):
        while True:
            with self.lock:
                if not self.action_queue:
                    break
            time.sleep(0.1)  # Adjust sleep time as needed


# Instantiate the action executor
action_executor = ActionExecutor()

def wait():
    action_executor.wait()

# High precision sleep
# TODO: However, for turning the camera, it still cannot guarantee the consistency of the offset angle each time, the error here still needs to be resolved
def precise_sleep(target_duration):
    start_time = time.perf_counter()
    while True:
        current_time = time.perf_counter()
        elapsed_time = current_time - start_time
        if elapsed_time >= target_duration:
            break
        time.sleep(max(0, target_duration - elapsed_time))


def precise_sleep_non_blocking(target_duration):
    # This function does not need to perform any operations in the current implementation
    pass


def pause(second=0.04):
    action_executor.add_action(keys.directKey, "T")
    action_executor.add_action(keys.directKey, "T", keys.key_release, delay=second)


def mouse_move(times):
    for i in range(times):
        action_executor.add_action(keys.directMouse, i, 0)
        # Add delay
        time_per_move = 0.004
        action_executor.add_action(lambda: None, delay=time_per_move)


def mouse_move_v2(times):
    for i in range(times):
        action_executor.add_action(keys.directMouse, i, 0)
        # Add delay
        time_per_move = 0.01
        action_executor.add_action(lambda: None, delay=time_per_move)


def lock_view(second=0.2):
    action_executor.add_action(keys.directMouse, buttons=keys.mouse_mb_press)
    action_executor.add_action(
        keys.directMouse, buttons=keys.mouse_mb_release, delay=second
    )


def run_with_direct(second=0.5, direct="W"):
    action_executor.add_action(keys.directKey, direct)
    action_executor.add_action(keys.directKey, "LSHIFT")
    action_executor.add_action(keys.directKey, direct, keys.key_release, delay=second)
    action_executor.add_action(keys.directKey, "LSHIFT", keys.key_release, delay=second)
    
def run_with_direct_v2(second=0.5, directs=["W"]):
    # press all direct keys at the same time
    for direct in directs:
        print("Pressing %s" % direct)
        action_executor.add_action(keys.directKey, direct)
    
    # press LSHIFT
    action_executor.add_action(keys.directKey, "LSHIFT")
    
    # wait for "second" time
    precise_sleep(second)
    
    # release all keys
    for direct in directs:
        action_executor.add_action(keys.directKey, direct, keys.key_release)
    
    # release LSHIFT
    action_executor.add_action(keys.directKey, "LSHIFT", keys.key_release)


def eat(second=0.04):
    action_executor.add_action(keys.directKey, "R")
    action_executor.add_action(keys.directKey, "R", keys.key_release, delay=second)


def recover(delay=1.5):
    log("Taking medicine")
    eat()
    action_executor.add_action(keys.directKey, "D")
    action_executor.add_action(keys.directKey, "D", keys.key_release, delay=delay)


def attack(second=0.2):
    action_executor.add_action(keys.directMouse, buttons=keys.mouse_lb_press)
    action_executor.add_action(
        keys.directMouse, buttons=keys.mouse_lb_release, delay=second
    )
    log("Attacking")


def heavy_attack(second=0.2):
    action_executor.add_action(keys.directMouse, buttons=keys.mouse_rb_press)
    action_executor.add_action(
        keys.directMouse, buttons=keys.mouse_rb_release, delay=second
    )
    log("Heavy attack")


def jump(second=0.04):
    action_executor.add_action(keys.directKey, "LCTRL")
    action_executor.add_action(keys.directKey, "LCTRL", keys.key_release, delay=second)


# Dodge
def dodge(second=0.04):
    action_executor.add_action(keys.directKey, "SPACE")
    action_executor.add_action(keys.directKey, "SPACE", keys.key_release, delay=second)
    log("Dodging")


def go_forward(second=0.04):
    action_executor.add_action(keys.directKey, "W")
    action_executor.add_action(keys.directKey, "W", keys.key_release, delay=second)
    log("Moving forward")
    
def go_forward_v2(second=0.04):
    action_executor.add_action(keys.directKey, "W")
    precise_sleep(second)
    action_executor.add_action(keys.directKey, "W", keys.key_release)
    log("Moving forward")


def go_back(second=0.04):
    action_executor.add_action(keys.directKey, "S")
    action_executor.add_action(keys.directKey, "S", keys.key_release, delay=second)
    
def go_back_v2(second=0.04):
    action_executor.add_action(keys.directKey, "S")
    precise_sleep(second)
    action_executor.add_action(keys.directKey, "S", keys.key_release)


def go_left(second=0.04):
    action_executor.add_action(keys.directKey, "A")
    action_executor.add_action(keys.directKey, "A", keys.key_release, delay=second)
    
def go_left_v2(second=0.04):
    action_executor.add_action(keys.directKey, "A")
    precise_sleep(second)
    action_executor.add_action(keys.directKey, "A", keys.key_release)


def go_right(second=0.04):
    log("Moving right")
    action_executor.add_action(keys.directKey, "D")
    action_executor.add_action(keys.directKey, "D", keys.key_release, delay=second)
    
def go_right_v2(second=0.04):
    log("Moving right")
    action_executor.add_action(keys.directKey, "D")
    precise_sleep(second)
    action_executor.add_action(keys.directKey, "D", keys.key_release)


def press_E(second=0.04):
    action_executor.add_action(keys.directKey, "E")
    action_executor.add_action(keys.directKey, "E", keys.key_release, delay=second)


def press_ESC(second=0.04):
    action_executor.add_action(keys.directKey, "ESC")
    action_executor.add_action(keys.directKey, "ESC", keys.key_release, delay=second)


def press_Enter(second=0.04):
    action_executor.add_action(keys.directKey, "ENTER")
    action_executor.add_action(keys.directKey, "ENTER", keys.key_release, delay=second)
    log("Pressed Enter")


def press_up(second=0.04):
    action_executor.add_action(keys.directKey, "UP")
    action_executor.add_action(keys.directKey, "UP", keys.key_release, delay=second)
    log("Pressed Up arrow")


def press_down(second=0.04):
    action_executor.add_action(keys.directKey, "DOWN")
    action_executor.add_action(keys.directKey, "DOWN", keys.key_release, delay=second)
    log("Pressed Down arrow")


def press_left(second=0.04):
    action_executor.add_action(keys.directKey, "LEFT")
    action_executor.add_action(keys.directKey, "LEFT", keys.key_release, delay=second)
    log("Pressed Left arrow")


def press_right(second=0.04):
    action_executor.add_action(keys.directKey, "RIGHT")
    action_executor.add_action(keys.directKey, "RIGHT", keys.key_release, delay=second)
    log("Pressed Right arrow")


def use_skill(skill_key="1", second=0.04):
    action_executor.add_action(keys.directKey, skill_key)
    action_executor.add_action(
        keys.directKey, skill_key, keys.key_release, delay=second
    )
    log("Used skill {}".format(skill_key))


def pause_game(ctx: Context, emergence_break=0):

    # Check if the "T" key is pressed
    if is_key_pressed('T'):
        # Toggle pause state
        ctx.paused = not ctx.paused
        if ctx.paused:
            log("Game paused. Press T to resume.")
            precise_sleep(1)
        else:
            ctx.begin_time = int(time.time())
            log("Game resumed.")
            precise_sleep(1)

    # If the game is paused
    if ctx.paused:
        time.sleep(0.5) # To prevent high CPU usage and rapid key press detection
        
        log("Game is currently paused. Move to boss, and press T to training")
        if emergence_break == 100:
            restart.restart()

        # Continuously check for key presses in pause state
        while ctx.paused:
            if is_key_pressed('T'):
                ctx.paused = False
                ctx.begin_time = int(time.time())
                log("Game resumed.")
                precise_sleep(1)
            time.sleep(0.01)  # To prevent high CPU usage
    return ctx


def is_skill_1_in_cd():
    return window.get_skill_1_window().health_count() < 50


def take_action(action, ctx: Context) -> Context:
    base_sleep = 0.1
    sleep = 0
    if action == 0:  # No operation, wait
        # run_with_direct(0.5, 'W')
        ctx.dodge_weight = 1
        sleep = 0.25

    elif action == 1:  # Attack
        attack()
        ctx.dodge_weight = 1
        sleep = 0.25

    elif action == 2:  # Dodge
        #if ctx.magic_num > 10:
        dodge()
        ctx.dodge_weight += 10  # To prevent converging to a state of constant dodging
        sleep = 0.25

    elif action == 3:  # Use skill 1
        log("skill_1 cd :%d" % is_skill_1_in_cd())
        if not is_skill_1_in_cd():
            use_skill("1")
            ctx.dodge_weight = 1
            sleep = 0.25

    elif action == 4:  # Recover
        delay = 2
        recover(delay)
        ctx.medicine_nums -= 1
        ctx.dodge_weight = 1
        sleep = delay

    elif action == 5:  # Heavy attack
        heavy_attack()
        ctx.dodge_weight = 1
        sleep = 2

    time.sleep(max(sleep - base_sleep, 0))
    # There's a delay in health bar update, need 5 frames to update
    for i in range(5):
        window.get_self_health_window().update(grabscreen.grab_screen())
        time.sleep(0.02)
    return ctx


# Ensure to stop the action executor thread when the program ends
def stop_action_executor():
    action_executor.stop()

