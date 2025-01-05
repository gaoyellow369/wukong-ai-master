import actions
from log import log
import window
import grabscreen
import time
from timing_decorator import timeit
MAX_CONECUTIVE = 5

class Context(object):
    def __init__(self):
        self.boss_health = 100
        self.next_boss_health = 100
        self.self_health = 100
        self.next_self_health = 100
        self.self_energy = 100
        self.next_self_energy = 100
        self.skill_1_cd = 0
        self.next_skill_1_cd = 0
        self.medicine_nums = 10
        self.next_medicine_nums = 10
        self.combo_attack = 0
        self.next_combo_attack = 0
        self.boss_state = 0
        self.next_boss_state = 0

        self.magic_num = 100
        self.next_magic_num = 100
        self.stop = 0
        self.emergence_break = 0
        self.dodge_weight = 1
        self.attack_weight = 1
        self.reward = 0
        self.done = 0
        self.paused = True
        self.begin_time = int(time.time())
        self.last_reward_time = int(time.time())

    def refresh_screens(self):
        # Grab the screen frame all at once
        frame = grabscreen.grab_screen()

        # Update all window objects
        window.get_boss_health_window().update(frame)
        window.get_self_health_window().update(frame)
        window.get_self_energy_window().update(frame)
        window.get_self_magic_window().update(frame)
        window.get_main_screen_window().update(frame)
        window.get_skill_1_window().update(frame)
    
    #@timeit
    def updateContext(self, action):
        self.boss_health = self.next_boss_health
        self.self_health = self.next_self_health
        self.self_energy = self.next_self_energy
        self.skill_1_cd = self.next_skill_1_cd
        self.medicine_nums = self.next_medicine_nums
        self.combo_attack = self.next_combo_attack

        self.refresh_screens()

        # Update the values for the next frame
        self.next_boss_health = window.get_boss_health_window().health_count()
        self.next_self_health = window.get_self_health_window().health_count()
        self.next_self_energy = window.get_self_energy_window().health_count()
        self.next_magic_num = window.get_self_magic_window().health_count()
        self.next_skill_1_cd = actions.is_skill_1_in_cd()
        
        damage_to_boss = self.boss_health - self.next_boss_health
        if damage_to_boss > 0:
            self.next_combo_attack += 1
            if self.next_combo_attack > MAX_CONECUTIVE:
                self.next_combo_attack = 0
        elif action != 1: # reset counter if casting skill or other actions
            self.next_combo_attack = 0
        if action == 4:
            self.next_medicine_nums -= 1
            if self.next_medicine_nums < 0:
                self.next_medicine_nums = 0


        health_diff = self.next_self_health - self.self_health
        if health_diff > 0:
            log("Health diff: " + str(health_diff))
        else:
            log("Normal")
        return self

    def update_boss_state(self, new_boss_state):
        self.boss_state = self.next_boss_state
        self.next_boss_state = new_boss_state


    def get_states(self):
        # Normalize to 0~1
        return [
            self.boss_state,
            int(self.skill_1_cd),
            self.combo_attack,
            self.medicine_nums,
            self.self_health,
            self.boss_health
        ]
    
    def get_next_states(self):
        return [
            self.next_boss_state,
            int(self.next_skill_1_cd),
            self.next_combo_attack,
            self.next_medicine_nums,
            self.next_self_health,
            self.next_boss_health
        ]
