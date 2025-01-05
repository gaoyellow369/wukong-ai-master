import math
import random
def calculate_rewards(prev_state, new_state, attack_weight=1.0, dodge_weight=1.0, log=False):
    """
    Calculate the reward based on the changes between the previous and new states.

    Parameters:
    - prev_state (dict): The state before the action was taken.
    - new_state (dict): The state after the action was taken.
    - attack_weight (float): Weight factor for attacking rewards.
    - dodge_weight (float): Weight factor for dodge penalties.
    - log (bool): Flag to enable or disable logging.

    Returns:
    - float: The calculated reward.
    """
    reward = 0

    # Check if the episode has ended
    player_dead = new_state['player_health'] <= 0
    boss_dead = new_state['boss_health'] <= 0

    if boss_dead and not player_dead:
        reward += 5000
        if log:
            print("Boss is dead. Reward += 10000.")
        return reward
    elif player_dead and not boss_dead:
        reward -= 5000
        if log:
            print("Player is dead. Reward -= 10000.")
        return reward
    elif boss_dead and player_dead:
        # Define behavior when both die; assuming penalty
        reward -= 3000
        if log:
            print("Both player and boss are dead. Reward -= 5000.")
        return reward

    # Calculate health changes
    player_health_change = prev_state['player_health'] - new_state['player_health']
    if player_health_change > 1:
        # Penalty for losing health
        penalty = 200 * player_health_change * dodge_weight
        reward -= penalty
        if log:
            print(f"Losing own health: {player_health_change}. Penalty -= {penalty}.")
    elif player_health_change < -1:
        prev_health_percentage = prev_state['player_health'] / 100 * 100
        alpha = 0.02197  # Controls steepness of the exponential decay
        recovery_weight = math.exp(-alpha * prev_health_percentage)
        # Reward for gaining health
        health_recovered = abs(player_health_change)
        reward += recovery_weight * 400 * health_recovered
        if log:
            print(f"Gaining health: {health_recovered}. Reward += {recovery_weight * 400 * health_recovered}.")

    # Calculate boss health changes
    boss_health_change = prev_state['boss_health'] - new_state['boss_health']
    if 0 < boss_health_change < 20:
        # Reward for damaging boss
        attack_reward = 100 * boss_health_change * attack_weight
        reward += attack_reward
        if log:
            print(f"Boss loses health: {boss_health_change}. Reward += {attack_reward}.")
    elif boss_health_change >= 50:
        if log:
            print(f"Boss loses health {boss_health_change} too much.")

    # Calculate medicine usage
    medicine_used = prev_state['medicine_count'] - new_state['medicine_count']
    if medicine_used > 0:
        penalty = 3000 * medicine_used
        reward -= penalty
        if log:
            print(f"Medicine used: {medicine_used}. Penalty -= {penalty}.")

    if log:
        print(f"Calculated reward: {reward}")
    
    return reward


class CombatEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        # Initialize state variables
        self.boss_state = 0  # 0: non-attack, 1: attack, 2: freeze
        self.freeze_skill_cd = 0  # 1 means on cooldown, 0 means ready
        self.consecutive_attack_count = 0  # Consecutive successful attacks
        self.medicine_count = 5  # Medicines available
        self.player_health = 100  # Player's health
        self.boss_health = 100  # Boss's health
        self.boss_freeze_timer = 0  # Remaining freeze duration for boss
        self.freeze_skill_cd_timer = 0  # Remaining cooldown time for freeze skill
        self.done = False  # Whether the episode is done
        self.time = 0  # Time counter in seconds

        # Initialize flags
        self.player_successfully_dodged = False

        # Return the initial state
        return self._convert_state_to_tuple(self._get_state())

    def step(self, action):
        if self.done:
            raise Exception("Episode is done. Please reset the environment.")

        # Save previous state
        prev_state = self._get_state()

        # Simulate environment
        self.simulate_env(action)

        # Get new state
        new_state = self._get_state()

        # Calculate reward
        reward = calculate_rewards(prev_state, new_state)

        # Check for end of episode
        if self.player_health <= 0 or self.boss_health <= 0:
            self.done = True

        # Prepare next state
        next_state = self._get_state()

        return self._convert_state_to_tuple(new_state), reward, self.done, {}

    def simulate_env(self, action):
        # Time management
        delta_time = 1  # Time elapsed in this step (in seconds)
        self.time += delta_time

        # Update timers
        if self.boss_freeze_timer > 0:
            self.boss_freeze_timer -= delta_time
            if self.boss_freeze_timer <= 0:
                self.boss_freeze_timer = 0  # Reset freeze timer
                self.boss_state = 0  # Boss returns to non-attack state

        if self.freeze_skill_cd_timer > 0:
            self.freeze_skill_cd_timer -= delta_time
            if self.freeze_skill_cd_timer <= 0:
                self.freeze_skill_cd_timer = 0  # Reset cooldown timer
                self.freeze_skill_cd = 0  # Freeze skill is ready

        # Reset flags
        self.player_successfully_dodged = False

        # Process player's action
        if action == 0:
            # Do nothing
            pass
        elif action == 1:
            # Regular attack
            hit_chance = 0.4
            if self.boss_state == 2:
                hit_chance = 1
            elif self.boss_state == 1:
                hit_chance = 0.05

            if random.random() < hit_chance:
                self.consecutive_attack_count += 1
                # Damage increases with combo
                base_damage = random.uniform(2.0, 7.5)
                combo_multiplier = 0.1  # 10% increase per combo count
                damage = base_damage * (1 + combo_multiplier * self.consecutive_attack_count)
                self.boss_health -= damage
            else:
                # Missed attack
                self.consecutive_attack_count = 0
        elif action == 2:
            # Dodge
            if random.random() < 0.7:  # 70% chance to avoid attack
                self.player_successfully_dodged = True
        elif action == 3:
            # Use freeze skill
            if self.freeze_skill_cd == 0:
                self.freeze_skill_cd = 1
                self.freeze_skill_cd_timer = 30  # Cooldown for 30 seconds
                if self.boss_state == 1:
                    self.boss_state = 2  # Boss is frozen
                    self.boss_freeze_timer = 5  # Freeze for 5 seconds
                # Using freeze skill when boss is not attacking has no effect
            else:
                # Freeze skill is on cooldown; no effect
                pass
        elif action == 4:
            # Eat medicine
            if self.medicine_count > 0:
                if self.boss_state == 1:
                    pass
                elif self.player_health < 100:
                    self._heal_player()
                else:
                    # Health is full; consuming medicine has no effect on health
                    self.medicine_count -= 1  # Medicine is consumed
            else:
                # No medicine left; action has no effect
                pass
        else:
            # Invalid action; no state change
            pass

        # Boss attack (if boss is attacking and not frozen)
        if self.boss_state != 2:  # If boss is not frozen
            boss_damage_chance = 0.2
            if self.boss_state == 1:
                boss_damage_chance = 0.9
            else:
                boss_damage_chance = 0.2

            if random.random() < boss_damage_chance:
                if not self.player_successfully_dodged:
                    # Player gets hit
                    self.player_health -= random.uniform(5.0, 15.0)

        # Boss can change its state randomly if not frozen
        if self.boss_state != 2:  # If boss is not frozen
            self._update_boss_state()

    def _update_boss_state(self):
        # Boss can randomly change its state to attack or non-attack if not frozen
        self.boss_state = random.choice([0, 1])  # 0: non-attack, 1: attack

    def _heal_player(self):
        # Heal the player by 33% of max health
        heal_amount = 33
        self.player_health = min(100, self.player_health + heal_amount)
        self.medicine_count -= 1

    def _get_state(self):
        # Return the current state as a dictionary
        return {
            'boss_state': self.boss_state,               # Boss state: 0, 1, or 2
            'freeze_skill_cd': self.freeze_skill_cd,     # Freeze skill cooldown: 0 or 1
            'consecutive_attack_count': self.consecutive_attack_count, # Consecutive attack count， 0 to 5
            'medicine_count': self.medicine_count,       # Medicines left, 0 to 10
            'player_health': self.player_health,         # Player's health, 0 to 100
            'boss_health': self.boss_health,             # Boss's health, 0 to 100
        }
    
    def _convert_state_to_tuple(self, state):
        return (state['boss_state'], state['freeze_skill_cd'], state['consecutive_attack_count'], 
                state['medicine_count'], state['player_health'], state['boss_health'])
