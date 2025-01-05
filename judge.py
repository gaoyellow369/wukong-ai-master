import time
from context import Context
from log import log
from tracker import RewardTracker

# Initialize the data tracker
reward_tracker = RewardTracker(train_data_dir='train_data/data')

def action_judge(ctx: Context, extra_penalty=0, episode=-1):
    log(f"Current state: Own health={ctx.self_health}, Boss's health={ctx.boss_health}, Own stamina={ctx.self_energy}, paused={ctx.paused}")
    log(f"Future state: Own health={ctx.next_self_health}, Boss's health={ctx.next_boss_health}, Own stamina={ctx.next_self_energy}, paused={ctx.paused}")
    if episode != -1:
        reward_tracker.episode_num = episode # For benchmark.py

    ctx.reward = extra_penalty
    current_time = time.time()

    # Todo If the time interval is too long, it will miss the death judgment
    # if current_time - ctx.last_reward_time < 0.1:
    #     log(f"Less than 0.1 seconds since the last reward calculation, skipping this reward calculation.")
    #     return ctx

    # Update the last reward calculation time
    ctx.last_reward_time = current_time

    reward_weight = (current_time - ctx.begin_time) / 5 # The longer the survival time, the higher the reward weight,
    log(f"reward_weight={reward_weight}")

    # In the case of own death
    # 1. Own health drops to 0
    # 2. Boss's health drops abnormally
    dead_condition1 = ctx.next_self_health <= 1
    dead_condition2 = ctx.boss_health - ctx.next_boss_health > 40
    dead_condition3 = False #current_time - ctx.begin_time < 10 and ctx.boss_health < 80  # usually this is due to boss health detection errors.

    if dead_condition2 or dead_condition1 or dead_condition3:
        if (dead_condition2 and not dead_condition1) or dead_condition3:
            log(f"Boss's health drops abnormally, judged as the puppet is dead")

        boss_remain_health = max(ctx.next_boss_health, ctx.boss_health)

        dead_reward = 50000
        boss_remain_health_award = (100 - boss_remain_health) * (dead_reward / 100)
        ctx.reward += boss_remain_health_award
        ctx.reward -= dead_reward
        ctx.done, ctx.stop, ctx.emergence_break = 1, 0, 100
        ctx.dodge_weight, ctx.attack_weight = 1, 1

        log(f"Wukong is dead, Current state: done={ctx.done}, stop={ctx.stop}, "
            f"emergence_break={ctx.emergence_break}, dodge_weight={ctx.dodge_weight}, "
            f"attack_weight={ctx.attack_weight}")
        
        reward_tracker.add_reward(ctx.reward)  # Add current reward
        reward_tracker.end_episode(boss_remain_health, int(current_time) - ctx.begin_time)  # Record Boss's health at the end of each game

        reward_tracker.save_overall_data()
        return ctx

    # Defeating the boss
    if ctx.self_health > 1 and ctx.next_boss_health < 1:

        boss_remain_health = ctx.boss_health
        ctx.reward += 50000
        ctx.done, ctx.stop, ctx.emergence_break = 1, 0, 100
        ctx.dodge_weight, ctx.attack_weight = 1, 1

        log(f"Boss is dead, Current state: done={ctx.done}, stop={ctx.stop}, "
            f"emergence_break={ctx.emergence_break}, dodge_weight={ctx.dodge_weight}, "
            f"attack_weight={ctx.attack_weight}")
        
        reward_tracker.add_reward(ctx.reward)  # Add current reward
        reward_tracker.end_episode(boss_remain_health, int(current_time) - ctx.begin_time)  # Record Boss's health at the end of each game

        reward_tracker.save_overall_data()
        return ctx


    # In the case of losing own health
    health_change = ctx.self_health - ctx.next_self_health
    if health_change > 5:
        ctx.reward -= 200 * health_change  * ctx.dodge_weight * reward_weight  # Losing health in each game deducts points to offset the points for attacking
        ctx.attack_weight = max(1, ctx.attack_weight - health_change)
        ctx.dodge_weight = min(1, ctx.dodge_weight - health_change)
        ctx.stop = 1  # To prevent repeated calculation in continuous frames
        log(f"Losing own health：{health_change}%. Reward reduction {10 * health_change}. Current weight: attack_weight={ctx.attack_weight}, "
            f"dodge_weight={ctx.dodge_weight}, stop={ctx.stop}")
    elif health_change < -5:
        # gain health
        ctx.reward += 200 * abs(health_change) # A drink of 30% health gain can get 6000 points
    else:
        ctx.stop = 0

    # In the case of the boss losing health
    health_change = ctx.boss_health - ctx.next_boss_health
    if health_change > 0 and health_change < 20:  # Damage can't be too high, too high means a calculation error
        add_award = 100 * health_change * ctx.attack_weight * reward_weight
        ctx.reward += add_award  # Encourage attacking the boss
        ctx.attack_weight = min(1, ctx.attack_weight + health_change)
        log(f"Boss loses health：{health_change}%. Reward increase {add_award}. Current attack_weight={ctx.attack_weight}")
    elif health_change > 5:
        log(f"Boss loses health {health_change}% too much")

    # In the case of energy consumption
    energy_change = ctx.self_energy - ctx.next_self_energy
    if energy_change > 5 and energy_change < 30:
        ctx.reward -= 2 * energy_change * ctx.dodge_weight
        ctx.dodge_weight = min(1, ctx.dodge_weight + energy_change / 10)
        log(f"Energy consumption：{energy_change}%. Reward reduction {1 * energy_change * ctx.dodge_weight}. Current dodge_weight={ctx.dodge_weight}")
    elif energy_change < 5:
        log(f"Energy consumption {energy_change}% ignored")
    else:
        log(f"Energy consumption {energy_change}% too high Abnormal")

    # Final reward calculation
    log(f"one action final reward: {ctx.reward}")
    
    # Add current reward data to tracker
    reward_tracker.add_reward(ctx.reward)
    
    return ctx
