# coding=utf-8
import time

import actions
import grabscreen
from log import log
from getkeys import is_key_pressed
import window

# Pathfinding at Heifeng the Great King's location
def heifeng_restart():
    # actions.pause()
    log("Dead, auto-pathfinding training begins")

    # Wait to respawn at the Land Temple first
    wait = 30
    for i in range(wait, 0, -1):
        log('%d seconds to respawn' % i)
        actions.precise_sleep(1)
    
    log('Moving left')
    actions.run_with_direct(0.7, 'A')
    actions.precise_sleep(1) # Must wait a while, otherwise the key press won't respond
    log('Moving forward')
    actions.run_with_direct(9, 'W')

    actions.lock_view()
    actions.pause()

# Tiger Vanguard figure-eight pathfinding, more stable
def huxianfeng_restart_v1():
    log("Dead, auto-pathfinding training begins")

    # Wait to respawn at the Land Temple first
    wait = 25
    for i in range(wait, 0, -1):
        if is_key_pressed('B'):
            return
        log('%d seconds to respawn' % i)
        actions.precise_sleep(1)
    
    # Use the Land Temple to lock the initial view
    actions.press_E()
    actions.precise_sleep(3)
    actions.press_ESC()
    actions.precise_sleep(3)

    actions.go_back(2.5)
    actions.precise_sleep(1) # Must wait a while, otherwise the key press won't respond

    actions.go_right(2)
    actions.precise_sleep(1)

    times = 4
    while times > 0:
        times -= 1
        actions.run_with_direct(2, 'W')
        actions.precise_sleep(1)

        if times == 0:
            actions.mouse_move(35)
            actions.lock_view()
            actions.precise_sleep(1)
            break

        actions.go_right(3)
        actions.precise_sleep(1)
    actions.run_with_direct(2, 'W')
    actions.pause()

# Tiger Vanguard changing view angle pathfinding, affected by mouse view angle precision, unstable
def huxianfeng_restart_v2():
    log("Dead, auto-pathfinding training begins")

    # Wait to respawn at the Land Temple first
    wait = 25
    for i in range(wait, 0, -1):
        if is_key_pressed('B'):
            return
        log('%d seconds to respawn' % i)
        actions.precise_sleep(1)
    
    # Use the Land Temple to lock the initial view
    actions.press_E()
    actions.precise_sleep(4)
    actions.press_ESC()
    actions.precise_sleep(3)

    actions.go_back(2.5)
    actions.precise_sleep(1) # Must wait a while, otherwise the key press won't respond

    actions.go_right(1.6)
    actions.precise_sleep(1)

    actions.mouse_move(35)
    actions.precise_sleep(1)

    actions.run_with_direct(11, 'W')
    actions.precise_sleep(1)

    actions.lock_view()
    actions.pause()
    actions.run_with_direct(1, 'W')

def huxianfeng_restart_v3():
    log("Dead, auto-pathfinding training begins")

    # Wait for the restart animation to finish 
    # and for the character to the revival point(the local temple)

    wait = 25
    for i in range(wait, 0, -1):
        if is_key_pressed('B'):
            return
        log('Revive after %ds' % i)
        actions.precise_sleep(1)
    
    # Use the local temple to lock the initial perspective
    actions.press_E()
    actions.precise_sleep(3)
    actions.press_ESC()
    actions.precise_sleep(3)

    # Move to the target point
    actions.go_back_v2(2.5)
    actions.go_right_v2(1.5)
    actions.precise_sleep(0.5)
    
    
    times = 2
    while times > 0:
        log(times)
        times -= 1
        actions.run_with_direct_v2(4.5, ['W','D'])
        actions.precise_sleep(0.1)
        actions.run_with_direct_v2(0.5, ['W'])
        if times == 0:
            actions.mouse_move(35)
            actions.lock_view()
            break
    actions.run_with_direct_v2(1, 'W')
    actions.pause()
    actions.run_with_direct_v2(1.5, 'W')


def yinhu_restart():
    log("Dead, auto-pathfinding training begins")

    # Wait to respawn at the Land Temple first
    wait = 18
    for i in range(wait, 0, -1):
        if is_key_pressed('B'):
            return
        log('%d seconds to respawn' % i)
        actions.precise_sleep(1)

    actions.go_forward(1.5)
    actions.precise_sleep(3) # Must wait a while, otherwise the key press won't respond

    actions.press_E()
    actions.precise_sleep(5)

    # The menu will stay on the last selected item (if the mouse doesn't move)
    # actions.press_down()
    # actions.precise_sleep(1)
    # actions.press_down()
    # actions.precise_sleep(1)

    actions.dodge()
    actions.precise_sleep(5)

    log('Hold E to skip the animation!')
    actions.press_E(3)
    actions.precise_sleep(10)
    
    log('ready!')
    # Auto lock view is on
    #actions.lock_view()
    actions.run_with_direct(2, 'W')
    actions.pause()


def yinhu_restart_V2():
    log("Dead, auto-pathfinding training begins")

    log('Hold E to skip the animation!') 
    actions.press_E(3)

    # Wait to respawn at the Land Temple first
    wait = 8
    for i in range(wait, 0, -1):
        if is_key_pressed('B'):
            return
        log('%d seconds to respawn' % i)
        actions.precise_sleep(1)

    actions.go_forward(1.5)
    actions.precise_sleep(3) # Must wait a while, otherwise the key press won't respond

    # Interaction
    actions.press_E(0.5)
    actions.precise_sleep(6)

    # If the cursor isn't on the challenge button after the last click, move it down to the challenge
    actions.press_down()
    actions.precise_sleep(0.2)
    actions.press_down()
    actions.precise_sleep(0.2)
    actions.dodge()
    actions.precise_sleep(2)

    log('Hold E to skip the animation!') 
    actions.press_E(3)
    actions.precise_sleep(9)
    
    log('ready!')
    actions.run_with_direct(2, 'W')
    actions.pause()

def teleport(delay_seconds = 18):
    actions.press_ESC()
    actions.precise_sleep(1)
    actions.press_ESC()
    actions.precise_sleep(1)
    actions.press_ESC()
    actions.precise_sleep(1)
    actions.go_right()
    actions.precise_sleep(1)
    actions.press_E()
    actions.precise_sleep(1)
    actions.press_E()
    actions.precise_sleep(18)

    actions.mouse_move(35)
    log("teleport finished")

import time

last_boss_fight_time = 0

def wait_for_boss_health_bar(timeout_seconds = 10):
    while timeout_seconds > 0:
        frame = grabscreen.grab_screen()
        window.get_boss_health_window().update(frame)
        if window.get_boss_health_window().health_count() > 0:
            log("Boss health bar detected")
            break
        log("boss health bar not detected, waiting...")
        timeout_seconds -= 1
        actions.precise_sleep(1)


def datou_restart():
    global last_boss_fight_time
    log("Dead, auto-pathfinding training begins")
    current_time = time.time()
    wait = 25
    if current_time - last_boss_fight_time < 10:
        teleport()
        last_boss_fight_time = 0
        wait = 1
    
    # Wait to respawn at the Land Temple first
    for i in range(wait, 0, -1):
        if is_key_pressed('B'):
            return
        log('%d seconds to respawn' % i)
        actions.precise_sleep(1)

    # teleport()
    # actions.wait()
    # return
    # Use the Land Temple to lock the initial view
    actions.press_E()
    actions.precise_sleep(4)
    actions.press_ESC()
    actions.precise_sleep(3)


    actions.run_with_direct(4, 'W')
    actions.precise_sleep(5)
    # actions.lock_view()
    # actions.precise_sleep(0.5)
    # actions.lock_view()

    actions.run_with_direct(4, 'W')
    actions.precise_sleep(4.5)
    actions.lock_view()

    actions.run_with_direct(3, 'W')
    actions.precise_sleep(4)
    actions.lock_view()

    actions.run_with_direct(4.5, 'W')
    actions.precise_sleep(5)
    actions.go_right(0.8)
    actions.precise_sleep(0.8)
    actions.lock_view()
    actions.precise_sleep(0.3)
    actions.lock_view()

    actions.go_right(0.8)
    actions.precise_sleep(0.8)
    actions.lock_view()
    actions.precise_sleep(0.3)
    actions.lock_view()

    actions.go_right(0.8)
    actions.precise_sleep(0.8)
    actions.lock_view()

    actions.run_with_direct(5.5, 'W')
    actions.precise_sleep(5.5)

    actions.pause()
    last_boss_fight_time = time.time()
    log("boss fight start:", last_boss_fight_time)
    #actions.lock_view()
    #actions.run_with_direct(1, 'W')
    # actions.wait()
    #log("finished")

def datou_restart_v2():
    global last_boss_fight_time
    log("Dead, auto-pathfinding training begins")
    current_time = time.time()
    wait = 25
    if current_time - last_boss_fight_time < 10:
        teleport()
        last_boss_fight_time = 0
        wait = 1
    
    # Wait to respawn at the Land Temple first
    for i in range(wait, 0, -1):
        if is_key_pressed('B'):
            return
        log('%d seconds to respawn' % i)
        actions.precise_sleep(1)

    # Use the Land Temple to lock the initial view
    actions.press_E()
    # Into the temple
    actions.precise_sleep(4)
    # Out of the temple
    actions.press_ESC()
    actions.precise_sleep(3)


    actions.run_with_direct_v2(5, ['W'])
    actions.run_with_direct_v2(4.5, ['W'])
    actions.lock_view()

    actions.run_with_direct_v2(3, ['W'])
    actions.lock_view()

    actions.run_with_direct_v2(4.5, ['W'])
    actions.go_right_v2(0.8)
    actions.lock_view()
    actions.precise_sleep(0.3)
    actions.lock_view()

    actions.go_right_v2(0.8)
    actions.lock_view()
    actions.precise_sleep(0.3)
    actions.lock_view()

    actions.go_right_v2(0.8)
    actions.lock_view()

    actions.run_with_direct_v2(5.5, ['W'])

    wait_for_boss_health_bar()
    actions.pause()
    last_boss_fight_time = time.time()
    log("boss fight start:", last_boss_fight_time)

def restart():
    datou_restart()
    # datou_restart_v2()

if __name__ == "__main__":  
    restart()