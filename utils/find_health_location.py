# coding=utf-8
from ast import main
import cv2
import time
from models.resnet import ResnetClassification
import window
from log import log
import grabscreen

def main():
    wait_time = 1
    update_content = True  # Add a flag variable to control whether to update the content

    while wait_time > 0:
        log(f'{wait_time} seconds to start execution')
        wait_time -= 1
        time.sleep(1)

    last_time = time.time()
    
    # Initialize display windows
    main_screen_window = None
    self_health_window = None
    magic_health_window = None
    boss_health_window = None
    classifier = ResnetClassification()
    classifier.load_from_file('model_weight/resnet_model.pth')

    while True:
        if update_content:

            # Grab the screen frame once
            frame = grabscreen.grab_screen()

            # Update all window objects
            window.get_boss_health_window().update(frame)
            window.get_self_health_window().update(frame)
            window.get_self_energy_window().update(frame)
            window.get_self_magic_window().update(frame)
            window.get_main_screen_window().update(frame)
            window.get_skill_1_window().update(frame)


            main_screen_window = window.get_main_screen_window()

            self_health_window = window.get_self_health_window()
            magic_health_window = window.get_self_magic_window()
            self_skill_1_window = window.get_skill_1_window()

            boss_health_window = window.get_boss_health_window()

            self_health = self_health_window.health_count()
            boss_health = boss_health_window.health_count()
            magic_health = magic_health_window.health_count()
            self_skill_1 = self_skill_1_window.health_count()
            state_image = window.get_main_screen_window().color[:, :, :3]
            boss_state = classifier.classify(state_image)
            log('self_health: %d, boss_health:%d, skill colldown:%d, boss sate:%d' % (self_health, boss_health, self_skill_1, boss_state))

        # Display the current screen, whether or not the content is updated
        if main_screen_window:
            #cv2.imshow('main_screen', main_screen_window.color)
            #cv2.imshow('self_screen_gray', self_health_window.gray)
            #cv2.imshow('boss_screen_gray', boss_health_window.gray)
            #cv2.imshow('magic_health_gray', magic_health_window.gray)
            #cv2.imshow('energy_window', window.get_self_energy_window().gray)
            cv2.imshow('self_skill_1', self_skill_1_window.gray)


        

        # Time consumption
        # log('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            update_content = not update_content  # Toggle the content update status

    cv2.waitKey()  # Press any key to exit after video ends
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
