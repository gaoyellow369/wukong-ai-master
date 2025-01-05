import csv
import cv2
import base64
import numpy as np
import sys

# Action descriptions based on the provided script
action_descriptions = {
    0: 'No action, wait',
    1: 'Attack',
    2: 'Dodge',
    3: 'Use Skill 1',
    4: 'Recover',
    5: 'Heavy Strike'
}


def decode_image(base64_string):
    """Decode a base64 string to an image."""
    img_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

def main():
    # Set the maximum field size limit to a reasonable large value
    max_int_c_long = 2147483647  # This is the maximum for a 32-bit int
    try:
        csv.field_size_limit(max_int_c_long)
    except OverflowError:
        print("The limit provided is too large!")
        return

    # Open the CSV file
    with open('game_data.csv', 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header

        for row in csv_reader:
            timestamp, action, idx, damage, encoded_image = row
            if action == "Action":
                continue
            action = int(action)  # Convert action to integer
            image = decode_image(encoded_image)

            # Display the image
            print("damage:", damage)
            cv2.imshow(action_descriptions.get(action, 'Unknown Action') + ' index' + str(idx), image)
            key = cv2.waitKey(0)  # Wait for a key press to move to the next image

            # Close all windows before moving to the next image
            cv2.destroyAllWindows()

            # Break the loop if 'q' is pressed
            if key == ord('q'):
                break

if __name__ == '__main__':
    main()
