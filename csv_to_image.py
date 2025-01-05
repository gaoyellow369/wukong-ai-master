import os
import csv
import base64
import cv2
import numpy as np

csv.field_size_limit(2147483647)

def decode_and_save_image(base64_string, filename):
    img_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is not None:
        cv2.imwrite(filename, img)
    else:
        print(f"无法解码图像并保存到 {filename}")

image_folder = "images"
os.makedirs(image_folder, exist_ok=True)

with open('game_data.csv', 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  

    for i, row in enumerate(csv_reader):
        if row[0] == "TimeStamp":  
            continue
        
        timestamp, action, idx, damage, encoded_image = row
        action = int(action)

        image_filename = os.path.join(image_folder, f"image_{i}.png")
        decode_and_save_image(encoded_image, image_filename)

print("图像提取和保存完成")
