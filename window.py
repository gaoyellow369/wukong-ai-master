# window.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Window:
    def __init__(self, sx, sy, ex, ey):
        self.sx = sx
        self.sy = sy
        self.ex = ex
        self.ey = ey
        self.color = None  # Initialize as None

    def extract_region(self, frame):
        if frame is None:
            print("No frame received.")
            return None
        # When extracting the region, slightly expand the region to better detect the health bar edges
        safe_sy = max(0, self.sy - 2)
        safe_ey = max(0, self.ey + 2)
        safe_sx = max(0, self.sx - 2) 
        safe_ex = max(0, self.ex + 2)
        
        if safe_ey <= safe_sy or safe_ex <= safe_sx:
            print("Warning: Invalid slice indices. Returning original ROI.")
            return frame[self.sy:self.ey, self.sx:self.ex]
        
        return frame[safe_sy:safe_ey, safe_sx:safe_ex]

    def update(self, frame):
        self.color = self.extract_region(frame)

    def __repr__(self):
        return f"Window(sx={self.sx}, sy={self.sy}, ex={self.ex}, ey={self.ey})"


class GrayWindow(Window):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey)
        self.gray = None  # Initialize as None

    def update(self, frame):
        super().update(frame)
        if self.color is not None:
            self.gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = None


class HealthWindow(GrayWindow):
    def __init__(self, sx, sy, ex, ey, health_gray_min=90, health_gray_max=255, change_threshold=0.1):
        super().__init__(sx, sy, ex, ey)
        self.health_gray_min = health_gray_min
        self.health_gray_max = health_gray_max
        self.health_percentage = 0  # Initialize health percentage to 0
        self.previous_health_percentage = -1  # Used to store the previous health percentage
        self.health_history = []  # Cache for storing past health percentage data
        self.change_threshold = change_threshold  # Threshold for detecting health bar changes

    def update(self, frame):
        super().update(frame)
        # Only calculate health_percentage if the health bar is present
        current_health_percentage = self.calculate_value_V4()

        # Fill the history list if it has less than 10 entries
        if len(self.health_history) < 3:
            self.health_history.append(current_health_percentage)
        else:
            # Keep only the latest 10 entries
            self.health_history.pop(0)
            self.health_history.append(current_health_percentage)

        # Use mode to filter outliers
        values, counts = np.unique(np.array(self.health_history), return_counts=True)
        most_common_health = values[np.argmax(counts)]
        
        if most_common_health == -1 and current_health_percentage != -1:
            self.health_history = [current_health_percentage]
            most_common_health = current_health_percentage

        if self.previous_health_percentage == -1:
            self.health_percentage = most_common_health
        elif abs(most_common_health - self.previous_health_percentage) < self.change_threshold:
            self.health_percentage = self.previous_health_percentage
        else:
            self.health_percentage = most_common_health

        # Update previous_health_percentage
        self.previous_health_percentage = self.health_percentage
        return

    def resize_gray(self):
        ignore_edge_len = 2
        min_boundary_position = ignore_edge_len  # Avoid the boundary at the far left, set minimum boundary
        max_boundary_position = len(self.gray[0]) - ignore_edge_len  # Avoid boundary at far right, set maximum boundary
        gray = self.gray[:, min_boundary_position:max_boundary_position]  # Process only the valid region
        return gray

    def pick_middle(self, gray):
        height = gray.shape[0]

        # Start from 3 rows above the center line and extend to 3 rows below
        middle_line = height // 2
        middle_start = max(0, middle_line - 3)  # Ensure no out-of-bounds errors
        middle_end = min(height, middle_line + 4)  # Include 3 rows below, +4 to ensure the middle line and 3 below are covered

        middle_gray = gray[middle_start:middle_end, :]  # Middle region
        return middle_gray

    def calculate_value_V4(self):
        # Convert the image to grayscale and process
        gray = self.gray
        resized_gray = self.resize_gray()
        picked_middle = self.pick_middle(resized_gray)
        num_rows = picked_middle.shape[0]
        num_cols = picked_middle.shape[1]

        # Calculate the row-wise continuity ratio
        row_continuity = np.mean([np.sum(np.diff(row) == 0) / (len(row) - 1) for row in picked_middle])

        # Calculate the column-wise continuity ratio
        col_continuity = np.mean([np.sum(np.diff(col) == 0) / (len(col) - 1) for col in picked_middle.T])

        # Calculate the total continuity ratio
        total_continuity = (row_continuity * num_rows + col_continuity * num_cols) / (num_rows + num_cols)

        # Define the grayscale range for the health bar
        health_gray_min = self.health_gray_min
        health_gray_max = self.health_gray_max

        # Calculate the proportion of white pixels (assuming pixels with grayscale value 255 are white)
        white_pixel_count = np.sum(picked_middle == health_gray_max)
        total_pixel_count = picked_middle.size
        white_proportion = white_pixel_count / total_pixel_count

        # Use np.clip to limit the pixel values to the health bar range [health_gray_min, health_gray_max]
        clipped = np.clip(picked_middle, health_gray_min, health_gray_max)

        # Calculate the number of pixels where the clipped grayscale value matches the original
        count = np.count_nonzero(clipped == picked_middle)

        # Calculate the total number of pixels in picked_middle
        total_length = picked_middle.size

        # Calculate the current health bar percentage
        current_health_percentage = round((count / total_length) * 100, 2)
        bar_exist = False
        if row_continuity > 0.3 and current_health_percentage > 0.1:
            bar_exist = True

        final_result = current_health_percentage
        if not bar_exist:
            final_result = -1

        should_log = False

        if should_log:
            # Log output
            log_message = (
                f"Row continuity: {row_continuity:.4f}\n"
                f"Column continuity: {col_continuity:.4f}\n"
                f"Total continuity: {total_continuity:.4f}\n"
                f"White pixel proportion: {white_proportion:.4f} (Represents health bar level)\n"
                f"Bar exist: {bar_exist}\n"
                f"Current health percentage: {current_health_percentage}%\n"
            )
            print(log_message)
            
            # Adjust the layout to reduce the size of the three vertical images on the left and maintain the size of the bar chart on the right
            fig, axes = plt.subplots(3, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [1.5, 1]})

            # Original grayscale image
            axes[0, 0].imshow(gray, cmap='gray')
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')

            # resized_gray
            axes[1, 0].imshow(resized_gray, cmap='gray')
            axes[1, 0].set_title('Resized Gray')
            axes[1, 0].axis('off')

            # picked_middle
            axes[2, 0].imshow(picked_middle, cmap='gray')
            axes[2, 0].set_title('Picked Middle (Blood Region)')
            axes[2, 0].axis('off')

            # Display row and column continuity ratios and the health bar in a bar chart (right)
            bar_data = [row_continuity, col_continuity, total_continuity, white_proportion]
            bar_labels = ['Row Continuity', 'Column Continuity', 'Total Continuity', 'White Proportion (Blood)']
            axes[0, 1].bar(bar_labels, bar_data, color=['blue', 'green', 'orange', 'red'])
            axes[0, 1].set_title('Continuity & Blood Level')
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].set_ylabel('Proportion')

            # Hide unused subplots
            axes[1, 1].axis('off')
            axes[2, 1].axis('off')

            # Optimize layout
            plt.tight_layout()
            plt.show()

        return final_result

    def health_count(self) -> int:
        return self.health_percentage


class EnergyWindow(HealthWindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey, health_gray_min=135, health_gray_max=165)

class MagicWindow(HealthWindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey, health_gray_min=70, health_gray_max=120)

class SkillWindow(HealthWindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey)
        self.update_n = 3
        self.update_counter = self.update_n # only update health count after 3 successful updates
        self.health_window = [0] * self.update_n
    def update(self, frame):
        super().update(frame)
        if self.gray is not None:
            middle_row = self.gray[self.gray.shape[0] // 2, :]
            clipped = np.clip(middle_row, self.health_gray_min, self.health_gray_max)
            count = np.count_nonzero(clipped == middle_row)
            total_length = len(middle_row)
            self.health_window[self.update_counter - 1] = (count / total_length) * 100
            self.update_counter -= 1
            if self.update_counter == 0:
                self.update_counter = self.update_n
    
    def health_count(self) -> int:
        return np.min(self.health_window)

# Init windows
X_offset = 640
# self_health_window = HealthWindow(778-X_offset, 684, 1025-X_offset, 697)
self_health_window = HealthWindow(778-X_offset, 683, 1025-X_offset, 696)
skill_1_window = SkillWindow(1754-X_offset, 601, 1759-X_offset, 611)
skill_2_window = SkillWindow(1786-X_offset, 601, 1797-X_offset, 611)
self_energy_window = EnergyWindow(780-X_offset, 709, 995-X_offset, 713)
self_magic_window = MagicWindow(780-X_offset, 701, 956-X_offset, 705)
boss_health_window = HealthWindow(1152-X_offset, 640, 1418-X_offset, 645)
main_screen_window = Window(1050-X_offset, 100, 1600-X_offset, 700)


#self_energy_window = EnergyWindow(186,987,311,995)
self_energy_window = EnergyWindow(186, 956, 339, 963)


# 提供访问函数
def get_self_health_window():
    return self_health_window

def get_skill_1_window():
    return skill_1_window

def get_skill_2_window():
    return skill_2_window

def get_self_energy_window():
    return self_energy_window

def get_self_magic_window():
    return self_magic_window

def get_boss_health_window():
    return boss_health_window

def get_main_screen_window():
    return main_screen_window
