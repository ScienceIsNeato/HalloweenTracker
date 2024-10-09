import cv2
import numpy as np
import subprocess
import sys

def play_audio():
    # Replace this with the actual implementation of audio playback
    # For example, using pygame or playsound
    # subprocess.Popen(["python", "play_audio.py"])
    print("Playing audio clip...")

def handle_exit():
    # Terminate program when user presses ESC
    ch = cv2.waitKey(1) & 0xFF
    if ch == 27:
        return True
    return False

def display_debug_windows(frame2, diff, thresh, morphed, dilated):
    # Helper function to convert images to BGR
    def to_bgr(img):
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            return img

    # Helper function to resize images while maintaining aspect ratio
    def resize_with_aspect_ratio(image, max_width=None, max_height=None):
        (h, w) = image.shape[:2]
        if max_width is not None and w > max_width:
            scale = max_width / w
            new_w = int(w * scale)
            new_h = int(h * scale)
        elif max_height is not None and h > max_height:
            scale = max_height / h
            new_w = int(w * scale)
            new_h = int(h * scale)
        else:
            new_w, new_h = w, h
        return cv2.resize(image, (new_w, new_h))

    # Helper function to create a label bar above each image
    def create_label_bar(text, width):
        label_height = 40  # Height of the label bar
        label_bar = np.full((label_height, width, 3), border_color, dtype=np.uint8)  # Grey background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5  # Increased font size for larger images
        font_thickness = 3
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (label_height + text_size[1]) // 2
        cv2.putText(label_bar, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)
        return label_bar

    # Set the border color to grey
    border_color = [127, 127, 127]  # Grey color
    border_size = 10  # Increased border size for better visibility

    # Convert images to BGR
    frame2_bgr = to_bgr(frame2)
    diff_bgr = to_bgr(diff)
    thresh_bgr = to_bgr(thresh)
    morphed_bgr = to_bgr(morphed)
    dilated_bgr = to_bgr(dilated)

    # Resize images while keeping aspect ratio
    max_width = 960  # Increased size (3X larger)
    frame2_bgr = resize_with_aspect_ratio(frame2_bgr, max_width=max_width)
    diff_bgr = resize_with_aspect_ratio(diff_bgr, max_width=max_width)
    thresh_bgr = resize_with_aspect_ratio(thresh_bgr, max_width=max_width)
    morphed_bgr = resize_with_aspect_ratio(morphed_bgr, max_width=max_width)
    dilated_bgr = resize_with_aspect_ratio(dilated_bgr, max_width=max_width)

    # Create label bars
    frame2_label = create_label_bar('Frame', frame2_bgr.shape[1])
    diff_label = create_label_bar('Difference', diff_bgr.shape[1])
    thresh_label = create_label_bar('Threshold', thresh_bgr.shape[1])
    morphed_label = create_label_bar('Morphed', morphed_bgr.shape[1])
    dilated_label = create_label_bar('Dilated', dilated_bgr.shape[1])

    # Create horizontal border between label and image
    label_image_border = np.full((border_size, frame2_bgr.shape[1], 3), border_color, dtype=np.uint8)

    # Stack label bars above images with horizontal border
    frame2_with_label = np.vstack((frame2_label, label_image_border, frame2_bgr))
    diff_with_label = np.vstack((diff_label, label_image_border, diff_bgr))
    thresh_with_label = np.vstack((thresh_label, label_image_border, thresh_bgr))
    morphed_with_label = np.vstack((morphed_label, label_image_border, morphed_bgr))
    dilated_with_label = np.vstack((dilated_label, label_image_border, dilated_bgr))

    # Add border around each image
    images = [frame2_with_label, diff_with_label, thresh_with_label, morphed_with_label, dilated_with_label]
    images_with_borders = []
    for img in images:
        img_with_border = cv2.copyMakeBorder(
            img,
            border_size,  # Top
            border_size,  # Bottom
            border_size,  # Left
            border_size,  # Right
            cv2.BORDER_CONSTANT,
            value=border_color,
        )
        images_with_borders.append(img_with_border)

    images = images_with_borders

    # Find maximum image dimensions
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)

    # Pad images to have the same dimensions
    for i in range(len(images)):
        img = images[i]
        h, w = img.shape[:2]
        top = (max_height - h) // 2
        bottom = max_height - h - top
        left = (max_width - w) // 2
        right = max_width - w - left
        images[i] = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_color
        )

    # Create an empty image for the missing sixth slot
    empty_image = np.full((max_height, max_width, 3), border_color, dtype=np.uint8)

    # Arrange images in a grid without additional borders (since each image already has borders)
    top_row = np.hstack((images[0], images[1], images[2]))
    bottom_row = np.hstack((images[3], images[4], empty_image))

    # Create horizontal border between top and bottom rows
    horizontal_border_between_rows = np.full(
        (border_size, top_row.shape[1], 3), border_color, dtype=np.uint8
    )

    # Stack the rows vertically with a horizontal border
    combined_image = np.vstack((top_row, horizontal_border_between_rows, bottom_row))

    # Display the combined image
    cv2.imshow('Debug Window', combined_image)
