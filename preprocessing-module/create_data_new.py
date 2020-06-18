import os
import os.path
import pickle

import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from PIL import Image, ImageDraw

RAW_IMAGES_FOLDER = "../images/raw_images"
SCALED_IMAGES_FOLDER = "../images/scaled_images"


def scale_images(raw_images_path, scaled_images_path):
    # Loop through each raw image
    n = 0;
    for img_path in image_files_in_folder(raw_images_path):
        print(f"Processing image - {img_path}")
        image = face_recognition.load_image_file(img_path)
        face_bounding_boxes = face_recognition.face_locations(image)
        print(f"{len(face_bounding_boxes)} faces found")
        for top, right, bottom, left in face_bounding_boxes:
            n += 1
            face_image = image[top:bottom, left:right]
            face_image = Image.fromarray(face_image)
            face_image = face_image.resize((150, 150), Image.ANTIALIAS)
            face_image.save(f"{scaled_images_path}/img{n:05}.png", "PNG")


scale_images(RAW_IMAGES_FOLDER, SCALED_IMAGES_FOLDER)