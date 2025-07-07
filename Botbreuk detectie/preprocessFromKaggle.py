import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import random
import matplotlib.image as mpimg
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping , ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator , load_img
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils import class_weight
from tensorflow.keras.applications import DenseNet121
import albumentations as A
import tensorflow as tf
from albumentations.pytorch import ToTensorV2
from keras import backend as keras
from ultralytics import YOLO
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pathlib import Path
import pathlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import imgaug
import copy
train_images_path = './kaggle_boneFracture/BoneFractureYolo8/train/images'
train_labels_path = './kaggle_boneFracture/BoneFractureYolo8/train/labels'

test_images_path = './kaggle_boneFracture/BoneFractureYolo8/test/images'
test_labels_path = './kaggle_boneFracture/BoneFractureYolo8/test/labels'

val_images_path = './kaggle_boneFracture/BoneFractureYolo8/valid/images'
val_labels_path = './kaggle_boneFracture/BoneFractureYolo8/valid/labels'
# Get a list of all the image files in the training images directory
train_image_files_paths = os.listdir(train_images_path)

patch_width = 64
patch_height = 64

def get_sorted_files(image_dir, label_dir):

    train_image_path = Path(image_dir)
    train_label_path = Path(label_dir)
    
    train_image_files_paths = sorted(train_image_path.glob('*.jpg'))
    

    label_files_paths = sorted(train_label_path.glob('*.txt'))

    return train_image_files_paths, label_files_paths
random.seed(0)  # Ensures reproducibility
IMG_SIZE = 512
len_9 = 0
len_11 = 0
max_fractures = 0
max_non_fractures = 0
def calculate_offset(x_max, x_min, diff, left ):
    if  left:
        return x_max + (diff - x_min)
    else:
        return x_min - (diff - (IMG_SIZE - x_max))
max_x = IMG_SIZE - patch_width
max_y = IMG_SIZE - patch_height
def get_crop_coords(min_coord, max_coord, patch_size=64, img_size=IMG_SIZE):
    """
    Calculates the start and end coordinates for cropping a patch of size `patch_size`
    centered around the bounding box (min_coord, max_coord). Ensures coordinates stay within image bounds.
    """
    box_center = (min_coord + max_coord) / 2

    # Start and end positions based on desired patch size
    start = int(round(box_center - patch_size / 2))
    end = start + patch_size

    # Adjust if crop is out of bounds
    if start < 0:
        start = 0
        end = patch_size
    elif end > img_size:
        end = img_size
        start = img_size - patch_size

    return start, end


def perform_preprocessing(image_path, img_size=IMG_SIZE, iteration = 0):
    """
    Reads an image from the given path, resizes it to the specified size, and converts it to grayscale.
    """
    clip_limit = 3.5 if iteration == 0 else 0.5 + iteration
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))

    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # PREPROCESSING BLUR AND CONTRAST
    img = cv2.GaussianBlur(img, (3,3), 1.0) # i merkte dat de blur de ruis verminderde als heel grote contrast worden toegepast
    img = clahe.apply(img)


    return img

def get_image_and_label(label_path, image, dark=False):
    label = []


    with open(label_path, 'r') as file:
        label_content = file.readline().strip()
        label_content = label_content.split()
        
        # img = cv2.imread(image_path)
        # img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # # PREPROCESSING BLUR AND CONTRAST
        # img = cv2.GaussianBlur(img, (3,3), 0) # i merkte dat de blur de ruis verminderde als heel grote contrast worden toegepast
        # img = clahe.apply(img)
        bigger_than70 = []
        #TODO: maybe appl sobel            
        # Combine the two gradients
        if len(label_content) == 0:
            # print(f"Label '{len(label_content)}' due to Healthy bone.") but we change it iduring training because of the hirarchy in the directeory : fracture no_fracture
            label.append(0)  # class_id
            #en nu gaan we een random patch nemen die een grayscale gemiddeld van 0,5 heeft.
            h, w = image.shape
            patches = []
            #eerst slicen we de image in patchesvan 64
            for y in range(0, h-64 +1, 64):
                for x in range(0, w-64 +1, 64):
                    patch = image[y:y+64, x:x+64]
                    patches.append(patch)
            # vervolgens een random image met een grayscale gemiddelde van 70
            
            if dark: 
                patches = list(filter(lambda item: np.mean(item[1]) < 10, enumerate(patches)))
                if not patches:
                    print("No suitable patch found.")
                    return None, None, None
                chosen_patch = random.choice(patches)
                cropped_img = chosen_patch[1]
                return cropped_img, 0, (x, x+64, y, y+64)
            else:
                bigger_than70 = list(filter(lambda item: item[1].mean() > 70, enumerate(patches)))
            
                if not bigger_than70:
                    print("No suitable patch found.")
                    return None, None, None
                # Kies een willekeurige patch
                chosen_patch = random.choice(bigger_than70)
                cropped_img = chosen_patch[1]
                return cropped_img, 0, (x, x+64, y, y+64)
            
 
        if 9 <= len(label_content) <= 11:
            
            if label_content[0] == '2': # forearm fracture
                label.append(1)  # class_id where 0 = non fracture, 1 = fracture maar deze wordt 0 tijdens training

                x1, y1, x2, y2, x3, y3, x4, y4 = map(float, label_content[1:9])

                x_max = int(x1 * image.shape[1])
                y_min = int(y1 * image.shape[0])
                x_min = int(x3 * image.shape[1])
                y_max = int(y3 * image.shape[0])
                label.append(x_min)
                label.append(y_min)
                label.append(x_max)
                label.append(y_max)
                new_x_min, new_x_max = get_crop_coords(x_min, x_max)
                new_y_min, new_y_max = get_crop_coords(y_min, y_max)

                print(f"new_x_min: {new_x_min}, new_x_max: {new_x_max}, new_y_min: {new_y_min}, new_y_max: {new_y_max}")

                cropped_img = image[new_y_min:new_y_max, new_x_min:new_x_max]

                print(f"Image label: {label}")
                if(cropped_img.shape[0] == 0):
                    print("Image shape is 0")
                    print(f"Image label: {label}")
                    print(f"new_x_min: {new_x_min}, new_x_max: {new_x_max}, new_y_min: {new_y_min}, new_y_max: {new_y_max}")

                return cropped_img, 1,(new_x_min, new_x_max, new_y_min, new_y_max)
            else:
                return None, None, None
        # image[y1:y2, x1:x2]  # let op: (rows, cols)
        if len(label_content) > 11:
            #NB SOMMIGE LABELS 2 BOUNDING BOXDES
            return None, None, None
    return None, None, None
def apply_gabor_filters(image, ksize=31):
    filters = []
    responses = []
    for theta in np.arange(0, np.pi, np.pi /4):
        kernel = cv2.getGaborKernel(
            (ksize,ksize), 
            sigma=2.0,
            theta =theta, # orientatie van de filter
            lambd = 8.0, #wavelength  van de  sinusoidal
            gamma = 0.5, # spatial aspect ratio
            psi = 0, #phase offset
            ktype = cv2.CV_32F
            )
        filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
        responses.append(filtered)
        filters.append(kernel)
    return responses, filters
train_image_files_paths, train_label_files_paths = get_sorted_files(train_images_path, train_labels_path)
def save_cropped_image(image_array, save_path):
    tf.keras.utils.save_img(save_path, image_array, scale=True)

train_images = []
train_labels = []

max_rotated_imgs = 10
count = 0
def store_labels(image_files_paths, label_files_paths, save_path, subset_name = 'train', max = 3000, add_to_test = False):

    test_images = 0
    max_test = 30 # there are only 279 images in train fracture dataset, so we take 20 from fracture and other 20 from no fracture
   
    test_fracture_count = 0
    test_no_fracture_count = 0

    if add_to_test:
        # Store indices to remove after loop finishes
        indices_to_remove = []

        for idx, (image_file, label_file) in enumerate(zip(image_files_paths, label_files_paths)):
            if test_images >= max_test:
                print(f"Max {max_test} images reached for {subset_name} test set.")
                break

            image_path = str(image_file)
            label_path = str(label_file)
            
            
            tmp_image = perform_preprocessing(image_path, iteration=0)
            tmp_image, tmp_label, new_coordinates = get_image_and_label(label_path, tmp_image)

            if tmp_label is None:
                continue

            if tmp_label == 1 and test_fracture_count <= (max_test / 2):
                path = f'./processed_data/test/fracture/{image_file.name}'
                cv2.imwrite(path, tmp_image)
                indices_to_remove.append(idx)
                test_fracture_count += 1
                test_images += 1

            elif tmp_label == 0 and test_no_fracture_count < (max_test / 2):
                path = f'./processed_data/test/no_fracture/{image_file.name}'
                cv2.imwrite(path, tmp_image)
                indices_to_remove.append(idx)
                test_no_fracture_count += 1
                test_images += 1

        # Remove after the loop (reverse order to avoid index shifting)
        for i in sorted(indices_to_remove, reverse=True):
            del image_files_paths[i]
            del label_files_paths[i]
    max_fractures = 0
    print(f"Max fractures: {max_fractures}")
    #now there are 259 fracture images left for training
    length_image_files_paths = len(image_files_paths)
    print(f"Processing {length_image_files_paths} images for {subset_name} subset.")
    print(max_test)
    fracture_count = 0
    no_fracture_count = 0
    tmp_image = None

    for image_file, label_file in zip(image_files_paths, label_files_paths):
        print("entering training loop")
        image_path = str(image_file)    
        label_path = str(label_file)
        original_image = cv2.imread(image_path)
        original_image = cv2.resize(original_image, (IMG_SIZE, IMG_SIZE))
        if fracture_count + no_fracture_count >= max:
                    print(f"Max {max} images reached for {subset_name}.")
                    break
    
        if subset_name == 'train':
            
            tmp_image_ = perform_preprocessing(image_path, iteration=0)
            if no_fracture_count < 40:
                tmp_image, tmp_label, new_coordinates = get_image_and_label(label_path, tmp_image_, dark=True)
            else:
                tmp_image, tmp_label, new_coordinates = get_image_and_label(label_path, tmp_image_)
            
            if tmp_label is None:
                continue
            if tmp_label == 1 and fracture_count <= (max/2):

                tmp_image = np.array(tmp_image)
                tmp_img = copy.deepcopy(tmp_image)
                original_image = original_image[new_coordinates[2]:new_coordinates[3], new_coordinates[0]:new_coordinates[1]]
                path = f'./processed_data/{subset_name}/fracture/original_{image_file.name}'
                cv2.imwrite(path,original_image)
                fracture_count +=1

                for j in range(0,4):# let's try just with 3 rotations
                    if j > 0:
                        print(f"storing to fracture {subset_name} folder")
                        tmp_img = cv2.rotate(tmp_img, cv2.ROTATE_90_CLOCKWISE)
                        path = f'./processed_data/{subset_name}/fracture/clahe_{i}_{360-(j*90)}_degree_{image_file.name}'
                        cv2.imwrite(path, tmp_img)
                    else: 
                        path = f'./processed_data/{subset_name}/fracture/clahe_1_{image_file.name}'
                        cv2.imwrite(path, tmp_img)
                    fracture_count += 1

            if tmp_label == 0 and no_fracture_count < max/2:#is the number of agumented fracture images

                #* storing cropped original image
                if new_coordinates is not None:
                    original_image = original_image[new_coordinates[2]:new_coordinates[3], new_coordinates[0]:new_coordinates[1]]
                    cv2.imwrite(f'./processed_data/{subset_name}/no_fracture/original_{image_file.name}',original_image)
                    no_fracture_count += 1
                tmp_image = np.array(tmp_image)
                tmp_img = copy.deepcopy(tmp_image)
                for j in range(0,4):# let's try just with 3 rotations
                    if j > 0:
                        print(f"storing to fracture {subset_name} folder")
                        tmp_img = cv2.rotate(tmp_img, cv2.ROTATE_90_CLOCKWISE)
                        path = f'./processed_data/{subset_name}/no_fracture/clahe_{i}_{360-(j*90)}_degree_{image_file.name}'
                        cv2.imwrite(path, tmp_img)
                    else: 
                        path = f'./processed_data/{subset_name}/no_fracture/clahe_1_{image_file.name}'
                        cv2.imwrite(path, tmp_img)
                    no_fracture_count += 1
        if subset_name == 'valid':
            image_path = str(image_file)
            label_path = str(label_file)
            tmp_image = perform_preprocessing(image_path, iteration = 0)
            tmp_image, tmp_label = get_image_and_label(label_path, tmp_image)
            if tmp_label == 1 and fracture_count <= (max/2):
                path = f'./processed_data/{subset_name}/fracture/{image_file.name}'
                cv2.imwrite(path, tmp_image)
                fracture_count += 1
            elif tmp_label == 0 and no_fracture_count <= (max/2) :
                    # 262 non-fractured and 667 non-fracture quadrupled by rotation
                    no_fracture_count += 1
                    tmp_image = np.array(tmp_image)
                    path = f'./processed_data/{subset_name}/no_fracture/{image_file.name}'
                    cv2.imwrite(path, tmp_image)
            
    return True, fracture_count, no_fracture_count

## store validation data and test data
train_images_files, train_labels_files = get_sorted_files(train_images_path, train_labels_path )
train_img_stored, train_fractured, train_non_fractured = store_labels(train_images_files, train_labels_files, save_path='./processed_data/train', subset_name='train',  add_to_test=True)
print(f"Stored {train_fractured} fractured and {train_non_fractured} non-fractured images in training set.")
if train_img_stored:
    print("Training data stored successfully.")
val_max = int((train_fractured + train_non_fractured) * 0.15)
