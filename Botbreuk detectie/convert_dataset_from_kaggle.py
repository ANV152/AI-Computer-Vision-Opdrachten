import os
import pathlib
import numpy as np
import tensorflow as tf
import yaml
import warnings

warnings.simplefilter('ignore')

# Load YAML data
with open('./kaggle_dataset/data.yaml', 'r') as f:
    data_structure = yaml.safe_load(f)
    print(data_structure)

class_names = data_structure['names']
class_names.append('Normal')  # Assuming 'Normal' should be a separate class

train_image_path = pathlib.Path('./kaggle_dataset/train/images/')
train_label_path = pathlib.Path('./kaggle_dataset/train/labels/')

test_image_path = pathlib.Path('./kaggle_dataset/test/images/')
test_label_path = pathlib.Path('./kaggle_dataset/test/labels/')
def make_image_label(image_path, label_path, size=(224, 224)):
    if type(image_path) != pathlib.PosixPath:
        image_path = pathlib.Path(image_path)
        label_path = pathlib.Path(label_path)

    image_tensors = []
    label_tensors = []
    bbox_tensors = []
    n = 0
    is_blank = False
    for i in image_path.glob('*'):
        print(f"Processing image: {n}")
        n += 1
        is_blank = False
        file = i.stem  # Correct way to get the filename without extension
        label_correspond = file + '.txt'
        label_file_path = label_path / label_correspond

        if not label_file_path.exists():
            print(f"Label file {label_file_path} not found, skipping.")
            continue
        # Read and preprocess the label
        with open(label_file_path, 'r') as label_file:
            label_content = label_file.read().strip()
            if label_content == '':
                # Empty label file implies background
                is_blank = True
            else:
                parts = label_content.split(' ')
                label = parts[0]
                bbox = list(map(float, parts[1:5]))  # Assuming bbox format: [x_center, y_center, width, height]
                one_hot_label = [0 for _ in range(len(class_names))]
                one_hot_label[int(label)] = 1
            if not is_blank:
                label_tensors.append(one_hot_label)
                bbox_tensors.append(bbox)
        # Read and preprocess the image
        if not is_blank:
            image_file = tf.io.read_file(str(i))
            image_tensor = tf.io.decode_image(image_file)
            image_tensor = tf.image.resize(image_tensor, size)
            image_tensors.append(image_tensor.numpy())  # Convert to NumPy array
            print(f"Image tensor shape: {image_tensor.shape}, Label tensor: {one_hot_label}, BBox: {bbox}")

    print(f"Processed {n} images.")
    return image_tensors, label_tensors, bbox_tensors

# Use the function
train_image_tensors, train_label_tensors, train_bbox_tensors = make_image_label(train_image_path, train_label_path)

# Check if image and label tensors are correctly loaded
print("Number of images loaded:", len(train_image_tensors))
print("Number of labels loaded:", len(train_label_tensors))
print("Number of bounding boxes loaded:", len(train_bbox_tensors))

# Save the processed data
np.savez_compressed('train_data_with_bbox.npz', images=train_image_tensors, labels=train_label_tensors, bboxes=train_bbox_tensors)
# Process the test data
test_image_tensors, test_label_tensors, test_bbox_tensors = make_image_label(test_image_path, test_label_path)
print("Number of test images loaded:", len(test_image_tensors))

# Save the test data
np.savez_compressed('test_data_with_bbox.npz', images=test_image_tensors, labels=test_label_tensors, bboxes=test_bbox_tensors)