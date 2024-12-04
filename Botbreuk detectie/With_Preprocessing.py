import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import yaml
from sklearn import svm
from scipy import ndimage as ndi
from skimage import feature, data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
# Load class names from YAML data
with open('./kaggle_dataset/data.yaml', 'r') as f:
    data_structure = yaml.safe_load(f)

class_names = data_structure['names']
class_names.append('Normal')  # Assuming 'Normal' should be a separate class

# Laad de verwerkte trainingsdata
data = np.load('train_data_with_bbox.npz', allow_pickle=True)
train_image_tensors = np.array(data['images'])
train_label_tensors = np.array(data['labels'])  # Labels voor classificatie
train_bbox_tensors = np.array(data['bboxes'])  # Bounding boxes

# Controleer de dimensies
print(f"Train images shape: {train_image_tensors.shape}")
print(f"Train labels shape: {train_label_tensors.shape}")
print(f"Train bounding boxes shape: {train_bbox_tensors.shape}")

# Multi-output model
input_shape = train_image_tensors[0].shape

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
])

# Classificatie-uitgang (aantal klassen = len(class_names))
classification_output = Dense(len(class_names), activation='softmax', name='classification')(model.output)

# Bounding box-uitgang (4 coördinaten)
bbox_output = Dense(4, name='bbox')(model.output)

# Combineer de outputs in een multi-output model
multi_output_model = tf.keras.Model(inputs=model.input, outputs=[classification_output, bbox_output])

# Compileer het model
multi_output_model.compile(
    optimizer='adam',
    loss={
        'classification': 'categorical_crossentropy',  # Voor classificatie
        'bbox': 'mse'  # Voor bounding boxes
    },
    loss_weights={
        'classification': 1.0,  # Gewicht voor classificatie
        'bbox': 1.0  # Gewicht voor bounding boxes
    },
    metrics={
        'classification': ['accuracy'],  # Nauwkeurigheid voor classificatie
        'bbox': ['mae']  # Mean absolute error voor bounding boxes
    }
)

# Train het model
history = multi_output_model.fit(
    train_image_tensors,
    {'classification': train_label_tensors, 'bbox': train_bbox_tensors},
    epochs=10,
    batch_size=32
)

######### TEST ###########
# Laad de testdata
test_data = np.load('test_data_with_bbox.npz', allow_pickle=True)
test_image_tensors = np.array(test_data['images'])
test_label_tensors = np.array(test_data['labels'])  # Labels voor classificatie
test_bbox_tensors = np.array(test_data['bboxes'])  # Bounding boxes

# Evalueer het model
test_loss = multi_output_model.evaluate(
    test_image_tensors,
    {'classification': test_label_tensors, 'bbox': test_bbox_tensors},
    verbose=2
)

print(f"Test Losses: {test_loss[0]} | Test Accuracy: {test_loss[3] * 100} %")
