import pandas as pd
import tensorflow as tf
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def process_data(csv_path, image_folder, output_file, image_size=(224, 224)):
    df = pd.read_csv(csv_path)

    # Verwerk labels naar numerieke waarden
    label_encoder = LabelEncoder()
    df['class'] = label_encoder.fit_transform(df['class'])

    # Voorbereiden van de lijsten voor beelden, labels en bounding boxes
    images = []
    labels = []
    bboxes = []

    for index, row in df.iterrows():
        # Lees de afbeelding
        img_path = os.path.join(image_folder, row['filename'])
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, image_size)

        # Normeer de bounding box coördinaten
        x_center = (row['xmin'] + row['xmax']) / 2 / row['width']
        y_center = (row['ymin'] + row['ymax']) / 2 / row['height']
        width = (row['xmax'] - row['xmin']) / row['width']
        height = (row['ymax'] - row['ymin']) / row['height']

        # Voeg de gegevens toe aan de lijsten
        images.append(image)
        labels.append(row['class'])
        bboxes.append([x_center, y_center, width, height])

    # Zet de lijsten om naar numpy arrays
    images = np.array([image.numpy() for image in images])
    labels = np.array(labels)
    bboxes = np.array(bboxes)

    # Sla de verwerkte gegevens op zodat ze later opnieuw gebruikt kunnen worden
    np.savez(output_file, images=images, labels=labels, bboxes=bboxes)
    print(f"Data saved to {output_file}")
process_data('./bone fracture detection.v4-v4.tensorflow/test/_annotations.csv','./bone fracture detection.v4-v4.tensorflow/test/' ,'test_data.npz')
process_data('./bone fracture detection.v4-v4.tensorflow/train/_annotations.csv',"./bone fracture detection.v4-v4.tensorflow/train/", 'train_data.npz')

