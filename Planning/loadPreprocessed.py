import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.layers import LeakyReLU
from keras import backend as K
import matplotlib.pyplot as plt
import random as rd
import numpy as np
img_height = 64
img_width = 64
batch_size = 32
rd.seed(0)
tf.random.set_seed(0)
def load_preprocessed_data(dir):
    dataset = tf.keras.utils.image_dataset_from_directory(
        './processed_data/' + dir + '/',
        labels='inferred',
        label_mode='binary',
        batch_size=batch_size,
        color_mode='grayscale',
        image_size=(img_height,img_width),
        shuffle=True
    )    
    return dataset
def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label
train_subset = load_preprocessed_data('train')
valid_subset = load_preprocessed_data('valid')


print("train subset size: ", len(train_subset))
train_dataset = train_subset.map(normalize_img)
validation_dataset = valid_subset.map(normalize_img)

print("train labels: ", train_dataset.take(1).as_numpy_iterator().next()[1][0])

#*checking the content of the train dataset

#*end of checking

print("Train dataset ready")
n_epochs  = 25
amount_of_filters = 25 # 10, 20, 30, 40, 50, 60, 70, 80, 90, 100

#* Building the model
dropout = 0.4
def create_cnn_model(input_shape=(64,64,1), num_classes=1):

    model = models.Sequential([
        # layers.Conv2D(15 , (5, 5), input_shape=input_shape, padding='same'),# 50 because it gives more features for curves and
        # layers.GaussianNoise(0.1, input_shape=input_shape),
        layers.Conv2D(32 , (5, 5), input_shape=input_shape,activation = 'relu'),# 50 because it gives more features for curves and
        layers.Dropout(0.2),
        layers.MaxPooling2D((2, 2)),#output size :(30,30,20)
        
        layers.GaussianNoise(0.1),
        layers.Conv2D(64,(3,3), activation='relu'),
        layers.Dropout(0.2),
        layers.MaxPooling2D((2,2), ),#output: (14,14,30),
        
        layers.GaussianNoise(0.1),    
        layers.Conv2D(128,(3,3), activation='relu'),
        layers.Dropout(0.2),
        layers.MaxPooling2D((2,2)),#output: (6,6,25)

        layers.GaussianNoise(0.1),
        layers.Conv2D(128, (3,3), activation='relu'),# 32 because it gives more features for curves and edges
        # layers.LeakyReLU(alpha=0.1),
        layers.Dropout(0.2),
        layers.MaxPooling2D((2,2)),# output size :(2,2,128)
        
        # layers.GaussianNoise(0.2),
        # layers.Conv2D(num_classes, (2,2), activation='sigmoid'),# 32 because it gives more features for curves and edges        

        layers.Flatten(),# output size 512
        # layers.GaussianNoise(0.1),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),

        # layers.GaussianNoise(0.1),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),

        # layers.GaussianNoise(0.1),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(num_classes, activation='sigmoid')


    ])
    return model

model = create_cnn_model()
sample_images, _ = next(iter(train_subset.take(1)))
layer_output = [layer.output for layer in model.layers if 'conv' in layer.name]
activation_model = Model(inputs=model.input, outputs=layer_output)
activations = activation_model.predict(sample_images)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_crossentropy', 'accuracy'])
model.summary()

#train
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=100,
)
#* changing learning rate
# default learning rate is 0.001, but we can change it to 0.0001
# K.set_value(model.optimizer.learning_rate, 0.0001)
# history = model.fit(
#     train_dataset,
#     validation_data=validation_dataset,
#     epochs=50,
# )

# Plot
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.plot(history.history['binary_crossentropy'], label='train binary_crossentropy')
plt.plot(history.history['val_binary_crossentropy'], label='val_binary_crossentropy')
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("accuracy")
plt.show()
""" Epoch 10/10
18/18 [==============================] - 2s 106ms/step - loss: 0.6928 - accuracy: 0.5090 - val_loss: 0.6925 - val_accuracy: 0.5385
validation loss is too high for this model, so we are overfitting. This means that we have to augment the data or use a more complex model.

Epoch 50/50
18/18 [==============================] - 2s 98ms/step - loss: 0.5098 - accuracy: 0.7491 - val_loss: 0.5793 - val_accuracy: 0.7051
loss still very high but accuracy is improving in both training and validation sets and overfitting is reduced a little bit

Epoch 70/70
18/18 [==============================] - 2s 109ms/step - loss: 0.6929 - accuracy: 0.5376 - val_loss: 0.6928 - val_accuracy: 0.5385

Epoch 70/70
18/18 [==============================] - 2s 105ms/step - loss: 0.3859 - accuracy: 0.8100 - val_loss: 0.6122 - val_accuracy: 0.7692
Still overfitting and I found a limit around 75-80% train accuracy when training
"""
model.save(f'my_model_{str(n_epochs)}_epochs_{amount_of_filters}_{dropout}')


