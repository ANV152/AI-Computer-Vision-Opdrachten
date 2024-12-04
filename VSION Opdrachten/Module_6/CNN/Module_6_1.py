
from scripts.load_data import load_train, load_test, load_example
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import PIL
print(PIL.__version__)
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.layers import Flatten
train_data, train_labels = load_train()
# De kleurwaarden in de afbeelding zijn nu 0 tot 255, we zetten deze om naar -0.5 tot 0.5
train_data = (train_data / 255) - 0.5


plt.imshow(train_data[2])
plt.title(f"{train_labels[2]}")
print(f"Label: {train_labels[2]}")
plt.show()
num_samples, height_train, width_train = train_data.shape
train_labels = to_categorical(train_labels, 10)
print(train_data.shape)
train_data = train_data.reshape(num_samples, height_train * width_train)# Zet mij in de goede vorm

print(train_data.shape)
model = Sequential()
# input_dim moet gelijk zijn aan de lengte van 1 input
# model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(32, input_dim = train_data.shape[1])) # FIXME
model.add(Dense(10, activation="softmax"))

model.summary()
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=10)
print("hi")


# test_data, test_labels = load_test()

# test_data = test_data/255.0 - 0.5

# nbr_spl, height_test, width_test = test_data.shape

# test_data = test_data.reshape(nbr_spl, height_test * width_test) # FIXME¿
# test_labels = to_categorical(test_labels, 10)
# result = model.evaluate(test_data, test_labels)

# result = model.evaluate(test_data, test_labels)

# print(f"loss: {result[0]}, accuracy: {result[1]} van de 1.0")

(example_r, example_l), label = load_example(index= 5)
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

axs[0].imshow(example_r)
axs[0].set_title("Padding on right side (Like training)")

axs[1].imshow(example_l)
axs[1].set_title("Padding on left side (Like testing)")

plt.show()
examples = np.array([example_r, example_l]) # FIXME
examples = examples.reshape(examples.shape[0], examples.shape[1]*examples.shape[2])
predicted_probabilities = model.predict(examples)
predicted_classes = np.argmax(predicted_probabilities, axis=-1)
print(predicted_classes)
print("hi")