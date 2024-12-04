from scripts.load_data import load_train, load_test, load_example
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
import numpy as np
# Laad de trainingsdata en labels
train_data, train_labels = load_train(padding=((0, 0), (0, 0), (20, 0)))
# De kleurwaarden in de afbeelding zijn nu 0 tot 255, we zetten deze om naar -0.5 tot 0.5
train_data = (train_data / 255) - 0.5


plt.imshow(train_data[2])
plt.title(f"{train_labels[2]}")
print(f"Label: {train_labels[2]}")
plt.show()
# Afvlakken van de afbeeldingen

train_labels = to_categorical(train_labels, 10)
train_data = train_data.reshape(train_data.shape[0], -1)
print(train_data.shape)
model = Sequential()

# input_dim moet gelijk zijn aan de lengte van 1 input
model.add(Dense(32, input_dim=(train_data.shape[1]))) 
model.add(Dense(10, activation="softmax"))

model.summary()
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data, train_labels, epochs=10)
test_data, test_labels = load_test(padding=((0, 0), (0, 0), (0, 20)))

test_data = (test_data/255) - 0.5

# --------------------- Omvorming van data ---------------
test_data =  test_data.reshape(test_data.shape[0], -1)
test_labels = to_categorical(test_labels, 10)
# --------------------- Evaluatie van de data ------------
result = model.evaluate(test_data, test_labels)

print(f"loss: {result[0]}, accuracy: {result[1]} van de 1.0")
(example_r, example_l), label = load_example(paddingL = ((0,0), (20,0)), paddingR=((0,0), (0, 20)))
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

axs[0].imshow(example_r)
axs[0].set_title("Padding on right side (Like training)")

axs[1].imshow(example_l)
axs[1].set_title("Padding on left side (Like testing)")

plt.show()
examples = np.array([example_r, example_l]) # FIXME
examples = examples.reshape(examples.shape[0], -1)
predicted = np.argmax(model.predict(examples), axis= -1 )
print("Predicted class for example with padding on right side:", predicted)