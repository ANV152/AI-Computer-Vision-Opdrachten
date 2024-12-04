from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt
def test_model(index = 0): # Kies een index (bijv. de eerste afbeelding)
    test_image = test_images[index]  # Vorm: (224, 224, 3)
    true_label = test_labels[index]
    true_bbox = test_bboxes[index]

    #een batch van één afbeelding (model verwacht batches)
    test_image_batch = np.expand_dims(test_image, axis=0)  # Vorm: (1, 224, 224, 3)

    #afbeelding door het model
    predicted_class, predicted_bbox = model.predict(test_image_batch)

    # het classificatie-output
    predicted_class_id = np.argmax(predicted_class[0])  # Hoogste kans
    predicted_class_prob = np.max(predicted_class[0])  # Waarschijnlijkheid

    #  afbeelding en bounding box
    plt.imshow(test_image.astype('uint8'))  # Zet de afbeelding om naar integer pixelwaarden
    plt.title(f"Predicted: Class {predicted_class_id} ({predicted_class_prob:.2f}),\n"
            f"True: Class {true_label}")
    plt.xlabel(f"Predicted BBox: {predicted_bbox[0]}\nTrue BBox: {true_bbox}")
    plt.gca().add_patch(plt.Rectangle(
        (predicted_bbox[0][0], predicted_bbox[0][1]),  # (x, y) linkerbovenhoek
        predicted_bbox[0][2],  # breedte
        predicted_bbox[0][3],  # hoogte
        edgecolor='red', facecolor='none', linewidth=2, label='Predicted BBox'
    ))
    plt.legend()
    plt.show()
# Laad de verwerkte gegevens
train_data = np.load('train_data.npz', allow_pickle=True)
train_images = train_data['images']
train_labels = train_data['labels']
train_bboxes = train_data['bboxes']

test_data = np.load('test_data.npz', allow_pickle=True)
test_images = test_data['images']
test_labels = test_data['labels']
test_bboxes = test_data['bboxes']

input_img = Input(shape=(224, 224, 3))

# Bouw het model
x = Conv2D(32, (3, 3), activation='relu')(input_img)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)

# Uitgang voor classificatie
class_output = Dense(len(np.unique(train_labels)), activation='softmax', name='class_output')(x)

# Uitgang voor regressie van bounding boxes
bbox_output = Dense(4, name='bbox_output')(x)

# Definieer het model
model = Model(inputs=input_img, outputs=[class_output, bbox_output])

# Compileer het model met meerdere loss-functies
model.compile(optimizer='adam',
              loss={'class_output': 'sparse_categorical_crossentropy', 'bbox_output': 'mse'},
              metrics={'class_output': 'accuracy', 'bbox_output': 'mse'})

model.fit(train_images, {'class_output': train_labels, 'bbox_output': train_bboxes}, epochs=10, batch_size=32)
test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Total loss: ",test_acc[0], " | Accuracy: ", test_acc[3] ) 

####### Test #######
test_model(0)