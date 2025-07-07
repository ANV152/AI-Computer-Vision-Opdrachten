import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import random
# from loadPreprocessed import create_cnn_model
from MavTools_NN import ViewTools_NN
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
import numpy as np
test_labels_count = 0
#* select image from 0 to 5. this images are  taken from the test subdirectory in kaggle_boneFracture not from the training set. these last mentioned images will be added afterwards
i_test_image = 4

random.seed(0)

#*belangrijk: 0 is fracture en 1 is non fracture
def parse_label_file(label_path):
    with open(label_path,'r') as f:
        content = f.read().strip()
        if not content:
            return None, None
        parts = content.split()
        print("Parts:", parts)
        category = int(parts[0])
        print("Category:", category)
        if category == 2 :
            coords = list(map(float, parts[1:9]))
            return category, coords
        else:
            return category, None

def load_test_images(images_dir, labels_dir, target_category=2, limit_per_class=None):
    fractured_images = []
    healthy_images = []
    for filename in os.listdir(images_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        base_name = os.path.splitext(filename)[0]
        image_path = os.path.join(images_dir, filename)
        label_path = os.path.join(labels_dir, base_name + '.txt')

        if not os.path.exists(label_path):
            healthy_images.append((image_path, None))
            continue
        category, coords = parse_label_file(label_path)
        if category == target_category:
            print("Fractured image:", image_path, coords)
            fractured_images.append((image_path, coords))
        else:
            healthy_images.append((image_path, coords))
    
    n = min(len(fractured_images), len(healthy_images))
    if limit_per_class:
        print(n)
        n = round(min(n, limit_per_class) / 2)
        print("Limiting to", n, "per class")
    random.shuffle(fractured_images)
    random.shuffle(healthy_images)
    # print(len(fractured_images))
    print(fractured_images[3][1])
    image = cv.imread(fractured_images[0][0], cv.IMREAD_GRAYSCALE)
    print("Image shape:", image.shape)
    
    return fractured_images[:n], healthy_images[:n]
def get_crop_coords(min_coord, max_coord, patch_size=64, img_size=512):
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
clahe = cv.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
def get_sliced(images):
    img_slices = []
    img_labels = []
    print("images: ", len(images))
    # images, labels = zip(*images)
    for image, label in images:
        # print(label)
        img = cv.imread(image, cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, (254, 254))  # Resize to 512x512
        if label is None:
            h, w = img.shape
            patches = []
            
            for y in range(0, h - 64 + 1, 64):
                for x in range(0, w - 64 + 1, 64):
                    patch = img[y:y+64, x:x+64]
                    patches.append(((x, y), patch))
            best_avg = -np.inf
            best_patch = None
            best_coord = None
            for (x, y), patch in patches:
                avg = patch.mean()
                if avg > best_avg:
                    best_avg = avg
                    best_patch = patch
                    best_coord = (x, y)
            img_slices.append(best_patch)
            img_labels.append(1)
            continue
        else:
            
            # img = cv.GaussianBlur(img, (3,3), 0)
            # img = clahe.apply(img)
            h, w = img.shape
            x1, y1, x2, y2, x3, y3, x4, y4 = label
            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x3 = int(x3 * w)
            y3 = int(y3 * h)
            new_x_min, new_x_max = get_crop_coords(x1, x3)
            new_y_min, new_y_max = get_crop_coords(y1, y3)
            img_slices.append(img[new_y_min:new_y_max, new_x_min:new_x_max])
            # img_labels.append([new_x_min, new_y_min, new_x_max, new_y_max])
            img_labels.append(0)

    # img_labels = np.array(img_labels, dtype=object)

    return img_slices, img_labels

#* Load test images
test_images_path = './kaggle_boneFracture/BoneFractureYolo8/test/images'
test_labels_path = './kaggle_boneFracture/BoneFractureYolo8/test/labels'

fractured_images, healthy_images = load_test_images(test_images_path, test_labels_path, target_category=2, limit_per_class=50)

print("Fractured images count:", len(fractured_images))
print("Healthy images count:", len(healthy_images))

sliced_fractured_images, y_fracture = get_sliced(fractured_images)
sliced_non_fractured_images, y_non_fracture = get_sliced(healthy_images)
#here i'm adding the sliced images from the test subset. These images were taken from the training set, but they are not used for training.
for filename in os.listdir("./processed_data/test/fracture"):
    sliced_fractured_images.append(cv.imread(os.path.join("./processed_data/test/fracture", filename), cv.IMREAD_GRAYSCALE))
    y_fracture.append(0)
for filename in os.listdir("./processed_data/test/no_fracture"):
    sliced_non_fractured_images.append(cv.imread(os.path.join("./processed_data/test/no_fracture", filename), cv.IMREAD_GRAYSCALE))
    y_non_fracture.append(1)

sliced_fractured_images = np.array(sliced_fractured_images, dtype=np.uint32)/255.0
sliced_non_fractured_images = np.array(sliced_non_fractured_images, dtype=np.uint32)/255.0

y_fracture = np.array(y_fracture, dtype=np.uint32)
y_non_fracture = np.array(y_non_fracture, dtype=np.uint32)
X_test = np.concatenate((sliced_fractured_images, sliced_non_fractured_images), axis=0)
y_test = np.concatenate((y_fracture, y_non_fracture), axis=0)

print("AfterFractured images count:", len(fractured_images))
print("Healthy images count:", len(healthy_images))

num_classes = 2
#iik gebruik de onderstaande one-hot niet omdat de labels al 0 en 1 zijn
y_one_hot = tf.keras.utils.to_categorical(y_test, num_classes)
# print("One-hot encoded labels shape:", y_one_hot.shape)
# print("Test images shape:", X_test.shape)
if len(X_test.shape) == 2:  # (64, 64)
    X_test = np.expand_dims(X_test, axis=0)     # (1, 64, 64)
    X_test = np.expand_dims(X_test, axis=-1)    # (1, 64, 64, 1)

# If X_test is already a batch of images (n_samples, 64, 64)

elif len(X_test.shape) == 3:  # (n_samples, 64, 64)
    X_test = np.expand_dims(X_test, axis=-1)    # (n_samples, 64, 64, 1)
# test_subset = tf.data.Dataset.from_tensor_slices((X_test, y_one_hot))
# test_subset = test_subset.batch(32)

#  this model detect bones as fractures. 



# model = load_model('my_model_20_epochs') #this has the best results so far
# model = load_model('my_model_25_epochs_25')
model = load_model('my_model_25_epochs_25_0.4')



#(batch_size, height, width, channels)
test_label_fracture_count = 0
test_label_non_fracture_count = 0
# for images, labels in test_subset.take(10):  # Just take one batch to inspect
#     for label in labels:
#         if label[0] == 1:
#             test_label_fracture_count += 1
#         else:
#             test_label_non_fracture_count += 1
            
    # print("Image batch shape:", images.shape)
    # print("Label batch shape:", labels.shape)

print("Test label fracture count:", test_label_fracture_count)
print("Test label non-fracture count:", test_label_non_fracture_count)
# print(" \"%\" of fracture labels", test_label_fracture_count / (test_label_fracture_count + test_label_non_fracture_count)*100)
# model.evaluate(test_subset, verbose = 1)

# sliding window inference process
def sliding_window_inference(model, image, window_size=(64, 64), step_size=32):
    height, width = image.shape[:2]
    predictions = []
    count = 0
    
    for y in range(0, height - window_size[0] + 1, step_size):
        for x in range(0, width - window_size[1] + 1, step_size):
            window = image[y:y + window_size[0], x:x + window_size[1]]
            window = tf.expand_dims(window, axis=0)  # Add batch dimension
            print("Window shape:", window.shape)
            pred = model.predict(window)
            predictions.append((x, y, pred))
            count += 1
    print("image shape:", image.shape)
    print(f"Total windows processed: {count}")

    return predictions


#in nummer 3 raakt het model in de war, omdat er bij deze afbeelding veel boten zitten met complexere patterns. 
# de tissue van de bot is wat donker 
target_size_ = (512, 512)  if i_test_image not in [1,4,5] else (256, 512)  # Adjust target size for the last image
clahe = cv.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
test_image = tf.keras.utils.load_img(fractured_images[i_test_image][0], color_mode='grayscale', target_size=target_size_)
test_image = tf.keras.utils.img_to_array(test_image) # Normalize to [0, 1]
test_image = cv.GaussianBlur(test_image, (3,3), 0)
test_image = clahe.apply((test_image ).astype(np.uint8))
#normalize to [0, 1]
test_image = tf.cast(test_image, tf.float32) / 255.0
# test_image = test_image / 255.0
# plt.imshow(test_image, cmap='gray')
# plt.title("Test Image")
# plt.axis('off')
# plt.show()
def get_color(value):
    if value < 0.05:
        return 'red'
    elif value < 0.1:
        return 'orange'
    elif value < 0.15:
        return '#FFD580'
    elif value < 0.2:
        return 'yellow'
    else:
        return 'blue'
prediction = sliding_window_inference(model, test_image)
fig, ax = plt.subplots()
ax.imshow(test_image, cmap='gray')
ax.set_title("Test Image with Sliding Window Predictions")
# minus infity
lowest_prediction =  [(0,0,[[float('inf')]]), (0,0,[[float('inf')]]), (0,0,[[float('inf')]])]# dit kan nul zijn omdat de output van de model ligt tussen 0 en 1, dus deze hoeft geen min infity zijn
lowest_prediction = sorted(prediction, key=lambda x: x[2][0][0])
for prediction in lowest_prediction:
    if prediction[2][0][0] < 0.3:
        color = get_color(prediction[2][0][0])
        print(f"Window at ({prediction[0]}, {prediction[1]}) - Prediction: {prediction[2][0][0]}")
        ax.add_patch(
            patches.Rectangle(
                (prediction[0],prediction[1]),
                64, 64,
                linewidth=1, edgecolor=color,
                facecolor='none'
            )
        )
        plt.text(prediction[0] + 16, prediction[1] + 16, f'{prediction[2][0][0]:.2f}', color=color, fontsize=8, ha='center')
print("Lowest prediction:", lowest_prediction[0])

# we are going to use the first 6 images for this sliding test
test_coord_x1 = fractured_images[i_test_image][1][0] * test_image.shape[1]
test_coord_y1 = fractured_images[i_test_image][1][1] * test_image.shape[0]
test_coord_x2 = fractured_images[i_test_image][1][2] * test_image.shape[1]
test_coord_y2 = fractured_images[i_test_image][1][3] * test_image.shape[0]
test_coords = [test_coord_x1, test_coord_y1, test_coord_x2 - test_coord_x1, test_coord_y2 - test_coord_y1]
# predicted_coords = lowest_prediction[:2]  # x, y from the lowest prediction
print("Test coordinates:", test_coords)
test_image_rect = patches.Rectangle(
    (test_coords[0], test_coords[1]), 
    test_coords[2], test_coords[3], 
    linewidth=1, edgecolor='g', 
    facecolor='none')
ax.add_patch(test_image_rect)

plt.show()

#########* evaluation of the model *#############
# study the meaning of the filtered outputs by comparing them for
# multiple samples
nLastLayer = len(model.layers)-1
nLayer=nLastLayer
print("lastLayer:",nLastLayer)
print(sliced_fractured_images.shape)
fig, ax = plt.subplots()
model.summary()
plt.imshow(sliced_fractured_images[0], cmap='gray')
plt.title('Sliced fractured img')
# plt.axis('off')
plt.show()
baseFilenameForSave=None
x_test_flat=None
plt.title("featuremaps no fractured images ")

# ViewTools_NN.printFeatureMapsForLayer(0, model, x_test_flat, sliced_non_fractured_images, 0, baseFilenameForSave)
# # plt.title(f"feature map: no fracture layer {layer}")
# test_image_sliced = test_image.copy()[lowest_prediction[0][1]:lowest_prediction[0][1]+64, lowest_prediction[0][0]:lowest_prediction[0][0]+64]
# # # # x_test_flat = np.expand_dims(test_image_sliced, axis=0)  # Add
# ViewTools_NN.printFeatureMapsForLayer(1, model, x_test_flat, sliced_non_fractured_images, 0, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(2, model, x_test_flat, sliced_non_fractured_images, 0, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(3, model, x_test_flat, sliced_non_fractured_images, 1, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(4, model, x_test_flat, sliced_non_fractured_images, 1, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(5, model, x_test_flat, sliced_non_fractured_images, 1, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(6, model, x_test_flat, sliced_non_fractured_images, 0, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLastLayer, model, x_test_flat, sliced_non_fractured_images, 1, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLastLayer, model, x_test_flat, sliced_non_fractured_images, 3, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLastLayer, model, x_test_flat, sliced_non_fractured_images, 4, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLastLayer, model, x_test_flat, sliced_non_fractured_images, 5, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLastLayer, model, x_test_flat, sliced_non_fractured_images, 6, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLastLayer, model, x_test_flat, sliced_non_fractured_images, 7, baseFilenameForSave)

# # # plt.title("Feature maps fractured images")
# ViewTools_NN.printFeatureMapsForLayer(nLastLayer, model, x_test_flat, sliced_fractured_images, 0, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(0, model, x_test_flat, sliced_fractured_images, 1, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(1, model, x_test_flat, sliced_fractured_images, 1, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(2, model, x_test_flat, sliced_fractured_images, 1, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLastLayer-1, model, x_test_flat, sliced_fractured_images, 1, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLastLayer, model, x_test_flat, sliced_fractured_images, 1, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLastLayer, model, x_test_flat, sliced_fractured_images, 2, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLastLayer, model, x_test_flat, sliced_fractured_images, 3, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(0, model, x_test_flat, sliced_fractured_images, 4, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(1, model, x_test_flat, sliced_fractured_images, 4, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLastLayer, model, x_test_flat, sliced_fractured_images, 4, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLastLayer, model, x_test_flat, sliced_fractured_images, 5, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLastLayer, model, x_test_flat, sliced_fractured_images, 6, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLastLayer, model, x_test_flat, sliced_fractured_images, 7, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLastLayer, model, x_test_flat, sliced_fractured_images, 8, baseFilenameForSave)

# ax.imshow(test_image, cmap='gray')
# ax.add_patch(test_image_rect)
# plt.title('Test Image with Bounding Box')
# # plt.axis('off')
# plt.show()

y_out = model.predict(X_test)
# y_out2 = model2.predict(X_test)
# y_out3 = model3.predict(X_test)

#X_test = [tf.expand_dims(x, axis=0) for x in X_test]
#y_out = [model.predict(x) for x in X_test]

for l, out in zip(y_test, y_out):
    print("Label:", l, "Prediction:", out)
test_threshold = 0.3
y_out = y_out.tolist()
y_out = [1 if out[0] >= test_threshold else 0 for out in y_out] #model.predict outputs eacht
# y_out2 = y_out2.tolist()
# y_out2 = [1 if out[0] >= 0.5 else 0 for out in y_out2]
# y_out3 = y_out3.tolist()
# y_out3 = [1 if out[0] >= 0.5 else 0 for out in y_out3] 
sum = 0
def check_mean_accuracy(y_test, y_out):
    sum = 0
    for l, out in zip(y_test, y_out):
        if out == l:
            sum += 1
    return sum / len(y_test)
mean = check_mean_accuracy(y_test, y_out)
# mean2 = check_mean_accuracy(y_test, y_out2)
# mean3 = check_mean_accuracy(y_test, y_out3)
print("Mean accuracy model 1:", mean)
# print("Mean accuracy model 2:", mean2)
# print("Mean accuracy model 3:", mean3)
print("Number of test images:", len(y_test))
val, count = np.unique(y_test, return_counts=True)
highest_count_index = np.argmax(count)
print(f"Most common label: {val[highest_count_index]} with baseline accuracy {count[highest_count_index] / len(y_test)}")
    


cm = tf.math.confusion_matrix(y_test, y_out)

img = X_test[0]
img = cv.resize(img, (512, 512))  # Resize to 512x512
# plt.imshow(img, cmap='gray')
# test_title = "test image " + str("fractured" if not y_test[0] else "non fractured")
# plt.title(test_title)
# plt.axis('off')
# plt.show()

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Set ticks and labels
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           xlabel='Predicted label',
           ylabel='True label',
           title='Confusion Matrix')

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.numpy().max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j].numpy(), 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.show()
    plt.savefig('./images for logboek/confusion_matrix.png', dpi=300, bbox_inches='tight')
    # Example usage
class_names = ["fracture", "non-fracture"]
# cm2 = tf.math.confusion_matrix(y_test, y_out2)
# cm3 = tf.math.confusion_matrix(y_test, y_out3)
plot_confusion_matrix(cm, class_names)
# plot_confusion_matrix(cm2, class_names)
# plot_confusion_matrix(cm3, class_names)