import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

# Load the ResNet50 model, pre-trained on ImageNet

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

Downloading data from <https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5>
94765736/94765736 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step

import tensorflow as tf
import os, glob
import cv2
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, Dense, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Flatten
import os
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
from PIL import Image

<ipython-input-2-68b62de3dcf6>:4: TqdmDeprecationWarning: This function will be removed in tqdm==5.0.0
Please use `tqdm.notebook.*` instead of `tqdm._tqdm_notebook.*`
  from tqdm._tqdm_notebook import tqdm_notebook as tqdm

# Cell 1: Mount Google Drive

# This cell mounts Google Drive to access the dataset stored in it

from google.colab import drive
drive.mount('/content/drive')

# Define the dataset directory path and categories

data_dir = '/content/drive/MyDrive/dataset4cat/'  # Adjust this path based on your dataset location in Google Drive
categories = ['normal', 'cataract', 'diabetic_retinopathy', 'glaucoma']

%pip install seaborn
%pip install matplotlib
print('Dataset extraction complete')

# This cell will calculate and visualize how many images exist in each category

category_counts = []

for category in categories:
    folder_path = os.path.join(data_dir, category)
    category_counts.append((category, len(os.listdir(folder_path))))

df_counts = pd.DataFrame(category_counts, columns=['Classes', 'Count'])

# Plot the number of images per category

plt.figure(figsize=(8, 6))
sns.barplot(x='Classes', y='Count', data=df_counts)
plt.title('Number of Images per Classes')
plt.show()

# This cell will display a sample image from each class for visualization purposes

import matplotlib.image as mpimg

fig, axes = plt.subplots(1, len(categories), figsize=(15, 5))

for i, category in enumerate(categories):
    folder_path = os.path.join(data_dir, category)
    img_file = os.listdir[folder_path](0)  # Display the first image as a sample
    img_path = os.path.join(folder_path, img_file)
    img = mpimg.imread(img_path)
    axes[i].imshow(img)
    axes[i].set_title(category)
    axes[i].axis('off')

plt.tight_layout()
plt.show()

import os

# Path to the dataset4cat folder in your Google Drive

dataset_path = '/content/drive/MyDrive/dataset4cat'

# Recursively list all files and directories

for root, dirs, files in os.walk(dataset_path):
    print(f'Root: {root}')
    print(f'Directories: {dirs}')
    # Uncomment the next line to print files if needed
    # print(f'Files: {files}')
    print('-' * 40)

count = 0

# Adjust the path to your dataset4cat folder in Google Drive

for root, folders, filenames in os.walk('/content/drive/MyDrive/dataset4cat'):
    print(root, folders)

import os
import cv2
import numpy as np
from tqdm import tqdm

# Step 1: Set the correct dataset path (adjusted for Google Drive)

dataset_path = '/content/drive/MyDrive/dataset4cat'  # This is the path to the dataset4cat folder

# Step 2: Initialize arrays for storing images and labels

X = []
y = []

# Define the classes (assuming four types of eye diseases)

classes = ['cataract', 'glaucoma', 'normal', 'diabetic_retinopathy']

# Step 3: Loop through each class folder and load images

for i, class_name in enumerate(classes):
    class_path = os.path.join(dataset_path, class_name)  # Path to each class folder (inside dataset4cat)

    for img_name in tqdm(os.listdir(class_path)):  # List all image files in the class folder
        img_path = os.path.join(class_path, img_name)  # Full path to the image
        img = cv2.imread(img_path)  # Read the image

        if img is not None:  # Make sure the image is loaded correctly
            img = cv2.resize(img, (224, 224))  # Resize to the required input size for models like VGG16 or ResNet
            X.append(img)  # Add the image to the list
            y.append(i)  # Use class index as label

# Step 4: Convert to NumPy arrays for model input

X = np.array(X)
y = np.array(y)

# Display the shape of X and y

print(f"Loaded {X.shape[0]} images with shape {X.shape[1:]} and {y.shape[0]} labels.")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Step 2: Split the training data into training and validation sets

# We create a validation set from the training set (10% of the training data)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Step 3: One-hot encode the labels for training, validation, and test sets

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)  # Fit and transform the training labels
y_val = lb.transform(y_val)          # Transform the validation labels
y_test = lb.transform(y_test)        # Transform the test labels

# Step 6: Print the shape of the data to verify the splits

print("X_train Shape: ", X_train.shape)  # Shape of the training images
print("X_val Shape: ", X_val.shape)      # Shape of the validation images
print("X_test Shape: ", X_test.shape)    # Shape of the test images
print("y_train Shape: ", y_train.shape)  # Shape of the one-hot encoded training labels
print("y_val Shape: ", y_val.shape)      # Shape of the one-hot encoded validation labels
print("y_test Shape: ", y_test.shape)    # Shape of the one-hot encoded test labels

# Now you can use train_generator for model training

# Example of training the model

# history = model.fit(train_generator, validation_data=val_generator, epochs=30)

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# Step 1: Load the ResNet50 base model (pre-trained on ImageNet) without the top classification layers

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Step 2: Freeze the base model layers to prevent them from being updated during training

base_model.trainable = False

# Step 3: Define custom layers on top of the base model

# These layers will fine-tune the model for your specific classification task (4 classes)

def add_custom_layers(bottom_model, num_classes):
    # Global average pooling layer: Reduces the dimensionality of the output from the base model
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)

    # Fully connected layers (Dense layers) to learn features from the output of the base model
    top_model = Dense(1024, activation='relu')(top_model)  # First dense layer
    top_model = Dense(1024, activation='relu')(top_model)  # Second dense layer
    top_model = Dense(512, activation='relu')(top_model)   # Third dense layer
    top_model = Dense(num_classes, activation='softmax')(top_model)  # Output layer: 4 classes, softmax for multi-class classification

    return top_model

# Step 4: Define the number of classes for your classification task

num_classes = 4  # Assuming 4 classes (cataract, glaucoma, normal, diabetic_retinopathy)

# Step 5: Add custom layers on top of the base model

custom_model_output = add_custom_layers(base_model, num_classes)

# Step 6: Create the final model using the base model and custom layers

model = Model(inputs=base_model.input, outputs=custom_model_output)

# Step 7: Freeze the base model layers to avoid training them

# This will train only the custom layers on top of the base model

base_model.trainable = False

# Step 8: Compile the model

# The optimizer is Adam, the loss function is categorical cross-entropy (for multi-class classification), and we track accuracy

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 9: Display the model summary to show the structure of the network

model.summary()

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# Step 1: Set a specific learning rate for Adam optimizer

learning_rate = 0.0001  # You can adjust this value as needed

# Step 2: Compile the model with the custom learning rate

# The model is compiled with

# - Adam optimizer and the specified learning rate

# - Categorical crossentropy loss for multi-class classification

# - Accuracy as a metric to track during training

model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 3: Train the model

# We train the model using the training data (X_train, y_train) and validate it with the validation data (X_val, y_val)

# - `epochs=30`: The model will go through the entire training data 30 times

# - `validation_data=(X_val, y_val)`: During training, the model will evaluate performance on the validation data

# - `verbose=1`: Display progress information during training

history = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val), verbose=1)
from tensorflow.keras.models import load_model  # Import to load the model later

# Save the model (new section)

model_save_path = '/content/drive/MyDrive/saved_model/resnet50_model.h5'  # Adjust the path as needed
model.save(model_save_path)  # Save the trained model in HDF5 format
print(f"Model saved at: {model_save_path}")

import numpy as np
import matplotlib.pyplot as plt

# Assuming `history` contains the training history of the model (history.history dict)

best_epoch = np.argmax(history.history['val_accuracy']) + 1  # Add 1 as epochs start from 1

# Plot Training & Validation Accuracy

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.axvline(x=best_epoch-1, linestyle='--', color='r', label=f'Best Epoch: {best_epoch}')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot Training & Validation Loss

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.axvline(x=best_epoch-1, linestyle='--', color='r', label=f'Best Epoch: {best_epoch}')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

from sklearn.metrics import roc_curve, auc

# Step 1: Compute ROC curve and ROC area for each class

n_classes = y_test.shape[1]  # Number of classes (4 classes in this case)
fpr = dict()
tpr = dict()
roc_auc = dict()

y_pred = model.predict(X_test)

for i in range(n_classes):
    fpr[i], tpr[i],_ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Step 2: Plot all ROC curves

plt.figure(figsize=(10, 8))
colors = ['red', 'blue', 'yellow', 'green']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve for {classes[i]} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ResNet50 ROC Curve for Multi-Class Eye Disease Classification')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Step 1: Convert predicted probabilities to class labels (for test data)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Step 2: Convert one-hot encoded true labels to class labels (for train/val/test data)

y_train_classes = np.argmax(y_train, axis=1)
y_val_classes = np.argmax(y_val, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Evaluate on the training, validation, and test sets

train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

# Confusion matrix for accuracy per class

conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
print("Confusion Matrix:")
print(conf_matrix)

# Print metrics

print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}\n")

# Print accuracy per class

classes = ['Normal', 'Glaucoma', 'Diabetic Retinopathy', 'Cataract']
for i, class_name in enumerate(classes):
    print(f"Accuracy for {class_name}: {class_accuracy[i]:.4f}")

from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# Classification Report

class_report = classification_report(y_test_classes, y_pred_classes, target_names=classes)
print("Classification Report:\n", class_report)

# Confusion Matrix Plot

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(' ResNet50 Confusion Matrix')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()

# The model generates predictions for the test data

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true_classes, y_pred_classes)
precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
auc = roc_auc_score(y_test, y_pred, multi_class='ovr', average='weighted')

# Print the metrics

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC: {auc}")
