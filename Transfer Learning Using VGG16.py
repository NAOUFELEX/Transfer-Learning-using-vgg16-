#_______________________________import libraries_______________________________
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from glob import *

#_______________________________Loading dataset_______________________________

IMAGE_SIZE = [128,128]
train_path = "BRAIN_MRI/Training"
test_path = "BRAIN_MRI/Testing"


#_______________________________Implementing transfer learning_______________________________

#Now that the dataset has been loaded, it’s time to implement transfer learning.
#   Begin by importing VGG16 from keras.applications and provide the input image size.
#   Weights are directly imported from the ImageNet classification problem. When top=False, it means to discard the weights of the input layer and the output layer as you will use your own inputs and outputs.

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet',
include_top=False)

vgg.input

for layer in vgg.layers:
  layer.trainable = False

#get the number of folders using glob.
folders = glob('BRAIN_MRI/Training/*')
print("the number of folders:", len(folders))
x = Flatten() (vgg.output)
prediction = Dense (len(folders), activation='softmax') (x)
model = Model (inputs=vgg.input, outputs=prediction)
model.summary()

#The following step experiments with Adam optimizer, binary_crossentropy loss function and accuracy as metrics.

adam = optimizers.Adam()
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

#_______________________________Data augmentation_______________________________

#The next stage is image augmentation. You will import prepocess_input as there were some preprocessing steps when the actual model was trained in the imagenet problem. To achieve similar results, you need to make sure that you use the exact preprocessing steps. Some, including shifting and zooming, are used to reduce overfitting.

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    brightness_range=(0.85, 1.15),
    width_shift_range=0.002,
    height_shift_range=0.002,
    shear_range=12.5,
    zoom_range=0,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode="nearest"
)


#No augmentation of the test data, just rescaling
test_datagen = ImageDataGenerator(rescale=1./255)


#Specify the target size of the output, batch size, and the class.
train_set = train_datagen.flow_from_directory(
    train_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

#The same is done for the testing set.
test_set = test_datagen.flow_from_directory(
    test_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

#__Data Augmentation Class Indices__
# Accessing class indices for training data generator
class_indices_train = train_set.class_indices
class_indices_train_list = list(train_set.class_indices.keys())


# Displaying categorical types
print("Categorical types for the training data:")
print(class_indices_train)

#_______________________________Training the model_______________________________

#Now that data augmentation has been completed, it’s time to train the model. Model checkpoint is used to save the best model. We will use 50 epochs with 179 steps per epoch. The validation steps equal to 32.

history = model.fit(train_set,
          steps_per_epoch=len(train_set),
          validation_data=test_set,
          epochs=50)
model.save('brain_tumor_vgg16_fin.h5')



#_______________________________Model Evaluating_______________________________
import matplotlib.pyplot as plt

# Extracting training history data
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(history.epoch) + 1)

# Plotting accuracy
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

plt.plot(epochs_range, acc, label='Train Set')
plt.plot(epochs_range, val_acc, label='Val Set')
plt.legend(loc="best")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Proposed VGG16 Model Accuracy')
plt.grid(True)
plt.show()



import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Predictions on the test set
predictions = model.predict(test_set)
y_pred = np.argmax(predictions, axis=1)
y_true = test_set.classes

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()



from sklearn.metrics import confusion_matrix , classification_report , ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_true , y_pred)
plt.title("Confusion Matrix", fontname = "monospace", fontsize = 15, weight = "bold")
plt.show()

import os, colorama
from colorama import Fore,Style,Back
class_names = class_indices_train

print("\nClassification Report:\n")
print(Fore.WHITE + classification_report(y_true, y_pred, target_names = class_names, digits= 4))
