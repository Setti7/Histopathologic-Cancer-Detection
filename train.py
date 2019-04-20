#!/usr/bin/env python
# coding: utf-8

# # Histopathologic Cancer Detection
# *Training of a model for identification of metastatic tissue in  histopathologic scans of lymph node sections.*  
# *https://www.kaggle.com/c/histopathologic-cancer-detection*
# 
# ***
# 
# ## Dataset details:
# 
# The images are 96x96 pixels with 3 channels (RGB). They are labelled as metastatic only if there is cancerous cells inside the center 32x32 pixel region of the image. Presence of tumor cells outside of this region does not influence the label.
# 
# This caveat in the method of labelling the images doesn't seems to matter, because if there's no cancer cells inside the center of the image, why would there be cancer cells outside? However, this assumption could be wrong, so multiple training sessions will be used to determine what is the best approach to the problem.
# 
# The file `data/train_labels.csv` contains a dataframe with image ids and theirs respective labels.
# The data for training and validation is in `data/train`. The trained model will be used to predict the labels of the images in the `data/test` folder.
# 
# ## Preparing the Images:
# 
# Before anything else, the training images were separated into 2 folders: *metastatic* and *non-metastatic*. This is an important step, because it enables the use of `flow_from_directory` method from `keras.preprocessing.image.ImageDataGenerator`.
# 
# ## First Set of Models:
# 
# 1. **Full-Image-Gray**: a model trained on the full 96x96 image in gray-scale.
# 2. **Full-Image-RGB**: a model trained on the full 96x96 image in RGB.
# 3. **Center-Image-Gray**: a model trained on the 32x32 center patch in gray-scale.
# 4. **Center-Image-RGB**: a model trained on the 32x32 center patch in RGB.
# 
# ### Results:
# 
# It rapidly became apparent that using only the center patch of the image was not beneficial to the model, it ended reducing validation accuracy in every test. Also, models trained on grasycale images performed equaly and even better than those trained on RGB images.
# 
# The most successful model in this set used the full image in grayscale: it was a simple 9216-32c5-p2-64c5-p2-512-10, which is a popular architeture for the MNIST digit dataset. With some small changes (some dropouts were added), this model achieved ~86.6% validation accuracy at the cancer classification task.
# 
# Some other models were tested, with more dense/convolutional layers and with varying parameters, but all performed worse.
# 
# 
# ## Second Set of Models:
# 
# Intending to pass the 90% validation accuracy mark, transfer learning was used at this set of models.
# 
# The best models were trained 
# 
# 
# 
# 
# ## Model Architecture:
# 
# ### Observations:
# 1. Does using whitening and increasing brightness on the data augmentation help the model?
# 2. Maybe add some more metrics?

# In[ ]:


import numpy as np
from tensorflow.python.keras.layers import (Dense, Conv2D, MaxPool2D, Dropout,
                                            Flatten, BatchNormalization, 
                                            Activation, Lambda, MaxPooling2D,
                                            GlobalAveragePooling2D, Input)
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.callbacks import (ReduceLROnPlateau,
                                               EarlyStopping, ModelCheckpoint,
                                              CSVLogger)
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.python.keras.applications import MobileNet, densenet
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.backend import set_session
from tensorflow.keras import backend as K
import tensorflow as tf
from sklearn.utils import class_weight

# gpu_options = tf.GPUOptions(allow_growth=True)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# set_session(sess)


# Defining constants:

# In[ ]:


batch_size = 32
epochs = 50
images_shape = (96, 96, 3)
target_size = (224, 224, 3)

datagen_conf = {
    'target_size': target_size[:2],
    'color_mode': 'rgb',
    'batch_size': batch_size,
    'class_mode': 'sparse'
}

# TODO: normalize the data using mean and std of the dataset
# reference: https://www.kaggle.com/qitvision/a-complete-ml-pipeline-fast-ai
# TODO: find optimal hyperparameters for LR and weigth decay
# TODO: visualization is a good way of understanding what are the images the model struggles with. It might also reveal something about the dataset such as bad quality data.
# TODO: heatmap visualization

# TODO: try new data augmentation techniques, and check what each of the
#  transformations does to the image.
augmentation_conf = {
    'rotation_range': 180,
    'horizontal_flip': True,
    'vertical_flip': True,
    'zoom_range': 0.2,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
}


# Adapating MobileNet for this dataset:

# In[ ]:


# Imports the pre-trained model without the last (prediction) layer.
base_model = densenet.DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=target_size
)

# Creating the new classification layers of the model
x = GlobalAveragePooling2D()(base_model.output)
y = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.inputs, outputs=y)


# Model summary:

# In[ ]:


trainable_params = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
non_trainable_params = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

print(f"\nTrainable parameters: \t\t{trainable_params}")
print(f"Non-trainable paramenters: \t{non_trainable_params}\n")


# Defining the optimizer:

# In[ ]:


# optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# optimizer = Adam()
optimizer = RMSprop(lr=0.0001, decay=1e-6)


# Creating generator for the data flow:

# In[ ]:


train_datagen = ImageDataGenerator(rescale=1./255, **augmentation_conf)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train', **datagen_conf)

validation_generator = test_datagen.flow_from_directory(
        'data/validation', **datagen_conf)


# Defining callbacks:

# In[ ]:


# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

# Model checkpoint callback configuration:
model_name = "cancer-model.h5"
checkpoint = ModelCheckpoint(model_name, save_best_only=True)

# Logging the epochs results:
csv_logger = CSVLogger('epochs.log')

# Stopping training early if val_loss has stopped falling for 15 epochs
early_stop = EarlyStopping(patience=15)

# TODO: remove training_data. AUC can only be calculated on the validation data
# auc = AUCCallback(training_data=train_generator, validation_data=validation_generator)

callbacks = [learning_rate_reduction, checkpoint, csv_logger, early_stop]


# Compilling model:

# In[ ]:


try:
    model = multi_gpu_model(model, gpus=2)
except:
    print("Continuing with only 1 GPU.")

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)


# Calculating weights for the unbalanced dataset

# In[ ]:


# https://stackoverflow.com/questions/41815354/keras-flow-from-directory-over-or-undersample-a-class
class_weights = class_weight.compute_class_weight(
           'balanced',
            np.unique(train_generator.classes), 
            train_generator.classes)


# Fitting the model:

# In[ ]:


model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        class_weight=class_weights,
        callbacks=callbacks
)

