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
# I believe this caveat in the method of labelling the images doesn't matter, because if there's no cancer cells inside the center of the image, why would there be cancer cells outside? However, I don't have enough medical knowledge to know if this assumption is correct, so multiple training sessions will be used to determine what is the best approach to the problem.
# 
# The file `data/train_labels.csv` contains a dataframe with image ids and theirs respective labels.
# The data for training and validation is in `data/train`. The trained model will be used to predict the labels of the images in the `data/test` folder.
# 
# ## Preparing the Images:
# 
# Before anything else, the training images were separated into 2 folders: *metastatic* and *non-metastatic*. This is an important step, because it enables the use of `flow_from_directory` method from `keras.preprocessing.image.ImageDataGenerator`.
# 
# ## Models:
# 
# 1. **Full-Image-Gray**: a model trained on the full 96x96 image in gray-scale.
# 2. **Full-Image-RGB**: a model trained on the full 96x96 image in RGB.
# 3. **Center-Image-Gray**: a model trained on the 32x32 center patch in gray-scale.
# 4. **Center-Image-RGB**: a model trained on the 32x32 center patch in RGB.
# 
# All models will make use of data augmentation techniques.
# Also, the center-image models should use zoom-outs instead of zoom-ins for data augmentation.
# 
# ## Model Architecture:
# 
# ### Observations:
# 1. Does using whitening and increasing brightness on the data augmentation help the model?
# 2. Maybe add some more metrics?

# In[ ]:


import numpy as np
from utils import augment_2d
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
from tensorflow.python.keras.applications import MobileNet
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.backend import set_session
import tensorflow as tf
from sklearn.utils import class_weight

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# set_session(sess)


# Defining constants:

# In[ ]:


batch_size = 128
epochs = 50
images_shape = (96, 96, 3)
target_size = (128, 128)

datagen_conf = {
    'target_size': target_size,
    'color_mode': 'rgb',
    'batch_size': batch_size,
    'class_mode': 'sparse'
}

data_augmentation_conf = {
    'rotation': 180,
    'horizontal_flip': True,
    'vertical_flip': True,
    'crop': True
}


# Adapating MobileNet for this dataset:

# In[ ]:


# Imports the mobilenet model and discards the last 1000 neuron layer.
base_model = MobileNet(
    weights='imagenet',
    include_top=False,
    input_shape=(128, 128, 3)
)

# Freezing base model pre-trained layers
base_model.trainable = False


inputs = Input(shape=images_shape, name='input')

x = Lambda(augment_2d,
                arguments={
                    'rotation': 180,
                    'horizontal_flip': True,
                    'vertical_flip': True,
                    'crop': True,
                }, name='augmentation_layer')(inputs)

x = base_model(x)

x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)

# Output layer
y = Dense(2, activation='softmax')(x)

model = Model(inputs=inputs, outputs=y)


# Model summary:

# In[ ]:


model.summary()


# Defining the optimizer:

# In[ ]:


# optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# optimizer = Adam()
optimizer = RMSprop(lr=0.0001, decay=1e-6)


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


# Creating generator for the data flow:

# In[ ]:


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train', **datagen_conf)

validation_generator = test_datagen.flow_from_directory(
        'data/validation', **datagen_conf)


# Calculating weights for the unbalanced dataset

# In[ ]:


# https://stackoverflow.com/questions/41815354/keras-flow-from-directory-over-or-undersample-a-class
class_weights = class_weight.compute_class_weight(
           'balanced',
            np.unique(train_generator.classes), 
            train_generator.classes)


# Fitting the model:

# In[ ]:


# model.fit_generator(
#         train_generator,
#         epochs=epochs,
#         validation_data=validation_generator,
#         class_weight=class_weights,
#         callbacks=callbacks
# )


# In[ ]:




