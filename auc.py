from sklearn.metrics import roc_auc_score
import numpy as np
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm


batch_size = 32
target_size = (224, 224, 3)

datagen_conf = {
    'target_size': target_size[:2],
    'color_mode': 'rgb',
    'batch_size': batch_size,
    'class_mode': 'sparse',
}


test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
        'data/validation', **datagen_conf)


model = load_model("cancer-model.h5")


x_val = validation_generator
y_val = validation_generator.classes


preds = np.empty(shape=[0, 2])
labels = np.empty(shape=[0, ])

i = 0
for batch_x, batch_y in tqdm(x_val):
    result = model.predict(batch_x)
    preds = np.concatenate([preds, result])
    labels = np.concatenate([labels, batch_y])

    if i >= len(x_val):
        break

    i += 1

predictions = np.argmax(preds, axis=-1)

roc_val = roc_auc_score(labels, predictions)

val = round(roc_val, 5)

print(f'AUC: {val}')
with open(f'AUC-{val}', 'w') as f:
    f.write('')