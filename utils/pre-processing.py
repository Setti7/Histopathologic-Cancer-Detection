# https://www.kaggle.com/c/histopathologic-cancer-detection
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Loading the dataset
df = pd.read_csv('data/train_labels.csv')
print('Dataset has', len(df), 'rows')


# Checking for null values:
y_train = df['label']
x_train = df['id']

if x_train.isnull().any() or y_train.isnull().any():
    print('Dataset has null values')

else:
    print('Dataset does not have null values')


# Checking if the dataset is balanced
print('Is the dataset balanced?')
print(y_train.value_counts())


# Moving training data into folders
imgs = os.listdir('data/train_all/')
print("Num of training images: ", len(imgs))

metastatic = df.loc[df['label'] == 1]
non_metastatic = df.loc[df['label'] == 0]

metas = metastatic['id'].values
print('Moving training data into folders')

for img in tqdm(imgs):
    img_id = os.path.splitext(img)[0]
    src_path = os.path.join('data', 'train_all', img)
    
    if img_id in metas:
        os.rename(src_path, 'data/train/metastatic/%s' % img)

os.rename('data/train_all/', 'data/train/non-metastatic/')

print('All metastatic images moved to data/train/metastatic/')
print('Non-metastatic images moved to data/train/non-metastatic/')


# Confirming moving of images were successful
metas = os.listdir('data/train/metastatic/')
non_metas = os.listdir('data/train/non-metastatic/')

print('These values should match: (sum of images in folders) (total number of rows)')
print(len(metas) + len(non_metas), len(df))
assert (len(metas) + len(non_metas)) == len(df)


# Creating validation folder
print('Creating validation folder')
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1)

train = pd.concat([x_train, y_train], axis=1)
val = pd.concat([x_val, y_val], axis=1)


metastatic = df.loc[df['label'] == 1]
metas = metastatic['id'].values

for img_id in tqdm(val['id'].values):

    if img_id in metas:
        path = os.path.join('data', 'train', 'metastatic', '%s.tif' % img_id)
        os.rename(path, 'data/validation/metastatic/%s.tif' % img_id)

    else:
        path = os.path.join('data', 'train', 'non-metastatic', '%s.tif' % img_id)
        os.rename(path, 'data/validation/non-metastatic/%s.tif' % img_id)


val_meta = os.listdir('data/validation/metastatic')
val_non = os.listdir('data/validation/non-metastatic')

train_meta = os.listdir('data/train/metastatic')
train_non = os.listdir('data/train/non-metastatic')

print('number of images in folder: (val_meta) (val_non) (train_meta) (train_non)')
print(len(val_meta), len(val_non), len(train_meta), len(train_non))
print('Total number of images: (validation) (training)')
print(len(val_meta) + len(val_non), len(train_meta) + len(train_non))
