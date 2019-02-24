import pathlib
import keras
from skimage.external.tifffile import imread
from tqdm import tqdm
from skimage import transform
import numpy as np

from keras.layers import Conv2D, MaxPooling2D, add, concatenate, Input, Concatenate
from keras.layers import SpatialDropout2D, Dropout, ZeroPadding2D
from keras.layers import UpSampling2D, Conv2DTranspose, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def generate_data():
    slicks = []
    for p in pathlib.Path('data/sx8tiff').glob('**/*.tif'):
        slicks.append(imread(str(p)))

    nonslicks = []
    for p in tqdm(list(pathlib.Path('data/ql8tiff').glob('**/*.tif'))):
        img = imread(str(p))
        for _ in range(10):
            x0 = np.random.randint(0, img.shape[0] - 256)
            y0 = np.random.randint(0, img.shape[1] - 256)
            nonslicks.append(img[x0:x0 + 256, y0:y0 + 256])

    slicks = np.stack([transform.resize(each, (256, 256)) for each in slicks])
    nonslicks = np.stack(
        [transform.resize(each, (256, 256)) for each in nonslicks])

    features = np.vstack([slicks, nonslicks])
    labels = np.r_[np.ones(slicks.shape[0]), np.zeros(nonslicks.shape[0])]

    perm = np.random.permutation(np.arange(labels.size))

    return features[perm, ..., np.newaxis], to_categorical(labels[perm])


features, labels = generate_data()

train_x, test_x, train_y, test_y = train_test_split(
    features, labels, train_size=.7)
test_x, valid_x, test_y, valid_y = train_test_split(test_x, test_y)

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=True,
    width_shift_range=3,
    zoom_range=.1,
    rotation_range=180)
datagen.fit(train_x)

input = Input((256, 256, 1))
x = Conv2D(128, (3, 3), padding='same', activation='relu')(input)
x = Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
x = Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
x = MaxPooling2D()(x)
x = Conv2D(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
x = Flatten()(x)
y = Dense(2, activation='sigmoid')(x)
model = keras.Model(input=input, output=y)
model.compile(keras.optimizers.Adam(), 'binary_crossentropy', metrics=['acc'])

batch_size = 64
g = model.fit_generator(
    datagen.flow(train_x, train_y, batch_size=batch_size),
    steps_per_epoch=train_y.shape[0] // batch_size,
    epochs=100,
    callbacks=[
        keras.callbacks.ModelCheckpoint('/tmp/model.{epoch:03d}.hdf5'),
        keras.callbacks.EarlyStopping(patience=3)
    ],
    validation_data=datagen.flow(valid_x, valid_y, batch_size=32),
    validation_steps=valid_y.shape[0] // batch_size,
)

predictions = model.predict_generator(
    datagen.flow(test_x, batch_size=1), steps=test_y.shape[0])
accuracy = (predictions.argmax(axis=1) == test_y.argmax(axis=1)).mean()
print(accuracy)
