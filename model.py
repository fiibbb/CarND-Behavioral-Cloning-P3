from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Flatten, Lambda, Dense, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers import Cropping2D
from sklearn.model_selection import train_test_split
import sklearn as sk
import numpy as np
import cv2
import csv
import argparse
from os import path
from random import shuffle

# Assume input shape is (160, 320, 3)
def nv_sdc():

    model = Sequential()

    # Normalization and cropping
    model.add(Lambda(lambda x: (x / 255. - 0.5), input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=(160, 320, 3)))

    # 3 5x5 convolutional layers and 2 3x3 convolutional layers
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # Flatten
    model.add(Flatten())

    # 3 fully connected layers
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))

    # Regression model outputs only one value
    model.add(Dense(1))

    return model

def load_samples(dirs):
    all_samples = []
    for dir in dirs:
        with open(path.join(dir, 'driving_log.csv')) as f:
            reader = csv.reader(f)
            lines = [line for line in reader]
            all_samples = all_samples + lines
    print('Loaded {} samples'.format(len(all_samples)))
    return all_samples

def sample_generator(samples, batch_size=60):  # batch_size should be multiple of 6
    correction = 0.1
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size // 6):
            batch_samples = samples[offset : offset + batch_size // 6]
            images, angles = [], []
            for batch_sample in batch_samples:
                new_images = [cv2.cvtColor(cv2.imread(batch_sample[i].strip(), cv2.COLOR_BGR2RGB)) for i in range(3)]
                new_angles = [float(batch_sample[3]), float(batch_sample[3]) + correction, float(batch_sample[3]) - correction]
                new_images = new_images + [cv2.flip(new_image, 1) for new_image in new_images]
                new_angles = new_angles + [-new_angle for new_angle in new_angles]
                images, angles = images + new_images, angles + new_angles
            yield sk.utils.shuffle(np.array(images), np.array(angles))

def train(data_dirs, epochs=4, saved_model=None, save_to=None):
    BATCH_SIZE = 60

    print('Loading...')
    all_samples = load_samples(data_dirs)
    train_samples, valid_samples = train_test_split(all_samples, test_size=0.2)
    train_generator = sample_generator(train_samples, batch_size=BATCH_SIZE)
    valid_generator = sample_generator(valid_samples, batch_size=BATCH_SIZE)

    print('Using {} training examples...'.format(len(train_samples)))

    if saved_model:
        print('Using model {}...'.format(saved_model))
        m = load_model(saved_model)
    else:
        print('Using new model')
        m = nv_sdc()
        m.compile(loss='mae', optimizer='adam')

    print('Training...')
    m.fit_generator( \
        train_generator, \
        validation_data=valid_generator, \
        steps_per_epoch=len(train_samples)*6/BATCH_SIZE, \
        epochs=epochs, \
        validation_steps=len(valid_samples)*6/BATCH_SIZE \
    )
    # m.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=6)

    if save_to:
        print('Saving...')
        m.save(save_to)
        print(' -- saved to {}'.format(save_to))

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('epochs')
    parser.add_argument('save_to')
    args = parser.parse_args()

    data_dirs = [
        'data/01_official',
        'data/02_keep_lane',
        'data/03_recover_lane',
        'data/04_curve',
        'data/05_reverse'
    ]
    train(data_dirs, epochs=int(args.epochs), save_to=args.save_to)

if __name__ == '__main__':
    run()
