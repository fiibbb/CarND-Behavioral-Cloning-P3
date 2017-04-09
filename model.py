from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Flatten, Lambda, Dense, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers import Cropping2D
import numpy as np
import cv2
import csv
import argparse
from os import path

# Assume input shape is (160, 320, 3)
def nv_sdc():

    model = Sequential()

    # Normalization and cropping
    model.add(Lambda(lambda x: (x / 255. - 0.5), input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))

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

def load_data(dir):
    print(' -- loading samples from {}'.format(dir))
    with open(path.join(dir, 'driving_log.csv')) as f:
        reader = csv.reader(f)
        lines = [line for line in reader][1:]

    correction = 0.2
    images = [cv2.imread(path.join(dir, line[0].strip())) for line in lines] + \
             [cv2.imread(path.join(dir, line[1].strip())) for line in lines] + \
             [cv2.imread(path.join(dir, line[2].strip())) for line in lines]
    angles = [float(line[3]) for line in lines] + \
             [float(line[3]) + correction for line in lines] + \
             [float(line[3]) - correction for line in lines]
    images = images + [cv2.flip(image.copy(), 1) for image in images]
    angles = angles + [-angle for angle in angles]

    print(' -- loaded {} x 6 samples'.format(len(lines)))
    return images, angles

def load_all_data(dirs):
    all_images, all_angles = [], []
    for dir in dirs:
        new_images, new_angles = load_data(dir)
        all_images, all_angles = all_images + new_images, all_angles + new_angles
    return np.array(all_images), np.array(all_angles)

def train(recording_dirs, saved_model=None, save_to=None):
    print('Loading...')
    X_train, y_train = load_all_data(recording_dirs)

    print('Using {} training examples...'.format(len(X_train)))

    if saved_model:
        print('Using model {}...'.format(saved_model))
        m = load_model(saved_model)
    else:
        print('Using new model')
        m = nv_sdc()
        m.compile(loss='mse', optimizer='adam')

    print('Training...')
    m.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=6)

    if save_to:
        print('Saving...')
        m.save(save_to)
        print(' -- saved to {}'.format(save_to))

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('save_to')
    args = parser.parse_args()

    train(['data'], save_to=args.save_to)

if __name__ == '__main__':
    run()
