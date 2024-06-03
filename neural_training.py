import os
from PIL import Image
import cv2
import numpy as np
from tensorflow import keras


def neural_training():
    data = []
    labels = []
    many_deer = os.listdir("dataset/deer")
    for deer in many_deer:
        imag = cv2.imread("dataset/deer/" + deer)
        img_from_ar = Image.fromarray(imag, 'RGB')
        resized_image = img_from_ar.resize((50, 50))
        data.append(np.array(resized_image))
        labels.append(0)

    cows = os.listdir("dataset/cows")
    for cow in cows:
        imag = cv2.imread("dataset/cows/" + cow)
        img_from_ar = Image.fromarray(imag, 'RGB')
        resized_image = img_from_ar.resize((50, 50))
        data.append(np.array(resized_image))
        labels.append(1)

    elks = os.listdir("dataset/moose")
    for moose in elks:
        imag = cv2.imread("dataset/moose/" + moose)
        img_from_ar = Image.fromarray(imag, 'RGB')
        resized_image = img_from_ar.resize((50, 50))
        data.append(np.array(resized_image))
        labels.append(2)

    # Unknown images
    unknowns = os.listdir("dataset/unknown")
    for unknown in unknowns:
        imag = cv2.imread("dataset/unknown/" + unknown)
        img_from_ar = Image.fromarray(imag, 'RGB')
        resized_image = img_from_ar.resize((50, 50))
        data.append(np.array(resized_image))
        labels.append(3)

    animals = np.array(data)
    labels = np.array(labels)

    np.save("animals", animals)
    np.save("labels", labels)

    s = np.arange(animals.shape[0])
    np.random.shuffle(s)
    animals = animals[s]
    labels = labels[s]

    num_classes = len(np.unique(labels))
    data_length = len(animals)

    (x_train, x_test) = animals[(int)(0.1 * data_length):], animals[:(int)(0.1 * data_length)]
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    train_length = len(x_train)
    test_length = len(x_test)

    (y_train, y_test) = labels[(int)(0.1 * data_length):], labels[:(int)(0.1 * data_length)]

    # One hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # import sequential model and all the required layers
    from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
    from keras.models import Sequential

    model = Sequential()

    # make model
    model.add(Conv2D(filters=16, kernel_size=2, padding="same", activation="relu", input_shape=(50, 50, 3)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=2, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation="softmax"))
    model.summary()

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=50, epochs=100, verbose=1)

    score = model.evaluate(x_test, y_test, verbose=1)
    print('\n', 'Test accuracy:', score[1])

    from keras.models import model_from_json
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
