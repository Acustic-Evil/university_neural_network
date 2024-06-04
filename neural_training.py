import os
from PIL import Image
import cv2
import numpy as np
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


# Custom callback to calculate precision, recall, and F1-score
class MetricsCallback(keras.callbacks.Callback):
    def __init__(self, validation_data):
        super(MetricsCallback, self).__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_precision = precision_score(val_targ.argmax(axis=1), val_predict.argmax(axis=1), average='weighted',
                                         zero_division=1)
        _val_recall = recall_score(val_targ.argmax(axis=1), val_predict.argmax(axis=1), average='weighted',
                                   zero_division=1)
        _val_f1 = f1_score(val_targ.argmax(axis=1), val_predict.argmax(axis=1), average='weighted', zero_division=1)
        logs['val_precision'] = _val_precision
        logs['val_recall'] = _val_recall
        logs['val_f1'] = _val_f1
        print(f" — val_precision: {_val_precision:.4f} — val_recall {_val_recall:.4f} — val_f1: {_val_f1:.4f}")


# Function to plot training history
def plot_training(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    precision = history.history.get('val_precision', [])
    recall = history.history.get('val_recall', [])
    f1 = history.history.get('val_f1', [])
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    if precision and recall and f1:
        plt.subplot(1, 3, 3)
        plt.plot(epochs, precision, 'go', label='Validation precision')
        plt.plot(epochs, recall, 'ro', label='Validation recall')
        plt.plot(epochs, f1, 'mo', label='Validation F1')
        plt.title('Validation precision, recall, and F1-score')
        plt.legend()

    plt.show()


def neural_training():
    data = []
    labels = []
    categories = ["deer", "cows", "moose", "unknown"]
    for label, category in enumerate(categories):
        for filename in os.listdir(f"dataset/{category}"):
            imag = cv2.imread(f"dataset/{category}/{filename}")
            img_from_ar = Image.fromarray(imag, 'RGB')
            resized_image = img_from_ar.resize((128, 128))
            data.append(np.array(resized_image))
            labels.append(label)

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

    split_index = int(0.1 * data_length)
    (x_train, x_test) = animals[split_index:], animals[:split_index]
    (y_train, y_test) = labels[split_index:], labels[:split_index]

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=2, padding="same", activation="relu", input_shape=(128, 128, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=2, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(500, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation="softmax"))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest')

    datagen.fit(x_train)

    metrics_callback = MetricsCallback(validation_data=(x_test, y_test))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    history = model.fit(datagen.flow(x_train, y_train, batch_size=60), epochs=100, validation_data=(x_test, y_test),
                        callbacks=[metrics_callback, early_stopping, reduce_lr])

    plot_training(history)

    score = model.evaluate(x_test, y_test, verbose=1)
    print('\n', 'Test accuracy:', score[1])

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("model.h5")
    print("Saved model to disk")


if __name__ == "__main__":
    neural_training()
