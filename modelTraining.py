"""
    This is the code for training model
"""

import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.regularizers import l2
from keras.utils import to_categorical
import loadData as lD
import keys


keys = keys.keys()

# Getting dataset
train_data, train_label, test_data, test_label = lD.load_datas(keys, 20000)
print(train_data[0])

# Data Preprocessing
train_label = to_categorical(train_label)
test_label = to_categorical(test_label)

# Creating models
model = Sequential()
model.add(Conv2D(16, kernel_size=(5, 5), padding='same', input_shape=(28, 28, 1), activation='relu'))
model.add(Conv2D(16, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(16, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())

model.add(Dense(128, activation="relu", kernel_regularizer=l2(0.01)))
model.add(Dense(256, activation="relu", kernel_regularizer=l2(0.01)))
model.add(Dense(10, activation="softmax"))

model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


history = model.fit(train_data, train_label, validation_split=0.2, epochs=20, batch_size=200)
loss, acc = model.evaluate(train_data, train_label)
print("訓練資料集準確度：{:.2f}".format(acc))
loss, acc = model.evaluate(test_data, test_label)
print("測試資料集準確度：{:.2f}".format(acc))

print("Saving Model as QuickDrawCNN.h5 ...")
model.save("QuickDrawCNN.h5")

loss = history.history["accuracy"]
epochs = range(1, len(loss)+1)
val_loss = history.history["val_accuracy"]
plt.plot(epochs, loss, "b-", label="Training Acc")
plt.plot(epochs, val_loss, "r--", label="Validation Acc")
plt.title("Training and Validation Acc")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.legend()
plt.show()

