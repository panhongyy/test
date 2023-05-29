from tensorflow.keras import layers, models, Model, Sequential


def mymodel(im_height=300, im_width=500, num_class=4):
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    x = input_image
    x = layers.Conv2D(8, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(16, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(32, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(4, activation="relu")(x)
    predict = layers.Softmax()(x)

    model = models.Model(inputs=input_image, outputs=predict)
    return model
