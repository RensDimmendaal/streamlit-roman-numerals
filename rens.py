import tensorflow as tf

from tensorflow.python.keras.preprocessing.image_dataset import (
    load_image as tf_load_image,
)


def load_img_32(fpath):
    return (
        tf_load_image(
            path=str(fpath),
            image_size=(32, 32),
            num_channels=3,
            interpolation="bilinear",
            smart_resize=False,
        )
        .numpy()
        .astype(int)
    )


def create_model(resnet_weights=None, compile=True, trained_weights=None):
    base_model = tf.keras.applications.ResNet50(
        input_shape=(32, 32, 3),
        include_top=False,
        weights=resnet_weights,
    )
    base_model = tf.keras.Model(
        base_model.inputs, outputs=[base_model.get_layer("conv2_block3_out").output]
    )

    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.applications.resnet.preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(10)(x)
    model = tf.keras.Model(inputs, x)

    if compile:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.0001),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    if trained_weights:
        model.load_weights(trained_weights)

    return model
