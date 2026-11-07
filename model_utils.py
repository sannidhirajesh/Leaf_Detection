
import tensorflow as tf
from tensorflow.keras import layers, Model


def build_model(
    num_classes,
    image_size=(224, 224),
    trainable_backbone=False,
):
    """
    Leaf Image Classification Model
    Architecture: MobileNetV2 CNN + SimpleRNN
    """

    # 1. Input image
    inputs = layers.Input(
        shape=(image_size[0], image_size[1], 3),
        name="leaf_image",
    )

    # 2. Data augmentation
    x = layers.RandomFlip(
        "horizontal",
        name="random_flip",
    )(inputs)

    x = layers.RandomRotation(
        0.08,
        name="random_rotation",
    )(x)

    x = layers.RandomZoom(
        height_factor=0.10,
        width_factor=0.10,
        name="random_zoom",
    )(x)

    x = layers.RandomTranslation(
        height_factor=0.08,
        width_factor=0.08,
        name="random_translation",
    )(x)

    x = layers.RandomContrast(
        0.10,
        name="random_contrast",
    )(x)

    # 3. MobileNetV2 preprocessing: [0, 255] -> [-1, 1]
    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="mobilenet_preprocessing",
    )(x)

    # 4. Pretrained CNN feature extractor
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(
            image_size[0],
            image_size[1],
            3,
        ),
        include_top=False,
        weights="imagenet",
    )

    backbone.trainable = trainable_backbone

    # Keep BatchNormalization in inference mode
    features = backbone(x, training=False)

    # 5. Convert CNN feature maps to a sequence
    sequence = layers.Permute(
        (2, 1, 3),
        name="feature_map_permute",
    )(features)

    sequence = layers.TimeDistributed(
        layers.Flatten(),
        name="feature_rows",
    )(sequence)

    sequence = layers.TimeDistributed(
        layers.Dense(
            128,
            activation="relu",
        ),
        name="feature_projection",
    )(sequence)

    sequence = layers.LayerNormalization(
        name="sequence_normalization",
    )(sequence)

    sequence = layers.Dropout(
        0.25,
        name="sequence_dropout",
    )(sequence)

    # 6. SimpleRNN learns sequential feature patterns
    x = layers.SimpleRNN(
        64,
        activation="tanh",
        return_sequences=False,
        dropout=0.20,
        recurrent_dropout=0.0,
        name="leaf_simple_rnn",
    )(sequence)

    # 7. Classification head
    x = layers.Dense(
        128,
        activation="relu",
        name="classification_dense",
    )(x)

    x = layers.BatchNormalization(
        name="classification_batch_norm",
    )(x)

    x = layers.Dropout(
        0.40,
        name="classification_dropout",
    )(x)

    # 8. Output probability for each leaf class
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="class_probabilities",
    )(x)

    # 9. Complete model
    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="Leaf_MobileNetV2_SimpleRNN",
    )

    return model

