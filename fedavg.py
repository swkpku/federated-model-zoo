import functools
from typing import Dict

from absl import app
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff


def preprocess(dataset: tf.data.Dataset, num_epoch: int,
               batch_size: int) -> tf.data.Dataset:
    def batch_format_fn(element: Dict[str, tf.Tensor]):
        return (tf.expand_dims(element["pixels"], axis=-1), element["label"])

    return dataset.repeat(num_epoch).shuffle(100).batch(batch_size).map(
        batch_format_fn)


def get_input_spec(preprocess_fn, emnist_test):
    client_id = emnist_test.client_ids[0]
    return preprocess_fn(emnist_test.create_tf_dataset_for_client(client_id),
                         1, 1).element_spec


def create_original_fedavg_cnn_model(
        only_digits: bool = True) -> tf.keras.Model:
    """The CNN model used in https://arxiv.org/abs/1602.05629."""
    data_format = "channels_last"

    max_pool = functools.partial(
        tf.keras.layers.MaxPooling2D,
        pool_size=(2, 2),
        padding="same",
        data_format=data_format,
    )
    conv2d = functools.partial(
        tf.keras.layers.Conv2D,
        kernel_size=5,
        padding="same",
        data_format=data_format,
        activation=tf.nn.relu,
    )

    model = tf.keras.models.Sequential([
        conv2d(filters=32, input_shape=(28, 28, 1)),
        max_pool(),
        conv2d(filters=64),
        max_pool(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10 if only_digits else 62,
                              activation=tf.nn.softmax),
    ])

    return model


def make_federated_data(
    tff_dataset: tff.simulation.ClientData,
    num_clients_per_round: int,
    client_epoch: int,
    batch_size: int,
):
    client_ids = np.random.choice(tff_dataset.client_ids,
                                  size=num_clients_per_round,
                                  replace=False)
    print(f"sampled client_ids: {client_ids}")
    return [
        preprocess(
            tff_dataset.create_tf_dataset_for_client(client_id),
            client_epoch,
            batch_size,
        ) for client_id in client_ids
    ]


def run_experiment():
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

    input_spec = get_input_spec(preprocess, emnist_test)

    def tff_model_fn():
        fedavg_cnn = create_original_fedavg_cnn_model()
        return tff.learning.from_keras_model(
            fedavg_cnn,
            input_spec=input_spec,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

    iterative_process = tff.learning.build_federated_averaging_process(
        tff_model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02
                                                            ),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
    )

    state = iterative_process.initialize()

    for round_num in range(5):
        federated_data = make_federated_data(emnist_train,
                                             num_clients_per_round=10,
                                             client_epoch=1,
                                             batch_size=20)
        state, metrics = iterative_process.next(state, federated_data)
        print("round {:2d}, metrics={}".format(round_num, metrics))


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Expected no command-line arguments, "
                             "got: {}".format(argv))

    run_experiment()


if __name__ == "__main__":
    app.run(main)
