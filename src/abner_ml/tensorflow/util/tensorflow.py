import tensorflow as tf


def tf_default_device(device_num=0):
    """
    Order of precedence: TPU, GPU and CPU
    """
    device_type = "tpu" if tf.config.list_logical_devices("TPU") else "gpu" if tf.config.list_physical_devices("GPU") else "cpu"
    return f'{device_type}:{device_num}'


def tf_initialize_tpu(tpu=''):
    """
    This is the TPU initialization code that has to be at the beginning.

    :param tpu:
    :return:
    """
    try:
        # tf.distribute.cluster_resolver.TPUClusterResolver is a special address just for Colab.
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        tf.config.list_logical_devices()
    except Exception as e:
        print("Unable to initialize TPU!\n", repr(e))
