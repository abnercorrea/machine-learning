import tensorflow as tf


def tf_while_loop_body():
    """
    Decorator for while_loop body functions that sets the shape of loop vars to match the shape of the input tensors.
    This prevents errors related to varying tensor shape across iterations.
    This decorator eliminates the need to explicitly call tensor.set_shape() for each function arg.
    """
    def decorator(f):
        def applicator(*args):
            loop_vars = f(*args)
            # sets shape of loop variables to prevent errors related to varying shape across iterations.
            for var_new, var_old in zip(loop_vars, args):
                var_new.set_shape(var_old.get_shape())
            return loop_vars
        return applicator
    return decorator


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
