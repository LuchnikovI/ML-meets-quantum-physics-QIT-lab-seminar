import tensorflow as tf


@tf.function
def double_flip_sample(samples, ind):
    """Function flips two indices in a given places.
    Args:
        samples: tensor of shape (batch_size, N, 2)
        ind: int tensor of shape (2,), number of sites
            where one needs to flip indices
    Return:
        tensor of shape (batch_size, N, 2), samples with
        two flipped indices"""

    sort_ind = tf.sort(ind)
    sample1 = tf.roll(samples[:, sort_ind[0]], axis=-1, shift=1)
    sample2 = tf.roll(samples[:, sort_ind[1]], axis=-1, shift=1)
    flipped_samples = tf.concat([samples[:, :sort_ind[0]],
                                 sample1[:, tf.newaxis],
                                 samples[:, sort_ind[0] + 1:sort_ind[1]],
                                 sample2[:, tf.newaxis],
                                 samples[:, sort_ind[1] + 1:]], axis=1)
    return flipped_samples


@tf.function
def single_flip_sample(samples, ind):
    """Function flips index in a given places.
    Args:
        samples: tensor of shape (batch_size, N, 2)
        ind: int tensor of shape (), number of site
            where one needs to flip index
    Return:
        tensor of shape (batch_size, N, 2), samples with
        flipped index"""
    sample = tf.roll(samples[:, ind], axis=-1, shift=1)
    flipped_samples = tf.concat([samples[:, :ind],
                                 sample[:, tf.newaxis],
                                 samples[:, ind + 1:]], axis=1)
    return flipped_samples
