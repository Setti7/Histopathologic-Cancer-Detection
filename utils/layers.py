import tensorflow as tf


# https://keras.io/examples/cifar10_cnn_tfaugment2d/
def augment_2d(
        inputs,
        rotation_range=0,
        horizontal_flip=False,
        vertical_flip=False,
        crop=False,
        output_size=None):
    """Apply additive augmentation on 2D data.

    # Arguments
      rotation_range: A float, the degree range for rotation (0 <= rotation < 180),
          e.g. 3 for random image rotation between (-3.0, 3.0).
      horizontal_flip: A boolean, whether to allow random horizontal flip,
          e.g. true for 50% possibility to flip image horizontally.
      vertical_flip: A boolean, whether to allow random vertical flip,
          e.g. true for 50% possibility to flip image vertically.

    # Returns
      input data after augmentation, whose shape is the same as its original.
    """
    if inputs.dtype != tf.float32:
        inputs = tf.image.convert_image_dtype(inputs, dtype=tf.float32)
    
    # Only the center 32x32 region is important for the classification
    if crop:
        with tf.name_scope('cropping'):
            inputs = tf.image.crop_to_bounding_box(
                inputs,
                offset_height=32,
                offset_width=32,
                target_height=32,
                target_width=32
            )
            
    # resizing all images to the size the pre-trained model requires
    if output_size is not None:
        inputs = tf.image.resize_images(inputs, output_size)

    with tf.name_scope('augmentation'):
        shp = tf.shape(inputs)
        batch_size, height, width = shp[0], shp[1], shp[2]
        width = tf.cast(width, tf.float32)
        height = tf.cast(height, tf.float32)

        transforms = []
        identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)

        if rotation_range > 0:
            angle_rad = rotation_range * 3.141592653589793 / 180.0
            angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
            f = tf.contrib.image.angles_to_projective_transforms(angles,
                                                                 height, width)
            transforms.append(f)

        if horizontal_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            shape = [-1., 0., width, 0., 1., 0., 0., 0.]
            flip_transform = tf.convert_to_tensor(shape, dtype=tf.float32)
            flip = tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1])
            noflip = tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])
            transforms.append(tf.where(coin, flip, noflip))

        if vertical_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            shape = [1., 0., 0., 0., -1., height, 0., 0.]
            flip_transform = tf.convert_to_tensor(shape, dtype=tf.float32)
            flip = tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1])
            noflip = tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])
            transforms.append(tf.where(coin, flip, noflip))

    if transforms:
        f = tf.contrib.image.compose_transforms(*transforms)
        inputs = tf.contrib.image.transform(inputs, f, interpolation='BILINEAR')
    return inputs
