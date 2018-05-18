import tensorflow as tf


# tf.image.decode_jpeg()  : returns uint8 [0 ~ 255]
# tf.image.resize_images(): returns float32 data range same as input

def random_rescale_image(image):
    shape = tf.shape(image)
    height = tf.to_float(shape[0])
    width = tf.to_float(shape[1])
    scale = tf.random_uniform([], minval=0.5, maxval=2.0, dtype=tf.float32)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)
    image = tf.image.resize_images(image, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR)
    return image


def random_crop_or_pad_image(image, crop_size):
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    image_pad = tf.image.pad_to_bounding_box(image, 0, 0,
                                             tf.maximum(crop_size, image_height),
                                             tf.maximum(crop_size, image_width))
    image_crop = tf.random_crop(image_pad, [crop_size, crop_size, 3])
    return image_crop


def random_flip_left_right_image(image):
    uniform_random = tf.random_uniform([], 0, 1.0)
    mirror_cond = tf.less(uniform_random, .5)
    image = tf.cond(mirror_cond, lambda: tf.reverse(image, [1]), lambda: image)
    return image


def preprocess_keras_vgg19(image, label):
    # same as image-mean subtraction
    image = tf.keras.applications.vgg19.preprocess_input(image)
    return image, label


def preprocess_tf_hub(image, label):
    # in order to get [0. ~ 1.] and wants to use tf.image.convert_image_dtype(),
    # one needs to first convert to uint8 datatype
    image = tf.cast(image, dtype=tf.uint8)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label


def preprocess_fn(image, label, is_training, crop_size):
    if is_training:
        image = random_rescale_image(image)
        image = random_crop_or_pad_image(image, crop_size)
        image = random_flip_left_right_image(image)
    else:
        image = tf.image.resize_images(image, size=[crop_size, crop_size], method=tf.image.ResizeMethod.BILINEAR)

    return image, label


def parse_tfrecord(raw_record):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'label': tf.FixedLenFeature((), tf.int64),
    }

    # parse feature
    parsed = tf.parse_single_example(raw_record, keys_to_features)

    # get int32 label
    label = tf.cast(parsed['label'], tf.int32)

    # decode: will return uint8
    image = tf.image.decode_jpeg(parsed['image/encoded'], channels=3)
    # image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
    # image.set_shape([None, None, 3])
    return image, label


def data_input_fn_keras_vgg19(filename, n_images, is_training, num_epochs, batch_size, crop_size):
    dataset = tf.data.TFRecordDataset(filename)

    if is_training:
        dataset = dataset.shuffle(buffer_size=n_images)

    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.map(lambda image, label: preprocess_fn(image, label, is_training, crop_size))
    dataset = dataset.map(lambda image, label: preprocess_keras_vgg19(image, label))
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    features = {
        'x': images,
    }
    return features, labels


def data_input_fn_tf_hub(filename, n_images, is_training, num_epochs, batch_size, crop_size):
    dataset = tf.data.TFRecordDataset(filename)

    if is_training:
        dataset = dataset.shuffle(buffer_size=n_images)

    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.map(lambda image, label: preprocess_fn(image, label, is_training, crop_size))
    dataset = dataset.map(lambda image, label: preprocess_tf_hub(image, label))
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    features = {
        'x': images,
    }
    return features, labels


def dataset_test(is_training, im_size, additional_preprocess):
    import numpy as np

    filenames_tensor = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filenames_tensor)

    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.map(lambda images, labels: preprocess_fn(images, labels, is_training, im_size))
    dataset = dataset.map(lambda images, labels: additional_preprocess(images, labels))
    dataset = dataset.prefetch(1)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(1)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    validate_fn_list = ['./data/flower-val.tfrecord']

    with tf.Session() as sess:
        sess.run(iterator.initializer, feed_dict={filenames_tensor: validate_fn_list})

        while True:
            try:
                image, label = sess.run(next_element)
                min_img = np.amin(image)
                max_img = np.amax(image)

                print(min_img, max_img)
            except tf.errors.OutOfRangeError:
                print('End of dataset')
                break
    return


def main():
    # im_size = 224
    # is_training = False
    # additional_preprocess = preprocess_keras_vgg19
    # dataset_test(is_training, im_size, additional_preprocess)

    im_size = 299
    is_training = True
    additional_preprocess = preprocess_tf_hub
    dataset_test(is_training, im_size, additional_preprocess)
    return


if __name__ == '__main__':
    main()
