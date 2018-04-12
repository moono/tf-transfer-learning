import os
import numpy as np
import tensorflow as tf
import cv2


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def dict_to_tf_example(image_path, label_val):
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(encoded_jpg),
        'label': int64_feature(label_val),
    }))
    return example


def create_flower_data():
    # set parameters
    output_dir = './data'

    # get the images
    flower_db_base = 'd:\\db\\flower_photos'
    out = [t for t in os.walk(flower_db_base)]

    train_val_percentage = 0.9
    train_tfrecord_fn = os.path.join(output_dir, 'flower-train.tfrecord')
    val_tfrecord_fn = os.path.join(output_dir, 'flower-val.tfrecord')

    # open two writer
    train_writer = tf.python_io.TFRecordWriter(train_tfrecord_fn)
    val_writer = tf.python_io.TFRecordWriter(val_tfrecord_fn)

    for class_id, flower_info in enumerate(out[1:]):
        # parse info
        db_base = flower_info[0]
        img_fns = flower_info[2]
        n_image = len(img_fns)

        flower_name = os.path.split(db_base)[1]
        print('Class id {:d} assigned to {:s}'.format(class_id, flower_name))

        # count how to split train & val images
        n_train = int(n_image * train_val_percentage)
        remainder = n_train % 10
        if remainder != 0:
            n_train = n_train - remainder
        print('[{:s}] n_train: {:d}'.format(flower_name, n_train))

        for ii, fn in enumerate(img_fns):
            img_path = os.path.join(db_base, fn)
            single_example = dict_to_tf_example(img_path, class_id)

            if ii < n_train:
                train_writer.write(single_example.SerializeToString())
            else:
                val_writer.write(single_example.SerializeToString())

    train_writer.close()
    val_writer.close()
    return


def parse_tfrecord(raw_record):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'label': tf.FixedLenFeature((), tf.int64),
    }

    # parse feature
    parsed = tf.parse_single_example(raw_record, keys_to_features)

    label = tf.cast(parsed['label'], tf.int32)

    image = tf.image.decode_jpeg(parsed['image/encoded'])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label


def test_tfrecords():
    filenames_tensor = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filenames_tensor)
    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.prefetch(1)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(1)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    training_fn_list = ['./data/flower-train.tfrecord']
    validate_fn_list = ['./data/flower-val.tfrecord']

    train_total_size = 0
    val_total_size = 0
    with tf.Session() as sess:
        sess.run(iterator.initializer, feed_dict={filenames_tensor: training_fn_list})

        while True:
            try:
                image, label = sess.run(next_element)
                train_total_size += label.shape[0]
            except tf.errors.OutOfRangeError:
                print('End of dataset')
                break

        sess.run(iterator.initializer, feed_dict={filenames_tensor: validate_fn_list})

        while True:
            try:
                image, label = sess.run(next_element)
                if val_total_size == 1:
                    image = np.squeeze(image, axis=0)
                    image = (image * 255.0).astype(np.uint8)
                    image = cv2.cvtColor(image, code=cv2.COLOR_RGB2BGR)
                    cv2.imwrite('testout.png', image)
                    print('Label: {}'.format(label))
                val_total_size += label.shape[0]
            except tf.errors.OutOfRangeError:
                print('End of dataset')
                break

    print(train_total_size)
    print(val_total_size)
    return


def main():
    create_flower_data()
    test_tfrecords()
    return


if __name__ == '__main__':
    main()
