# https://medium.com/@utsumukiMutsuki/using-inception-v3-from-tensorflow-hub-for-transfer-learning-a931ff884526
# https://www.tensorflow.org/hub/fine_tuning
# https://github.com/tensorflow/hub/issues/24
# https://stackoverflow.com/questions/37107223/how-to-add-regularizations-in-tensorflow

import os
import shutil
import tensorflow as tf
import tensorflow_hub as hub

# # can set custom caching directory if needed
# os.environ['TFHUB_CACHE_DIR'] = 'C:\\Users\\moono.song\\Desktop\\base-model'

tf.logging.set_verbosity(tf.logging.INFO)


# ======================================================================================================================
# data loader
# ======================================================================================================================
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


def preprocess_fn(image, label, is_training, crop_size):
    if is_training:
        image = random_rescale_image(image)
        image = random_crop_or_pad_image(image, crop_size)
        image = random_flip_left_right_image(image)
    else:
        # image = random_crop_or_pad_image(image, crop_size)
        image = tf.image.resize_images(image, size=[crop_size, crop_size], method=tf.image.ResizeMethod.BILINEAR)

    image.set_shape([crop_size, crop_size, 3])

    return image, label


def parse_tfrecord(raw_record):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'label': tf.FixedLenFeature((), tf.int64),
    }

    # parse feature
    parsed = tf.parse_single_example(raw_record, keys_to_features)

    label = tf.cast(parsed['label'], tf.int32)

    # below will decode to [0 ~ 255], uint8
    image = tf.image.decode_jpeg(parsed['image/encoded'], channels=3)

    # make 0.0 ~ 1.0: tensorflow_hub assumes inputs are normalized [0~1], float32
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    return image, label


def data_input_fn(filename, n_images, is_training, num_epochs, batch_size, crop_size):
    dataset = tf.data.TFRecordDataset(filename)

    if is_training:
        dataset = dataset.shuffle(buffer_size=n_images)

    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.map(lambda image, label: preprocess_fn(image, label, is_training, crop_size))
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    features = {
        'x': images,
    }
    return features, labels


# ======================================================================================================================
# model function
# ======================================================================================================================
def model_fn(features, labels, mode, params):
    # ================================
    # common operations for all modes
    # ================================
    input_size = params['input_size']
    n_output_class = params['n_output_class']
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    # preprocess input features for example
    inputs = tf.reshape(features['x'], shape=[-1, input_size, input_size, 3])

    # module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1",
    #                     trainable=True, tags={'train'})
    module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1")

    # outputs: [batch_size, 2048]
    start_from = module(inputs)

    # stack more layers
    # regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    # new_layer = tf.layers.dense(start_from, units=1024, activation=tf.nn.relu, kernel_regularizer=regularizer)
    # new_layer = tf.layers.dropout(new_layer, rate=0.4, training=is_training)
    # logits = tf.layers.dense(new_layer, units=n_output_class, activation=None, kernel_regularizer=regularizer)
    logits = tf.layers.dense(start_from, units=n_output_class, activation=None)

    # ================================
    # prediction & serving mode
    # mode == tf.estimator.ModeKeys.PREDICT == 'infer'
    # ================================
    predicted_classes = tf.argmax(logits, axis=1)
    predictions = {
        'class_id': tf.cast(predicted_classes, dtype=tf.int32),
        'probabilities': tf.nn.softmax(logits, name='probs'),
        'logits': logits,
    }
    # export output must be one of tf.estimator.export. ... class NOT a Tensor
    export_outputs = {
        'output_classes': tf.estimator.export.PredictOutput(predictions['class_id']),
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # compute loss
    # labels: integer 0 ~ 4
    # logits: score not probability
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # # add regularization loss - important in fine tuning
    # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # loss += tf.add_n(reg_losses)

    # compute evaluation metric
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')
    metrics = {'accuracy': accuracy}  # during evaluation
    tf.summary.scalar('accuracy', accuracy[1])  # during training

    # ================================
    # evaluation mode
    # ================================
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)

    # ================================
    # training mode
    # ================================
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    #     train_ops = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    train_ops = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_ops)


def train(fresh_training, args):
    # clear saved model directory
    if fresh_training:
        if os.path.isdir(args['model_dir']):
            shutil.rmtree(args['model_dir'])

    # create run config for estimator
    run_config = tf.estimator.RunConfig(keep_checkpoint_max=3)

    # create the Estimator
    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args['model_dir'],
        config=run_config,
        params={
            'input_size': args['input_size'],
            'n_output_class': args['n_output_class'],
        },
        warm_start_from=None
    )

    # start training...
    step = 10
    for ii in range(0, args['epochs'], step):
        # train model
        model.train(
            input_fn=lambda: data_input_fn(args['train_list'], args['n_train'], True, step,
                                           args['batch_size'], args['input_size']),
        )

        # evaluate model
        eval_results = model.evaluate(input_fn=lambda: data_input_fn(args['eval_list'], args['n_val'], False, 1,
                                                                     1, args['input_size']))
        print(eval_results)


    # # train model
    # model.train(
    #     input_fn=lambda: data_input_fn(args['train_list'], args['n_train'], True, args['epochs'], args['batch_size'],
    #                                    args['input_size']),
    #     hooks=None,
    #     steps=None
    # )
    return


def test(args):
    # create the Estimator
    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args['model_dir'],
        config=None,
        params={
            'input_size': args['input_size'],
            'n_output_class': args['n_output_class'],
        },
        warm_start_from=None
    )

    # evaluate the model and print results
    eval_results = model.evaluate(
        input_fn=lambda: data_input_fn(args['eval_list'], args['n_val'], False, 1, 1, args['input_size'])
    )
    print(eval_results)
    return


def image_feeder_test():
    import numpy as np

    filenames_tensor = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filenames_tensor)
    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.map(lambda image, label: preprocess_fn(image, label, False, 299))
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

                print('')
            except tf.errors.OutOfRangeError:
                print('End of dataset')
                break
    return


def main():
    # image_feeder_test()

    # program arguments
    args = {
        'model_dir': './models/flower-inception-v3',
        'train_list': ['./data/flower-train.tfrecord'],
        'eval_list': ['./data/flower-val.tfrecord'],
        'n_train': 3260,
        'n_val': 410,
        'batch_size': 40,
        'epochs': 100,
        'input_size': 299,
        'n_output_class': 5,
    }

    train(fresh_training=True, args=args)

    test(args)
    return


if __name__ == '__main__':
    main()
