import os
import shutil
import tensorflow as tf
from tensorflow.python.keras.backend import clear_session

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


def mean_image_subtraction(image, means):
    num_channels = image.get_shape().as_list()[-1]
    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)


def preprocess_fn(image, label, is_training, crop_size, image_mean):
    if is_training:
        image = random_rescale_image(image)
        image = random_crop_or_pad_image(image, crop_size)
        image = random_flip_left_right_image(image)
    else:
        image = random_crop_or_pad_image(image, crop_size)
        # image = tf.image.resize_images(image, size=[crop_size, crop_size], method=tf.image.ResizeMethod.BILINEAR)

    image = mean_image_subtraction(image, image_mean)
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

    image = tf.image.decode_jpeg(parsed['image/encoded'], channels=3)
    image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
    image.set_shape([None, None, 3])
    return image, label


def data_input_fn(filename, n_images, is_training, num_epochs, batch_size, crop_size, image_mean):
    dataset = tf.data.TFRecordDataset(filename)

    if is_training:
        dataset = dataset.shuffle(buffer_size=n_images)

    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.map(lambda image, label: preprocess_fn(image, label, is_training, crop_size, image_mean))
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

    # prepare pretrained network
    # 1. remove layers after convolution operations
    # 2. initialize weights with pretrained on 'imagenet'
    # 3. change input layer with ours and set appropriate input shape as well
    # 4. do not add additional layer because we will do it our selves
    base_model = tf.keras.applications.vgg19.VGG19(include_top=False,
                                                   weights='imagenet',
                                                   input_tensor=inputs,
                                                   input_shape=(input_size, input_size, 3),
                                                   pooling=None)

    # set weights trainable even if default is trainable
    for layer in base_model.layers:
        layer.trainable = True

    # base_model.output == base_model.layers[21].output: [batch_size, 7, 7, 512]
    start_from = base_model.output
    start_from = tf.layers.flatten(start_from)

    # stack more layers
    fc6 = tf.layers.dense(start_from, units=4096, activation=tf.nn.relu)
    fc6 = tf.layers.dropout(fc6, rate=0.4, training=is_training)
    fc7 = tf.layers.dense(fc6, units=4096, activation=tf.nn.relu)
    fc7 = tf.layers.dropout(fc7, rate=0.4, training=is_training)
    logits = tf.layers.dense(fc7, units=n_output_class, activation=None)

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
    # onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=n_output_class)
    # loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

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
    train_ops = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_ops)


def train(params):
    # clear saved model directory
    if os.path.isdir(params['model_dir']):
        shutil.rmtree(params['model_dir'])

    # create run config for estimator
    run_config = tf.estimator.RunConfig(keep_checkpoint_max=3)

    # create the Estimator
    model = tf.estimator.Estimator(model_fn=model_fn,
                                   model_dir=params['model_dir'],
                                   config=run_config,
                                   params=params,
                                   warm_start_from=None)

    # train model
    model.train(input_fn=lambda: data_input_fn(params['train_list'], params['n_train'], True, params['epochs'],
                                               params['batch_size'], params['input_size'], params['image_mean']),
                hooks=None,
                steps=None)
    return


def test(params):
    # create the Estimator
    model = tf.estimator.Estimator(model_fn=model_fn,
                                   model_dir=params['model_dir'],
                                   config=None,
                                   params=params,
                                   warm_start_from=None)

    # evaluate the model and print results
    eval_results = model.evaluate(input_fn=lambda: data_input_fn(params['eval_list'], params['n_val'], False, 1, 1,
                                                                 params['input_size'], params['image_mean']))
    print(eval_results)
    return


def main():
    # parameters
    params = {
        'train_list': ['./data/flower-train.tfrecord'],
        'eval_list': ['./data/flower-val.tfrecord'],
        'n_train': 3260,
        'n_val': 410,
        'model_dir': './models/flower-vgg19',
        'batch_size': 20,
        'epochs': 1000,
        'n_output_class': 5,
        'input_size': 224,
        'image_mean': (123.68, 116.779, 103.939),
    }

    train(params)

    # due to bug, must clear session before re invoking estimator with keras application module
    # https://github.com/tensorflow/tensorflow/issues/14356
    # https://stackoverflow.com/questions/46911596/why-does-tensorflow-say-a-tensor-is-not-an-element-of-this-graph-when-training-a
    clear_session()
    test(params)
    return


if __name__ == '__main__':
    main()
