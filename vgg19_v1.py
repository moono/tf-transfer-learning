import os
import shutil
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


# ======================================================================================================================
# data loader
# ======================================================================================================================
def parse_tfrecord(raw_record):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/class/label': tf.FixedLenFeature((), tf.int64),
    }

    # parse feature
    parsed = tf.parse_single_example(raw_record, keys_to_features)

    label = tf.cast(parsed['image/class/label'], tf.int32)

    image = tf.image.decode_png(parsed['image/encoded'], channels=1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label


def data_input_fn(data_fn, n_images, is_training, num_epochs, batch_size):
    dataset = tf.data.TFRecordDataset(data_fn)

    if is_training:
        dataset = dataset.shuffle(buffer_size=n_images)

    dataset = dataset.map(parse_tfrecord)
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
    inputs = tf.reshape(features['x'], shape=[-1, 28, 28, 1])
    inputs = tf.tile(inputs, multiples=[1, 1, 1, 3])
    inputs = tf.image.resize_bilinear(inputs, size=[input_size, input_size])

    # prepare pretrained network
    base_model = tf.keras.applications.vgg19.VGG19(include_top=False,
                                                   weights='imagenet',
                                                   input_tensor=inputs,
                                                   input_shape=(input_size, input_size, 3),
                                                   pooling=None)

    # base_model.output == base_model.layers[22].output: [batch_size, 512]
    start_from = base_model.output
    start_from = tf.layers.flatten(start_from)

    # stack more layers
    fc6 = tf.layers.dense(start_from, units=1024, activation=tf.nn.relu)
    fc6 = tf.layers.dropout(fc6, rate=0.4, training=is_training)
    logits = tf.layers.dense(fc6, units=n_output_class)

    # ================================
    # prediction & serving mode
    # mode == tf.estimator.ModeKeys.PREDICT == 'infer'
    # ================================
    predictions = {
        'output_classes': tf.cast(tf.argmax(logits, axis=1), dtype=tf.int32, name='output_class'),
        'probabilities': tf.nn.softmax(logits, name='probs'),
    }
    # export output must be one of tf.estimator.export. ... class NOT a Tensor
    export_outputs = {
        'output_classes': tf.estimator.export.PredictOutput(predictions['output_classes']),
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # ================================
    # training mode
    # ================================
    onehot_labels = tf.one_hot(labels, depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_ops = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_ops)

    # ================================
    # evaluation mode
    # ================================
    # int32_labels = tf.cast(onehot_labels, dtype=tf.int32)
    # correct_prediction = tf.cast(tf.equal(int32_labels, predictions['output_classes']), dtype=tf.float32)
    eval_metric_ops = {
        # 'accuracy': tf.reduce_mean(correct_prediction),
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['output_classes'])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def train(params):
    # clear saved model directory
    if os.path.isdir(params['model_dir']):
        shutil.rmtree(params['model_dir'])

    # create the Estimator
    model = tf.estimator.Estimator(model_fn=model_fn,
                                   model_dir=params['model_dir'],
                                   config=None,
                                   params=params,
                                   warm_start_from=None)

    # train model
    model.train(
        input_fn=lambda: data_input_fn(params['train_list'], params['n_train'], True, params['epochs'], params['batch_size']),
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
    eval_results = model.evaluate(input_fn=lambda: data_input_fn(params['eval_list'], params['n_eval'], False, 1, 1))
    print(eval_results)
    return


def main():
    # parameters
    params = {
        'train_list': ['./data/mnist-train-00.tfrecord', './data/mnist-train-01.tfrecord'],
        'eval_list': ['./data/mnist-val-00.tfrecord', './data/mnist-val-01.tfrecord'],
        'n_train': 55000,
        'n_eval': 10000,
        'model_dir': './models',
        'batch_size': 50,
        'epochs': 20,
        'input_size': 224,
        'n_output_class': 10,
    }

    # train(params)
    test(params)
    return


if __name__ == '__main__':
    main()
