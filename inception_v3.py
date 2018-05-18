# https://medium.com/@utsumukiMutsuki/using-inception-v3-from-tensorflow-hub-for-transfer-learning-a931ff884526
# https://www.tensorflow.org/hub/fine_tuning
# https://github.com/tensorflow/hub/issues/24
# https://stackoverflow.com/questions/37107223/how-to-add-regularizations-in-tensorflow

import os
import shutil
import tensorflow as tf
import tensorflow_hub as hub

from flower_data_utils import data_input_fn_inception_v3

# # can set custom caching directory if needed
# os.environ['TFHUB_CACHE_DIR'] = 'C:\\Users\\moono.song\\Desktop\\base-model'

tf.logging.set_verbosity(tf.logging.INFO)


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
            input_fn=lambda: data_input_fn_inception_v3(args['train_list'], args['n_train'], True, step,
                                                        args['batch_size'], args['input_size']),
        )

        # evaluate model
        eval_results = model.evaluate(input_fn=lambda: data_input_fn_inception_v3(args['eval_list'], args['n_val'],
                                                                                  False, 1, 1, args['input_size']))
        print(eval_results)
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
        input_fn=lambda: data_input_fn_inception_v3(args['eval_list'], args['n_val'], False, 1, 1, args['input_size'])
    )
    print(eval_results)
    return


def main():
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
