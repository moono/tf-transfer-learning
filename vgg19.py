import os
import shutil
import tensorflow as tf
from tensorflow.python.keras.backend import clear_session

from flower_data_utils import data_input_fn_keras_vgg19

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

    # train model
    model.train(
        input_fn=lambda: data_input_fn_keras_vgg19(args['train_list'], args['n_train'], True, args['epochs'],
                                                   args['batch_size'], args['input_size']),
        hooks=None,
        steps=None
    )
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
        input_fn=lambda: data_input_fn_keras_vgg19(args['eval_list'], args['n_val'], False, 1, 1, args['input_size'])
    )
    print(eval_results)
    return



# def get_trained_variable(args):
#     # create the Estimator
#     model = tf.estimator.Estimator(
#         model_fn=model_fn,
#         model_dir=args['model_dir'],
#         config=None,
#         params={
#             'input_size': args['input_size'],
#             'n_output_class': args['n_output_class'],
#         },
#         # warm_start_from=args['model_dir']
#         warm_start_from=None
#     )
#
#     # belowe raises
#     # ValueError: If the Estimator has not produced a checkpoint yet.
#     var_list = model.get_variable_names()
#
#     trained_block1_conv1_kernel = model.get_variable_value('block1_conv1/kernel')
#     trained_block1_conv1_bias = model.get_variable_value('block1_conv1/bias')
#     return trained_block1_conv1_kernel, trained_block1_conv1_bias
#
#
# def get_original_variables(args):
#     input_size = args['input_size']
#     inputs = tf.placeholder(tf.float32, shape=[None, input_size, input_size, 3])
#
#     # prepare pretrained network
#     base_model = tf.keras.applications.vgg19.VGG19(include_top=False,
#                                                    weights='imagenet',
#                                                    input_tensor=inputs,
#                                                    input_shape=(input_size, input_size, 3),
#                                                    pooling=None)
#     base_model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
#                        loss='categorical_crossentropy',
#                        metric='accuracy')
#     model = tf.keras.estimator.model_to_estimator(keras_model=base_model)
#
#     var_list = model.get_variable_names()
#     original_block1_conv1_kernel = model.get_variable_value('block1_conv1/kernel')
#     original_block1_conv1_bias = model.get_variable_value('block1_conv1/bias')
#
#     return original_block1_conv1_kernel, original_block1_conv1_bias
#
#
# def check_variables(args):
#     trained_block1_conv1_kernel, trained_block1_conv1_bias = get_trained_variable(args)
#     original_block1_conv1_kernel, original_block1_conv1_bias = get_original_variables(args)
#
#     diff_conv1_kernel = original_block1_conv1_kernel - trained_block1_conv1_kernel
#     diff_conv1_bias = original_block1_conv1_bias - trained_block1_conv1_bias
#     return


def main():
    # program arguments
    args = {
        'model_dir': './models/flower-vgg19',
        'train_list': ['./data/flower-train.tfrecord'],
        'eval_list': ['./data/flower-val.tfrecord'],
        'n_train': 3260,
        'n_val': 410,
        'batch_size': 20,
        'epochs': 1000,
        'input_size': 224,
        'n_output_class': 5,
    }

    train(fresh_training=True, args=args)

    # due to bug, must clear session before re invoking estimator with keras application module
    # https://github.com/tensorflow/tensorflow/issues/14356
    # https://stackoverflow.com/questions/46911596/why-does-tensorflow-say-a-tensor-is-not-an-element-of-this-graph-when-training-a
    clear_session()
    test(args)

    # clear_session()
    # check_variables(args)
    return


if __name__ == '__main__':
    main()
