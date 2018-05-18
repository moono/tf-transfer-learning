import os
import shutil
import tensorflow as tf
import tensorflow_hub as hub

from flower_data_utils import data_input_fn_tf_hub

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
    initial_lr = params['initial_lr']
    batch_size = params['batch_size']
    n_train = params['n_train']

    # preprocess input features for example
    inputs = tf.reshape(features['x'], shape=[-1, input_size, input_size, 3])

    module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/1",
                        trainable=True, tags={'train'})

    # outputs: [batch_size, 2048]
    start_from = module(inputs)
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

    # add regularization loss - important in fine tuning
    # below will return regularization losses (acts same)
    # tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # tf.losses.get_regularization_losses()
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss += tf.add_n(reg_losses)

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

    # single_epoch_step = n_train // batch_size
    # learning_rate = tf.train.exponential_decay(learning_rate=initial_lr,
    #                                            global_step=tf.train.get_global_step(),
    #                                            decay_steps=single_epoch_step * 2,
    #                                            decay_rate=0.94)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, epsilon=1.0)

    optimizer = tf.train.AdamOptimizer(learning_rate=initial_lr)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
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
            'batch_size': args['batch_size'],
            'n_train': args['n_train'],
            'initial_lr': args['initial_lr'],
        },
        warm_start_from=None
    )

    # start training...
    for ii in range(args['epochs']):
        # train model
        model.train(
            input_fn=lambda: data_input_fn_tf_hub(args['train_list'], args['n_train'], True, 1,
                                                  args['batch_size'], args['input_size']),
        )

        # evaluate model
        eval_results = model.evaluate(input_fn=lambda: data_input_fn_tf_hub(args['eval_list'], args['n_val'],
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
            'batch_size': args['batch_size'],
            'n_train': args['n_train'],
            'initial_lr': args['initial_lr'],
        },
        warm_start_from=None
    )

    # evaluate the model and print results
    eval_results = model.evaluate(
        input_fn=lambda: data_input_fn_tf_hub(args['eval_list'], args['n_val'], False, 1, 1, args['input_size'])
    )
    print(eval_results)
    return


def main():
    # program arguments
    args = {
        'model_dir': './models/flower-resnet-v2-101',
        'train_list': ['./data/flower-train.tfrecord'],
        'eval_list': ['./data/flower-val.tfrecord'],
        'n_train': 3260,
        'n_val': 410,
        'batch_size': 40,
        'epochs': 50,
        'input_size': 224,
        'n_output_class': 5,
        'initial_lr': 0.0001,
    }

    train(fresh_training=True, args=args)

    test(args)
    return


if __name__ == '__main__':
    main()
