import sys 
import numpy as np
import tensorflow as tf

def mnist_load():
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

    train_x = train_x.astype('float32') / 255.
    train_y = train_y.astype('int32')

    test_x = test_x.astype('float32') / 255.
    test_y = train_y.astype('int32')

    return (train_x, train_y), (test_x, test_y)

(train_x, train_y), (test_x, test_y) = mnist_load()

train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': train_x}, y=train_y, shuffle=True, batch_size=MINIBATCH_SIZE)

NUM_STEPS = 5000
MINIBATCH_SIZ = 128

feature_columns = [tf.feature_column.numeric_column('x', shape=[28, 28])]

dnn = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[200], n_classes=10, optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.2), model_dir='./model/DNNClassifier')

dnn.train(input_fn=train_input_fn, steps=NUM_STEPS)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': test_x}, y=test_y, shuffle=False)

test_acc - dnn.evaluate(input_fn=eval_input_fn, steps=1)['accuracy']
print('test Accuracy: {}'.format(test_acc))

'''

INFO:tensorflow:Calling model_fn.
  INFO:tensorflow:Done calling model_fn.
  INFO:tensorflow:Starting evaluation
  INFO:tensorflow:Graph was finalized.
  INFO:tensorflow:Restoring parameters from ./model/model.ckpt-119
  INFO:tensorflow:Running local_init_op.
  INFO:tensorflow:Done running local_init_op.
  INFO:tensorflow:Evaluation [1/1]
  INFO:tensorflow:Finished evaluation 
  INFO:tensorflow:Saving dict for global step 119: accuracy = 0.9453125, average_loss = 0.2138238, global_step = 119, loss = 27.369446
  test accuracy: 0.9453125

'''