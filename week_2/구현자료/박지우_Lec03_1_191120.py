import tensorflow as tf
import numpy

tf.set_random_seed(0)

X = [1,2,3,4]
Y = [1,2,3,4]

W = tf.Variable(tf.random_normal([1],-100,100))

for step in range(100):
    hypothesis = W * X
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    alpha = 0.01
    gradient = tf.reduce_mean(tf.multiply((tf.multiply(W,X)-Y),X))
    descent = W-tf.multiply(alpha, gradient)
    W.assign(descent)

    if step % 10 ==0:
        print(step, cost, W)
        
