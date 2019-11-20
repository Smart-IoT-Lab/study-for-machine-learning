import tensorflow as tf
tf.enable_eager_execution()

x1 = [73,93,89]
x2 = [70,30,47]
x3 = [78,98,88]
Y= [154,185,164]

w1 = tf.Variable(tf.random_normal([1]))
w2 = tf.Variable(tf.random_normal([1]))
w3 = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))


alpha = 0.00001

for i in range(1000+1):
    with tf.GradientTape() as tape:
        hypothesis = w1*x1 + w2*x2 + w3*x3 + b
        cost = tf.reduce_mean(tf.square(hypothesis - Y))

    w1_grad, w2_grad, w3_grad, b_grad = tape.gradient(cost, [w1,w2,w3,b])

    w1.assign_sub(alpha * w1_grad)
    w2.assign_sub(alpha * w2_grad)
    w3.assign_sub(alpha * w3_grad)
    b.assign_sub(alpha * b_grad)

    if i%100 == 0:
        print(i, cost.numpy())
