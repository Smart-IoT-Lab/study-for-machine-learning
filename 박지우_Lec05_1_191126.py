import tensorflow.contrib.eager as tfe
import tensorflow as tf

tf.enable_eager_execution()

x_train = [[1.,2.],[2.,3.],[3.,1.]]
y_train = [[0.],[0.],[1.]]

x_test= [[5,2]]
y_test = [1]

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))

W = tf.Variable(tf.zeros([2.,1.]), name='weight',dtype='float32')
b = tf.Variable(tf.zeros([1.]), name='bias')

def logistic_regression(features):
    hypothesis = tf.div(1., 1. + tf.exp(tf.matmul(features, W) + b))
    return hypothesis
def loss_fn(hypothesis, labels):
    cost = -tf.reduce_mean(labels * tf.log(hypothesis) + (1 - labels) * tf.log(1 - hypothesis))
    return cost
def grad(cost):
    with tf.GradientTape() as tape:
        return tape.gradient(cost, [W,b])

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

learning_rate = 0.1
EPOCHS = 1001
for step in range(EPOCHS):
    for features, labels in tfe.Iterator(dataset):
        grads = grad(loss_fn(logistic_regression(features),labels))
        #optimizer.apply_gradients(grads_and_vars=(grads))
        with tf.GradientTape() as tape:
            cost = loss_fn(logistic_regression(features),labels)
    
        w_grad, b_grad = tape.gradient(cost,[W,b])

        if step % 100 == 0:
            print(step, w_grad.numpy()[1],b_grad.numpy())

        W.assign_sub(learning_rate * w_grad) #Variable ! numpy로 출력
        b.assign_sub(learning_rate * b_grad)
#def accuracy_fn(hypothesis, labels):
#    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
#    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32))
#    return accuracy


#test_acc = accuracy_fn(logistic_regression(x_test),y_test)
