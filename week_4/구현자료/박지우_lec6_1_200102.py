import tensorflow as tf
import numpy as np
tf.enable_eager_execution()
tf.set_random_seed(777)  # for reproducibility
tfe = tf.contrib.eager

x_data = [[1, 2, 1, 1], [2, 1, 3, 2],[3, 1, 3, 4],[4, 1, 5, 5],
          [1, 7, 5, 5],[1, 2, 5, 6],[1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1],[0, 0, 1],[0, 0, 1],[0, 1, 0],[0, 1, 0],
          [0, 1, 0],[1, 0, 0],[1, 0, 0]]

x_data = np.asarray(x_data, dtype=np.float32)
y_data = np.asarray(y_data, dtype=np.float32)

nb_classes = 3 #3가지의 클래스 #one hot encoding 사용하기 위해서

#Weight and bias setting
W = tfe.Variable(tf.random_normal([4, nb_classes]), name='weight')#4개의 특징값
b = tfe.Variable(tf.random_normal([nb_classes]), name='bias')
variables = [W, b]

#print("w=",W,"\nb=",  b)

# softmax = exp(logits) / reduce_sum(exp(logits), dim)
def hypothesis(X):
    return tf.nn.softmax(tf.matmul(X, W) + b)#확률값으로 변환하는 과정


#x는 입력값 w는 가중치
#print(hypothesis(x_data))

##############################

# Softmax onehot test
sample_db = [[8,2,1,4]]
sample_db = np.asarray(sample_db, dtype=np.float32)

print(hypothesis(sample_db))
###############################

def cost_fn(X, Y):
    logits = hypothesis(X)
    cost = -tf.reduce_sum(Y * tf.log(logits), axis=1)#-y*log(y_hat)
    cost_mean = tf.reduce_mean(cost)# 평균값을 구함
    return cost_mean

#print(cost_fn(x_data, y_data))

def grad_fn(X, Y):#경사하강법 사용
    with tf.GradientTape() as tape:
        loss = cost_fn(X, Y)
        grads = tape.gradient(loss, variables)

        return grads

#print(grad_fn(x_data, y_data))

#학습 함수
def fit(X, Y, epochs=2000, verbose=100):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

    for i in range(epochs):
        grads = grad_fn(X, Y)
        optimizer.apply_gradients(zip(grads, variables))
        if (i == 0) | ((i + 1) % verbose == 0):
            print('Loss at epoch %d: %f' % (i + 1, cost_fn(X, Y).numpy()))


fit(x_data, y_data)
#예측
a=hypothesis(x_data)

print(a)
print(tf.argmax(a,1))
print(tf.argmax(y_data,1))
