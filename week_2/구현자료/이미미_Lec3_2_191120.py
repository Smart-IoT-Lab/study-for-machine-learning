#Gradient descent
import tensorflow as tf
tf.enable_eager_execution()#그래프를 생성하지 않고 함수를 바로 실행하는 명령형 프로그래밍 환경
tf.compat.v1.set_random_seed(0)#set_random_seed를 통해 모든 random value generation function들이 매번 같은 값을 반환함

x_data = [1., 2., 3., 4.]
y_data = [1., 3., 5., 7.]

W = tf.Variable(tf.random_normal([1], -100., 100.))#1개의 난수를 최소-100 최대 100

for step in range(300):
    hypothesis = W * x_data
    cost = tf.reduce_mean(tf.square(hypothesis - y_data))

    alpha = 0.01
    #cost function를 미분한 값, ((w*x)-y)x / m
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, x_data) - y_data, x_data))
    descent = W - tf.multiply(alpha, gradient)

    W.assign(descent) #w에 descent 값을 할당

    if step % 10 == 0:
        print('{:5} | {:10.4f} | {:10.6f}'.format(step, cost.numpy(), W.numpy()[0]))


