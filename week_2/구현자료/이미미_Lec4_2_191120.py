#matrix사용시#
import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()#그래프를 생성하지 않고 함수를 바로 실행하는 명령형 프로그래밍 환경
tf.compat.v1.set_random_seed(0)#set_random_seed를 통해 모든 random value generation function들이 매번 같은 값을 반환함

data = np.array([
    # X1,   X2,    X3,   y
    [73., 80., 75., 152.],
    [93., 88., 93., 185.],
    [89., 91., 90., 180.],
    [96., 98., 100., 196.],
    [73., 66., 70., 142.]
], dtype=np.float32)

# slice data: ','를 기준으로 앞은 행, 뒤는 열
X = data[:, :-1]  #처음부터 끝까지, 처음부터 마지막 제외
y = data[:, [-1]] #처음부터 끝까지, 마지막 하나

W = tf.Variable(tf.random_normal([3, 1]))#3가지 변수로 1개의 출력 값
b = tf.Variable(tf.random_normal([1]))

learning_rate = 0.000001

# hypothesis, prediction function
def predict(X):
    return tf.matmul(X, W) + b #matmul(X, W): x,w의 행렬곱

print("epoch | cost")
n_epochs = 2000 #한번 도는 것을 epochs라고 함
for i in range(n_epochs + 1):
    with tf.GradientTape() as tape: #cost함수의 미분값을 tf.GradientTape()에 기록
        cost = tf.reduce_mean((tf.square(predict(X) - y)))
    #tape.gradient를호출하여 w1, w2, w3, b에대한 기울기값을 구함
    W_grad, b_grad = tape.gradient(cost, [W, b])

    W.assign_sub(learning_rate * W_grad)  # updates parameters (W and b)
    b.assign_sub(learning_rate * b_grad)

    if i % 100 == 0:
        print("{:5} | {:10.4f}".format(i, cost.numpy()))


