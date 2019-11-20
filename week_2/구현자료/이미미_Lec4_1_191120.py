#maxtrix사용 안했을때#
import tensorflow as tf
tf.compat.v1.enable_eager_execution()#그래프를 생성하지 않고 함수를 바로 실행하는 명령형 프로그래밍 환경
#난수 생성 초기값 부여
tf.compat.v1.set_random_seed(0)#set_random_seed를 통해 모든 random value generation function들이 매번 같은 값을 반환함

x1 = [73., 93., 89., 96., 73.]
x2 = [80., 88., 91., 98., 66.]
x3 = [75., 93., 90., 100., 70.]
Y = [152., 185., 180., 196., 142.]

# random weights
#초기값 1로 지정
w1 = tf.Variable(tf.random.normal([1]))#w1,w2,w3 전부 지정
w2 = tf.Variable(tf.random.normal([1]))
w3 = tf.Variable(tf.random.normal([1]))
b  = tf.Variable(tf.random.normal([1]))

learning_rate = 0.000001

for i in range(1000 + 1):
    # tf.GradientTape() to record the gradient of the cost function
    with tf.GradientTape() as tape:#cost함수의 미분값을 tf.GradientTape()에 기록
        hypothesis = w1 * x1 + w2 * x2 + w3 * x3 + b
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
        #변수들을 tape에 기록
    #tape.gradient를 호출하여 w1,w2,w3,b에 대한 기울기 값을 구함
    w1_grad, w2_grad, w3_grad, b_grad = tape.gradient(cost, [w1, w2, w3, b])

    # update w1,w2,w3 and b
    #w-(learning_Rate*(w_grad)
    w1.assign_sub(learning_rate * w1_grad)#값을 변수마다 각각 할당해야함
    w2.assign_sub(learning_rate * w2_grad)
    w3.assign_sub(learning_rate * w3_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 50 == 0:
        print("{:5} | {:12.4f}".format(i, cost.numpy()))



ㅁ