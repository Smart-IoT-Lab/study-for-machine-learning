import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

#그래프에서 즉시 실행모드로 변환
tf.enable_eager_execution()
# x_data와 y_data동일
x_data = [1, 2, 3, 4, 5]
y_data = [1, 2, 3, 4, 5]
# 초기값을 임의 지정
W = tf.Variable(2.9)
b = tf.Variable(0.5)

# 가설함수
hypothesis = W * x_data + b

# tf.reduce_mean() 차원이 하나 줄어들면서 평균을 구한다.
# tf.square() 넘겨받은 값을 제곱한다.
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

learning_rate = 0.01

# minimize cost(W,b)하는 알고리즘=Gradient descent(경사를 내려하면서 w,b찾음)
# 변수(W,b)들의 변화하는 정보를 tape에 기록
#for문을 통해 W,b업데이트되는 것을 보여줌
for i in range(100):
    with tf.GradientTape() as tape:
        hypothesis = W * x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))

    # tape의 gradient메소드 호출하여 경사도(=미분값)을 구해 튜플로 반환
    W_grad, b_grad = tape.gradient(cost, [W, b])

    # A.assign_sub(B)   :    A -= B
    # leaering_rate는 w_grad를 얼마큼 반영할것인가를 결정
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)
    if i% 10 == 0:
        print("{:5}|{:10.4f}|{:10.4}|{:10.6f}".format(i, W.numpy(),b.numpy(), cost))
print()

# predict
print(W * 5 + b)
print(W * 2.5 + b)