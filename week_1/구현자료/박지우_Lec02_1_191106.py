import tensorflow as tf 
#이걸 실행해야 오류가 안남 왜인지는 모
tf.enable_eager_execution()

#데이터셋
x_data = [1,2,3,4,5]
y_data = [1,2,3,4,5]

#변수 선
W = tf.Variable(2.0) 
b = tf.Variable(0.1) 

#학습을 얼마나 섬세하게할건지?
learning_rate = 0.01 


for i in range (100+1) :
    # 컨텍스트(context) 안에서 실행된 모든 연산을 테이프(tape)에 "기록"    
    with tf.GradientTape() as tape :
        # y = Wx + b 식 선언
        hypothesis = W * x_data + b
        # cost 함수 식 선언
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))

    #테이프에 "기록된" 연산의 그래디언트를 계산
    #입력 텐서 [w,b]에 대한 cost의 도함수
    #GradientTape에 포함된 리소스가 해제
    W_grad , b_grad = tape.gradient(cost,[W,b])

    #값을 감소
    #최적화된 값이 안나온다!
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)
    
    if i % 10 == 0 :
        print("{:5}|{:10.4f} |{:10.4} |{:10.6f}".format(i,W.numpy(),b.numpy(),cost))
