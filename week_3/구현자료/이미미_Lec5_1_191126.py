import tensorflow as tf
tf.enable_eager_execution()#그래프를 생성하지 않고 함수를 바로 실행하는 명령형 프로그래밍 환경
tf.set_random_seed(777)#set_random_seed를 통해 모든 random value generation function들이 매번 같은 값을 반환함

x_train = [[1., 2.], [2., 3.],[3., 1.], [4., 3.],[5., 3.], [6., 2.]]
y_train = [[0.], [0.], [0.], [1.], [1.], [1.]]

x_test = [[5.,2.]]
y_test = [[1.]]

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))#batch: 데이터를 읽어올 개수를 지정하는 함수
W = tf.Variable(tf.zeros([2,1]), name='weight')#0의 값으로 2x1의 행렬로 만듦
b = tf.Variable(tf.zeros([1]), name='bias')#tf.zero : 0값

def logistic_regression(features):#시그모이드 함수를 가설로 선언
    hypothesis  = tf.div(1., 1. + tf.exp(tf.matmul(features, W) + b)) #1/1+e^-x : tf.matmul(features, W) + b)는 linear 값, exp(=sigmoid함수)
    return hypothesis

def loss_fn(hypothesis, features, labels): #cost함수
    cost = -tf.reduce_mean(labels * tf.log(logistic_regression(features)) + (1 - labels) * tf.log(1 - hypothesis))
    return cost

def grad(hypothesis, features, labels):
    with tf.GradientTape() as tape:# cost함수의 미분값을 tf.GradientTape()에 기록
        loss_value = loss_fn(logistic_regression(features),features,labels)#cost값
    return tape.gradient(loss_value, [W,b])#tape.gradient를 호출하여 w,b에 대한 기울기 값을 구함

def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)#cast("조건")에 따라 1, 0 반환
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32))#equal비교해 boolean값 반환
    return accuracy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)#경사하강법


EPOCHS = 1001 #한번 학습하는 것을 epochs라고 함

for step in range(EPOCHS):
    for features, labels  in iter(dataset):#iter:1회학습
        grads = grad(logistic_regression(features), features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads,[W,b]))#zip: 리스트를 묶어줌
        if step % 100 == 0:
            print("Iter: {}, Loss: {:.4f}".format(step, loss_fn(logistic_regression(features),features,labels)))

test_acc = accuracy_fn(logistic_regression(x_test),y_test)
print("\nTestset Accuracy: {:.4f}".format(test_acc))



