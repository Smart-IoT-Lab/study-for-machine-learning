import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import os

tf.enable_eager_execution()

#set hyper parameters
learning_rate = 0.001
training_epochs = 10
batch_size = 100

tf.set_random_seed(777)

cur_dir = os.getcwd()
ckpt_dir_name = 'checkpoints'
model_dir_name = 'minst_cnn_seq'

checkpoint_dir = os.path.join(cur_dir, ckpt_dir_name, model_dir_name)
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_prefix = os.path.join(checkpoint_dir, model_dir_name)

#make a data pipelining
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.astype(np.float32) / 255.
test_images = test_images.astype(np.float32) / 255.

train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)#차원을 추가

train_labels = to_categorical(train_labels, 10)#원핫인코딩
test_labels = to_categorical(test_labels, 10)

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(
    buffer_size=100000).batch(batch_size)#데이타 잘라 공급해주는 부분 batch_size만큼
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

def create_model():
    model = keras.Sequential()#sequential api를 쓰겠다=layer를 쌓아가면서 사용
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation=tf.nn.relu, padding='SAME',
                                  input_shape=(28, 28, 1)))#첫번째에만 input_shae사용
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu, padding='SAME'))
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=3, activation=tf.nn.relu, padding='SAME'))
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    model.add(keras.layers.Flatten())#벡터를 쭉 펴줌줌
    model.add(keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(10))#입력과 출력을 모두 연결해줌
    return model

model = create_model()
model.summary()


def loss_fn(model, images, labels):#모델과 이미지와 정답
    logits = model(images, training=True)#training=true이면 dropout이 적용
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=labels))
    return loss

def grad(model, images, labels):
    with tf.GradientTape() as tape:#loss값을 gradienttape에 기록
        loss = loss_fn(model, images, labels)
    return tape.gradient(loss, model.variables)#테이프를 거꾸로 감으면서 계산

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

#정확도 계산
def evaluate(model, images, labels):
    logits = model(images, training=False)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

checkpoint = tf.train.Checkpoint(cnn=model)

for epoch in range(training_epochs):
    avg_loss = 0.
    avg_train_acc = 0.
    avg_test_acc = 0.
    train_step = 0
    test_step = 0

    for images, labels in train_dataset:
        grads = grad(model, images, labels)
        optimizer.apply_gradients(zip(grads, model.variables))#weight들이 업데이트
        loss = loss_fn(model, images, labels)
        acc = evaluate(model, images, labels)
        avg_loss = avg_loss + loss
        avg_train_acc = avg_train_acc + acc
        train_step += 1
    avg_loss = avg_loss / train_step
    avg_train_acc = avg_train_acc / train_step

    for images, labels in test_dataset:
        acc = evaluate(model, images, labels)#한 에폭마다 정확도 측정
        avg_test_acc = avg_test_acc + acc
        test_step += 1
    avg_test_acc = avg_test_acc / test_step

    print('Epoch:', '{}'.format(epoch + 1), 'loss =', '{:.8f}'.format(avg_loss),
          'train accuracy = ', '{:.4f}'.format(avg_train_acc),
          'test accuracy = ', '{:.4f}'.format(avg_test_acc))

    checkpoint.save(file_prefix=checkpoint_prefix)














