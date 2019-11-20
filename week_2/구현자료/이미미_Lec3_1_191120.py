#cost function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#그래프에서 즉시 실행모드로 변환
tf.compat.v1.enable_eager_execution()

X = np.array([1, 2, 3])
Y = np.array([1, 2, 3])

def cost_func(W, X, Y):
  hypothesis = X * W
  return tf.reduce_mean(tf.square(hypothesis - Y))
#-3에서 5까지를 15로 나눈다.
W_values = np.linspace(-3, 5, num=15)
cost_values = []
print(W_values)
for feed_W in W_values:
    #w값에 따라 curr_cost값이 얼마나 나오는지
    curr_cost = cost_func(feed_W, X, Y)
    cost_values.append(curr_cost)
    print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))

plt.rcParams["figure.figsize"] = (8, 6)

plt.plot(W_values, cost_values, "b")
plt.ylabel('Cost(W)')
plt.xlabel('W')
plt.show()

