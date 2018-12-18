# --*coding--:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

rng = np.random

# 参数设定
learning_rate = 0.01
training_epochs = 10000
display_step = 100

# 训练数据
train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                      7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                      2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
print("train_X:", train_X)
print("train_Y:", train_Y)

# 设置placeholder
X = tf.placeholder("float")
Y = tf.placeholder("float")
n_samples = tf.placeholder("float")

# 设置模型的权重和偏置
W = tf.Variable(rng.randn(), name="Weight")
B = tf.Variable(rng.randn(), name="Bias")

# 设置线性回归的方程
pred = tf.add(tf.multiply(X, W), B)

# 设置cost为均方差
cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples)

# 梯度下降d
# 注意，minimize() 可以自动修正w和b，因为默认设置Variables的trainable=True
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 初始化所有variables
init = tf.global_variables_initializer()

# 开始训练
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        # 灌入所有训练数据
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y, n_samples: train_X.shape[0]})

        # 打印出每步长迭代的log日志
        if (epoch + 1) % display_step == 0:
            epoch_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y, n_samples: train_X.shape[0]})
            print("Epoch:", '%4d' % (epoch + 1), " cost:", '{:.9f}'.format(epoch_cost),
                  " Weight:", sess.run(W), " Bias:", sess.run(B))

        # if (epoch + 1) % 1000 == 0:
        #     plt.plot(train_X, train_Y, 'ro', label="Original Data")
        #     plt.plot(train_X, sess.run(pred, feed_dict={X: train_X}), label="Fitter line")
        #     plt.legend()
        #     plt.show()

    print("Optimization Finished!")
    train_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y, n_samples: train_X.shape[0]})
    print("Train cost:", '{:.9f}'.format(train_cost), " Weight:", sess.run(W), " Bias:", sess.run(B))

    # 作图
    plt.plot(train_X, train_Y, 'ro', label="Original Data")
    plt.plot(train_X, sess.run(pred, feed_dict={X: train_X}), label="Fitter line")
    plt.legend()
    plt.show()

    # 测试样本
    test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("Optimization Finished!")
    test_cost = sess.run(cost, feed_dict={X: test_X, Y: test_Y, n_samples: test_X.shape[0]})
    print("Test cost:", '{:.9f}'.format(test_cost), " Weight:", sess.run(W), " Bias:", sess.run(B))

    # 作图
    plt.plot(test_X, test_Y, 'bo', label="Test Data")
    plt.plot(train_X, sess.run(pred, feed_dict={X: train_X}), label="Fitter line")
    plt.legend()
    plt.show()

exit(0)
