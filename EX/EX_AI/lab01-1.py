'''import tensorflow as tf
print(tf.__version__)
a = 3
b = 4
c = a + b
print(c)'''

import tensorflow as tf
a = tf.constant(3)
b = tf.constant(4)
c = a + b
print(c)

''''#텐서플로우 실행
import tensorflow as tf_new
tf = tf_new.compat.v1
g = ft.Graph()
with g.as_default9) as graph:
    hello = tf.constant("Hello TensorFlow!")
    sess = tf.Session()
    print(sess.run(hello))
#3
import tensorflow as tf_new
tf = tf_new.compat.v1
g = tf.Graph()
with g.as_default() as graph:
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0, tf.float32)
    node3 = tf.add(node1, node2)
    sess = tf.Session()
    print("node1:",sess.run(node1))'''