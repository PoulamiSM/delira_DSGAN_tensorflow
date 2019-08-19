import tensorflow as tf
from delira.models.gan.lpips_tf import lpips


class lpips_score:

    def __init__(self, _shape):
        g = tf.Graph()
        with g.as_default():
            self.image0_ph = tf.placeholder(shape=[*_shape], dtype=tf.float32)
            self.image1_ph = tf.placeholder(shape=[*_shape], dtype=tf.float32)
            self.distance_t = lpips(self.image0_ph, self.image1_ph, model='net-lin', net='alex')

        self.sess = tf.Session(graph=g)

    def __call__(self, real_image, fake_image):
        distance = self.sess.run(self.distance_t, feed_dict={self.image0_ph: real_image, self.image1_ph: fake_image})
        return distance








