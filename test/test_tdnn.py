from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from model import tdnn


class TestTDNN(tf.test.TestCase):

    def test(self):

        with self.test_session() as sess:

            inp = tf.constant(np.array([
                [[1.0], [2.0], [2.0], [0.0]]
            ], dtype=np.float32))

            x = tdnn(inp, [2], [1])

            result = sess.run(x, {
                'TDNN/kernel_2/w:0': np.array([[[[1.0]], [[-1.0]]]]),
                'TDNN/kernel_2/b:0': np.array([1.0]),
            })

            print(result)
            self.assertAllClose(result, [[np.tanh(3.0)]])
