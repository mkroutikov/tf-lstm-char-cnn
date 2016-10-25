from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from model import linear


class TestLinear(tf.test.TestCase):

    def test(self):

        with self.test_session() as sess:

            m = tf.constant(np.array([
                [1.0, 2.0],
                [2.0, 0.0]
            ], dtype=np.float32))

            l = linear(m, 4)

            result = sess.run(l, {
                'SimpleLinear/Matrix:0': np.array([
                    [1.0, 2.0],
                    [1.0, 2.0],
                    [1.0, 2.0],
                    [1.0, 2.0],
                ]),
                'SimpleLinear/Bias:0': np.array([
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                ]),
            })

            self.assertAllClose(result, np.array([
                [5.0, 6.0, 7.0, 8.0],
                [2.0, 3.0, 4.0, 5.0],
            ]))
            print(result)
