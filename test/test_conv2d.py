from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from model import conv2d


class TestConv2D(tf.test.TestCase):

    def test(self):

        with self.test_session() as sess:

            inp = tf.constant(np.array([[
                [[1.0], [2.0]],
                [[2.0], [0.0]]
            ]], dtype=np.float32))

            h = conv2d(inp, 1, 1, 2)

            result = sess.run(h, {
                'conv2d/w:0': np.array([[[
                        [1.0],
                    ], [
                        [-1.0],
                    ]
                ]]),
                'conv2d/b:0': np.array([ 1.5 ])
            })

            self.assertAllClose(result, np.array([[
                [[0.5]],
                [[3.5]],
            ]]))
