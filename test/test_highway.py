from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from model import highway


class TestHighway(tf.test.TestCase):

    def test(self):

        with self.test_session() as sess:

            inp = tf.constant(np.array([
                [1.0, 2.0],
                [2.0, 0.0]
            ], dtype=np.float32))

            h = highway(inp, 2)

            result = sess.run(h, {
                'Highway/highway_lin_0/Matrix:0': np.array([
                    [1.0, 2.0],
                    [1.0, 2.0],
                ]),
                'Highway/highway_lin_0/Bias:0': np.array([
                    0.0,
                    0.0,
                ]),
                'Highway/highway_gate_0/Matrix:0': np.array([
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]),
                'Highway/highway_gate_0/Bias:0': np.array([
                    0.0,
                    0.0,
                ]),
            })

            self.assertAllClose(result, np.array([
                [1.47681165, 2.357608],
                [2.0, 0.238406],
            ]))
