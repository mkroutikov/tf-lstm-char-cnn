from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import model


class TestModel(tf.test.TestCase):

    def model(self):
        return model.inference_graph(char_vocab_size=5, word_vocab_size=5,
                        char_embed_size=3, batch_size=1, num_highway_layers=0,
                        num_rnn_layers=1, rnn_size=5, max_word_length=5,
                        kernels= [2], kernel_features=[1], num_unroll_steps=1,
                        dropout=0.0)

    def test_char_embedding_step(self):

        with self.test_session() as sess:

            m = self.model()

            input_embedded = sess.run(m.input_embedded, {

                'Embedding/char_embedding:0': np.array([
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [-1, 0, 1],
                ]),

                'input:0': np.array([[
                    [1,3,2,0,0],
                ]]),

            })

            print(input_embedded)
            self.assertAllClose(input_embedded, np.array([
                [[1,0,0], [0,0,1], [0,1,0], [0,0,0], [0,0,0]],
            ]))

    def test_cnn_step(self):

        with self.test_session() as sess:

            m = self.model()

            input_cnn = sess.run(m.input_cnn, {

                'TDNN/kernel_2/w:0': np.array([[
                    [[1], [1], [1]],
                    [[1], [1], [1]],
                ]]),
                'TDNN/kernel_2/b:0': np.array([0]),

                m.input_embedded: np.array([[
                    [1,0,0], [0,0,1], [0,1,0], [0,0,0], [0,0,0],
                ]])
            })

            self.assertAllClose(input_cnn, np.array([
                [[np.tanh(2)]],
            ]))

    def test_rnn_step(self):

        with self.test_session() as sess:

            m = self.model()

            rnn_outputs = sess.run(m.rnn_outputs, {

                'LSTM/RNN/BasicLSTMCell/Linear/Matrix:0': np.array([
                    [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]),
                'LSTM/RNN/BasicLSTMCell/Linear/Bias:0': np.array(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                ),

                m.input_cnn: np.array([
                    [[1]],
                ])
            })

            value1 = np.tanh(sigmoid(1)* np.tanh(1)) * 0.5
            value2 = np.tanh(sigmoid(0)* np.tanh(1)) * 0.5
            self.assertAllClose(rnn_outputs, np.array([
                [[value1, value2, value2, value2, value2]],
            ]))

    def test_logits(self):

        with self.test_session() as sess:

            m = self.model()

            value1 = np.tanh(sigmoid(1)* np.tanh(1)) * 0.5
            value2 = np.tanh(sigmoid(0)* np.tanh(1)) * 0.5

            logits = sess.run(m.logits, {

                'LSTM/WordEmbedding/SimpleLinear/Matrix:0': np.array([
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                ]),
                'LSTM/WordEmbedding/SimpleLinear/Bias:0': np.array([0,1,0,0,0]),

                m.rnn_outputs[0]: np.array([
                    [2, 1, 1, 1, 1],
                ])
            })

            print(logits)
            self.assertAllClose(logits, np.array([[[2, 2, 1, 1, 1]]]))

    def test_loss(self):

        with self.test_session() as sess:
            logits = tf.placeholder(tf.float32, [1, 1, 5], name='logits')
            l = model.loss_graph(logits, 1, 1)

            loss = sess.run(l.loss, {
                'logits:0': np.array([[[-10, -10, -10, -10, 10]]]),
                'Loss/targets:0': np.array([[4]])
            })

            print(loss, np.exp(loss))
            self.assertAllClose(loss, 0)

            loss = sess.run(l.loss, {
                'logits:0': np.array([[[0, 0, 0, 0, 0]]]),
                'Loss/targets:0': np.array([[0]])
            })

            print(loss, np.exp(loss))
            self.assertAllClose(loss, np.log(5))

    def test_loss_avg(self):

        with self.test_session() as sess:
            logits = tf.placeholder(tf.float32, [2, 2, 5], name='logits')
            l = model.loss_graph(logits, 2, 2)

            loss = sess.run(l.loss, {
                'logits:0': np.array([
                    [
                        [-10, -10, -10, -10, 10],
                        [-10, -10, -10, -10, 10],
                    ],
                    [
                        [-10, -10, -10, -10, 10],
                        [-10, -10, -10, -10, 10],
                    ],
                ]),
                'Loss/targets:0': np.array([[4, 4], [4, 4]])
            })

            print(loss, np.exp(loss))
            self.assertAllClose(loss, 0)

            loss = sess.run(l.loss, {
                'logits:0': np.array([
                    [
                        [-10, -10, -10, -10, 10],
                        [-10, -10, -10, -10, 10],
                    ],
                    [
                        [-10, -10, -10, -10, 10],
                        [-10, -10, -10, -10, 10],
                    ],
                ]),
                'Loss/targets:0': np.array([[0, 0], [4, 4]])
            })

            print(loss, np.exp(loss))
            self.assertAllClose(loss, 10)

    def xest(self):

        with self.test_session() as sess:

            m = model.inference_graph(char_vocab_size=5, word_vocab_size=5,
                        char_embed_size=3, batch_size=2, num_highway_layers=0,
                        num_rnn_layers=1, rnn_size=5, max_word_length=5,
                        kernels= [2], kernel_features=[2], num_unroll_steps=2,
                        dropout=0.0)

            logits, input_embedded = sess.run([
                    self.model.logits,
                    self.model.input_embedded,
                ], {
                'LSTM/RNN/BasicLSTMCell/Linear/Matrix:0': np.array([
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]),
                'LSTM/RNN/BasicLSTMCell/Linear/Bias:0': np.array(
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                ),

                'TDNN/kernel_2/w:0': np.array([[
                    [[1,1],[1,1],[1,1]],
                    [[1,1],[1,1],[1,1]]
                ]]),
                'TDNN/kernel_2/b:0': np.array([0, 0]),

                'Embedding/char_embedding:0': np.array([
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [-1, 0, 1],
                ]),

                'input:0': np.array([
                    [[1,3,2,0,0],[1,4,2,0,0]],
                    [[1,3,3,2,0],[1,4,4,2,0]]
                ]),

            })

            print(logits)
            print(input_embedded)
            self.assertAllClose(logits, np.array([
                [[0,1,0,0,0],[0,0,0,0,0]],
                [[0,0,0,0,0],[0,0,0,0,0]]
            ]))

def sigmoid(x):
    return 1. / (1. + np.exp(-x))
