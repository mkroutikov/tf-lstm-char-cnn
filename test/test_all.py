from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import model

from read_param_init import SOFTMAX_W, SOFTMAX_B, LSTM_W, LSTM_B, EMBEDDING, KERNEL_1_W, KERNEL_1_B

X = np.array([
 [[ 1,  3,  4,  5,  2,  0,  0,  0,  0,  0,  0],
  [ 1,  6,  3,  7,  8,  7,  9, 10,  4,  2,  0],
  [ 1,  6,  4,  5, 11, 12, 10, 13,  2,  0,  0]],

 [[ 1, 11,  4, 10,  2,  0,  0,  0,  0,  0,  0],
  [ 1, 10, 22,  4,  2,  0,  0,  0,  0,  0,  0],
  [ 1, 23,  9, 11, 11,  3,  5,  2,  0,  0,  0]],

 [[ 1,  9,  5,  2,  0,  0,  0,  0,  0,  0,  0],
  [ 1, 11,  9,  3,  7,  2,  0,  0,  0,  0,  0],
  [ 1, 10,  9,  2,  0,  0,  0,  0,  0,  0,  0]],

 [[ 1, 30,  2,  0,  0,  0,  0,  0,  0,  0,  0],
  [ 1, 30,  2,  0,  0,  0,  0,  0,  0,  0,  0],
  [ 1,  3,  7, 23,  2,  0,  0,  0,  0,  0,  0]]
], dtype=np.int32)

Y = np.array([[   2,    3,    4],
       [  32,  429, 7408],
       [3078,   64,   35],
       [  27,   48,  395]], dtype=np.int32)

class TestRNN(tf.test.TestCase):

    def model(self):
        m = model.inference_graph(char_vocab_size=51, word_vocab_size=10000,
                        char_embed_size=3, batch_size=4, num_highway_layers=0,
                        num_rnn_layers=1, rnn_size=5, max_word_length=11,
                        kernels= [2], kernel_features=[2], num_unroll_steps=3,
                        dropout=0.0)
        m.update(model.loss_graph(m.logits, batch_size=4, num_unroll_steps=3))

        return m

    def test(self):

        with self.test_session() as sess:

            m = self.model()

            num_unroll_steps=3

            feed = {
                'Embedding/char_embedding:0': EMBEDDING,
                'TDNN/kernel_2/w:0': np.reshape(np.transpose(KERNEL_1_W), [1, 2, num_unroll_steps, 2]),
                'TDNN/kernel_2/b:0': KERNEL_1_B,
                'LSTM/RNN/BasicLSTMCell/Linear/Matrix:0': LSTM_W,
                'LSTM/RNN/BasicLSTMCell/Linear/Bias:0': LSTM_B,
                'LSTM/WordEmbedding/SimpleLinear/Matrix:0': SOFTMAX_W,
                'LSTM/WordEmbedding/SimpleLinear/Bias:0': SOFTMAX_B,
                m.input:   X,
                m.targets: Y,
            }

            loss = sess.run(m.loss, feed)

            print(loss)
            assert False
