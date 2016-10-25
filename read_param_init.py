from __future__ import print_function

import torchfile
import numpy as np

EMBEDDING = torchfile.load('../lstm-char-cnn/param_init_1.t7')

KERNEL_1_W = torchfile.load('../lstm-char-cnn/param_init_2.t7')
KERNEL_1_B = torchfile.load('../lstm-char-cnn/param_init_3.t7')

LSTM_1_W = torchfile.load('../lstm-char-cnn/param_init_4.t7')
LSTM_B   = torchfile.load('../lstm-char-cnn/param_init_5.t7')
LSTM_2_W = torchfile.load('../lstm-char-cnn/param_init_6.t7')

# following manipulations make LSTM_W usable with BasicLSTMCell - need to flip some blocks to convert from Karpathy's LSTM implementation
LSTM_W = np.concatenate([LSTM_1_W, LSTM_2_W], axis=1)
a, b, c, d = np.split(LSTM_W, 4, axis=0)
LSTM_W = np.concatenate([a, d, c, b], axis=0)
LSTM_W = LSTM_W.transpose()

a, b, c, d = np.split(LSTM_B, 4)
LSTM_B = np.concatenate([a, d, c, b], axis=0)

SOFTMAX_W = torchfile.load('../lstm-char-cnn/param_init_7.t7')
SOFTMAX_B = torchfile.load('../lstm-char-cnn/param_init_8.t7')


if __name__ == '__main__':

    print(EMBEDDING)

    print(KERNEL_1_W)

    print(KERNEL_1_B)

    print(LSTM_1_W.shape)
    print(LSTM_2_W.shape)

    print(np.vstack([np.transpose(LSTM_1_W), np.transpose(LSTM_2_W)]))

    print(LSTM_B)

    '''
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x)
    local h2h = nn.Linear(rnn_size, 4 * rnn_size, false)(prev_h)
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local sigmoid_chunk = nn.Narrow(2, 1, 3*rnn_size)(all_input_sums)
    sigmoid_chunk = nn.Sigmoid()(sigmoid_chunk)
    local in_gate = nn.Narrow(2,1,rnn_size)(sigmoid_chunk)
    local out_gate = nn.Narrow(2, rnn_size+1, rnn_size)(sigmoid_chunk)
    local forget_gate = nn.Narrow(2, 2*rnn_size + 1, rnn_size)(sigmoid_chunk)
    local in_transform = nn.Tanh()(nn.Narrow(2,3*rnn_size + 1, rnn_size)(all_input_sums))

    -- perform the LSTM update
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate, in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    '''
    x = np.array([-0.04201929,  0.02275813])
    prev_h = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    prev_c = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    i2h = np.dot(LSTM_1_W, x) + LSTM_B
    h2h = np.dot(LSTM_2_W, prev_h)
    all_input_sums = i2h + h2h
    print('ALL_INPUT_SUMS', all_input_sums)
    '''
ALL_INPUT_SUMS [ 0.02735383  0.03522781 -0.03592717 -0.02283547  0.04040729
0.01193809  0.00140385 -0.01781952 -0.0431703   0.01421306
-0.02227222 -0.02860017 -0.0485126   0.02249379 -0.02521783
-0.03297023  0.00699924  0.02405969  0.03880194  0.01295331]
    '''

    sigmoid_chunk = all_input_sums[0:15]

    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    sigmoid_chunk = sigmoid(sigmoid_chunk)
    print(sigmoid_chunk)

    in_gate = sigmoid_chunk[0:5]
    out_gate = sigmoid_chunk[5:10]
    forget_gate = sigmoid_chunk[10:15]
    in_transform = all_input_sums[15:20]
    print(forget_gate, prev_c)
    print(in_gate, in_transform)

    next_c = forget_gate * prev_c + in_gate * in_transform
    print('next_c:', next_c)

    next_h = out_gate * np.tanh(next_c)
    print('next_h:', next_h)
    '''
next_c: [-0.01671056  0.00356125  0.01181377  0.01917946  0.00660749]
next_h: [-0.00840437  0.00178187  0.00585398  0.00938162  0.00332717]
    '''
