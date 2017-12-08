from __future__ import print_function
from __future__ import division


import tensorflow as tf


class adict(dict):
    ''' Attribute dictionary - a convenience data structure, similar to SimpleNamespace in python 3.3
        One can use attributes to read/write dictionary content.
    '''
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self


def conv2d(input_, output_dim, k_h, k_w, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
        b = tf.get_variable('b', [output_dim])

    return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b


def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]

    Args:
        args: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).

    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


def tdnn(input_, kernels, kernel_features, scope='TDNN'):
    '''

    :input:           input float tensor of shape [(batch_size*num_unroll_steps) x max_word_length x embed_size]
    :kernels:         array of kernel sizes
    :kernel_features: array of kernel feature sizes (parallel to kernels)
    '''
    assert len(kernels) == len(kernel_features), 'Kernel and Features must have the same size'

    max_word_length = input_.get_shape()[1]
    embed_size = input_.get_shape()[-1]

    # input_: [batch_size*num_unroll_steps, 1, max_word_length, embed_size]
    input_ = tf.expand_dims(input_, 1)

    layers = []
    with tf.variable_scope(scope):
        for kernel_size, kernel_feature_size in zip(kernels, kernel_features):
            reduced_length = max_word_length - kernel_size + 1

            # [batch_size*num_unroll_steps, 1, reduced_length, kernel_feature_size]
            conv = conv2d(input_, kernel_feature_size, 1, kernel_size, name="kernel_%d" % kernel_size)

            # [batch_size*num_unroll_steps, 1, 1, kernel_feature_size]
            pool = tf.nn.max_pool(tf.tanh(conv), [1, 1, reduced_length, 1], [1, 1, 1, 1], 'VALID')

            layers.append(tf.squeeze(pool, [1, 2]))

        if len(kernels) > 1:
            output = tf.concat(layers, 1)
        else:
            output = layers[0]

    return output


def inference_graph(char_vocab_size, word_vocab_size,
                    char_embed_size=15,
                    batch_size=20,
                    num_highway_layers=2,
                    num_rnn_layers=2,
                    rnn_size=650,
                    max_word_length=65,
                    kernels         = [ 1,   2,   3,   4,   5,   6,   7],
                    kernel_features = [50, 100, 150, 200, 200, 200, 200],
                    num_unroll_steps=35,
                    dropout=0.0):

    assert len(kernels) == len(kernel_features), 'Kernel and Features must have the same size'

    input_ = tf.placeholder(tf.int32, shape=[batch_size, num_unroll_steps, max_word_length], name="input")

    ''' First, embed characters '''
    with tf.variable_scope('Embedding'):
        char_embedding = tf.get_variable('char_embedding', [char_vocab_size, char_embed_size])

        ''' this op clears embedding vector of first symbol (symbol at position 0, which is by convention the position
        of the padding symbol). It can be used to mimic Torch7 embedding operator that keeps padding mapped to
        zero embedding vector and ignores gradient updates. For that do the following in TF:
        1. after parameter initialization, apply this op to zero out padding embedding vector
        2. after each gradient update, apply this op to keep padding at zero'''
        clear_char_embedding_padding = tf.scatter_update(char_embedding, [0], tf.constant(0.0, shape=[1, char_embed_size]))

        # [batch_size x max_word_length, num_unroll_steps, char_embed_size]
        input_embedded = tf.nn.embedding_lookup(char_embedding, input_)

        input_embedded = tf.reshape(input_embedded, [-1, max_word_length, char_embed_size])

    ''' Second, apply convolutions '''
    # [batch_size x num_unroll_steps, cnn_size]  # where cnn_size=sum(kernel_features)
    input_cnn = tdnn(input_embedded, kernels, kernel_features)

    ''' Maybe apply Highway '''
    if num_highway_layers > 0:
        input_cnn = highway(input_cnn, input_cnn.get_shape()[-1], num_layers=num_highway_layers)

    ''' Finally, do LSTM '''
    with tf.variable_scope('LSTM'):
        def create_rnn_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True, forget_bias=0.0, reuse=False)
            if dropout > 0.0:
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.-dropout)
            return cell
        
        if num_rnn_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([create_rnn_cell() for _ in range(num_rnn_layers)], state_is_tuple=True)
        else:
            cell = create_rnn_cell()

        initial_rnn_state = cell.zero_state(batch_size, dtype=tf.float32)

        input_cnn = tf.reshape(input_cnn, [batch_size, num_unroll_steps, -1])
        input_cnn2 = [tf.squeeze(x, [1]) for x in tf.split(input_cnn, num_unroll_steps, 1)]

        outputs, final_rnn_state = tf.contrib.rnn.static_rnn(cell, input_cnn2,
                                         initial_state=initial_rnn_state, dtype=tf.float32)

        # linear projection onto output (word) vocab
        logits = []
        with tf.variable_scope('WordEmbedding') as scope:
            for idx, output in enumerate(outputs):
                if idx > 0:
                    scope.reuse_variables()
                logits.append(linear(output, word_vocab_size))

    return adict(
        input = input_,
        clear_char_embedding_padding=clear_char_embedding_padding,
        input_embedded=input_embedded,
        input_cnn=input_cnn,
        initial_rnn_state=initial_rnn_state,
        final_rnn_state=final_rnn_state,
        rnn_outputs=outputs,
        logits = logits
    )


def loss_graph(logits, batch_size, num_unroll_steps):

    with tf.variable_scope('Loss'):
        targets = tf.placeholder(tf.int64, [batch_size, num_unroll_steps], name='targets')
        target_list = [tf.squeeze(x, [1]) for x in tf.split(targets, num_unroll_steps, 1)]

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = target_list), name='loss')

    return adict(
        targets=targets,
        loss=loss
    )


def training_graph(loss, learning_rate=1.0, max_grad_norm=5.0):
    ''' Builds training graph. '''
    global_step = tf.Variable(0, name='global_step', trainable=False)

    with tf.variable_scope('SGD_Training'):
        # SGD learning parameter
        learning_rate = tf.Variable(learning_rate, trainable=False, name='learning_rate')

        # collect all trainable variables
        tvars = tf.trainable_variables()
        grads, global_norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    return adict(
        learning_rate=learning_rate,
        global_step=global_step,
        global_norm=global_norm,
        train_op=train_op)


def model_size():

    params = tf.trainable_variables()
    size = 0
    for x in params:
        sz = 1
        for dim in x.get_shape():
            sz *= dim.value
        size += sz
    return size


if __name__ == '__main__':

    with tf.Session() as sess:

        with tf.variable_scope('Model'):
            graph = inference_graph(char_vocab_size=51, word_vocab_size=10000, dropout=0.5)
            graph.update(loss_graph(graph.logits, batch_size=20, num_unroll_steps=35))
            graph.update(training_graph(graph.loss, learning_rate=1.0, max_grad_norm=5.0))

        with tf.variable_scope('Model', reuse=True):
            inference_graph = inference_graph(char_vocab_size=51, word_vocab_size=10000)
            inference_graph.update(loss_graph(graph.logits, batch_size=20, num_unroll_steps=35))

        print('Model size is:', model_size())

        # need a fake variable to write scalar summary
        tf.scalar_summary('fake', 0)
        summary = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter('./tmp', graph=sess.graph)
        writer.add_summary(sess.run(summary))
        writer.flush()
