import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops


def attention_decoder(decoder_inputs, initial_state, encoder_states, enc_padding_mask, cell,
                      initial_state_attention=False):
    with variable_scope.variable_scope("attention_decoder") as scope:
        # if this line fails, it's because the batch size isn't defined
        batch_size = encoder_states.get_shape()[0].value

        # if this line fails, it's because the attention length isn't defined
        attn_size = encoder_states.get_shape()[2].value

        # shape (batch_size, attn_len, 1, attn_size)
        encoder_states = tf.expand_dims(encoder_states, axis=2)
        attention_vec_size = attn_size

        W_h = variable_scope.get_variable("W_h", [1, 1, attn_size, attention_vec_size])

        # shape (batch_size,attn_length,1,attention_vec_size)
        encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1], "SAME")

        # Get the weight vectors v and w_c (w_c is for coverage)
        v = variable_scope.get_variable("v", [attention_vec_size])

        def attention(decoder_state):
            with variable_scope.variable_scope("attention"):

                # Pass the decoder state through a linear layer
                # shape (batch_size, attention_vec_size)
                decoder_features = linear(decoder_state, attention_vec_size, True)

                # reshape to (batch_size, 1, 1, attention_vec_size)
                decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1)

                def masked_attention(e):
                    """Take softmax of e then apply enc_padding_mask and re-normalize"""
                    attn_dist = nn_ops.softmax(e)  # take softmax. shape (batch_size, attn_length)
                    attn_dist *= enc_padding_mask  # apply mask
                    masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
                    return attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize


                # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
                e = math_ops.reduce_sum(
                        v * math_ops.tanh(encoder_features + decoder_features), [2, 3])
                # Calculate attention distribution
                attn_dist = masked_attention(e)

                # Calculate the context vector from attn_dist and encoder_states
                context_vector = math_ops.reduce_sum(
                    array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states,
                    [1, 2])  # shape (batch_size, attn_size).
                context_vector = array_ops.reshape(context_vector, [-1, attn_size])

            return context_vector, attn_dist

        outputs = []
        attn_dists = []
        state = initial_state
        context_vector = array_ops.zeros([batch_size, attn_size])

        # Ensure the second shape of attention vectors is set.
        context_vector.set_shape([None, attn_size])

        if initial_state_attention:  # true in decode mode
            # Re-calculate the context vector from the previous step 
            # so that we can pass it through a linear layer with 
            # this step's input to get a modified version of the input
            context_vector, _ = attention(initial_state)
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()

            # Merge input and previous attentions into one vector x of the same size as inp
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            x = linear([inp] + [context_vector], input_size, True)

            cell_output, state = cell(x, state)

            # Run the attention mechanism.
            if i == 0 and initial_state_attention:  # always true in decode mode
                with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                                   reuse=True):
                    context_vector, attn_dist = attention(state)
            else:
                context_vector, attn_dist = attention(state)
            attn_dists.append(attn_dist)

            # Concatenate the cell_output (= decoder state) and the context vector, 
            # and pass them through a linear layer
            with variable_scope.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + [context_vector], cell.output_size, True)
            outputs.append(output)

        return outputs, state, attn_dists

def dual_attention_shared_decoder(decoder_inputs, initial_state, encoder_states,
                      dec_2_decoder_states, enc_padding_mask, cell, dec_2_padding_mask,
                      initial_state_attention=False, reuse=False, use_dual=False, 
                      pad_last_state=False):

    if reuse:
        reuse = tf.AUTO_REUSE

    with variable_scope.variable_scope("dual_attention_decoder", reuse=reuse) as scope:
        # if this line fails, it's because the batch size isn't defined
        batch_size = encoder_states.get_shape()[0].value

        # if this line fails, it's because the attention length isn't defined
        attn_size = encoder_states.get_shape()[2].value

        enc_padding_mask = tf.reshape(enc_padding_mask, [batch_size, -1])
        shared_params = flat_attention_init(encoder_states, enc_padding_mask, reuse=reuse)
        attention = flat_attention

        dual_attn_size = dec_2_decoder_states.get_shape()[2].value
        if use_dual:
            dual_shared_params = flat_attention_init(dec_2_decoder_states, dec_2_padding_mask, 
                    use_dual=True, scope="dual_Attention")

        def stop_when(time, unused_state, unused_last_states, unused_context_vector, 
                unused_dual_context_vector, unused_decoder_states,
                unused_attn_dists, unused_dual_attn_dists, unused_outputs):
            return tf.less(time, tf.constant(len(decoder_inputs)))

        def one_step(time, state, last_states, context_vector, dual_context_vector, 
                decoder_states, attn_dists, dual_attn_dists, outputs):
            inp = tf.gather(decoder_inputs, time)
            # Merge input and previous attentions into one vector x of the same size as inp
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)

            if use_dual:
                x = linear([inp] + [context_vector] + [dual_context_vector], input_size, True,
                           scope='dual_input_transform')
            else:
                x = linear([inp] + [context_vector], input_size, True, scope='input_transform')
            # Run the decoder RNN cell. cell_output = decoder state
            cell_output, state = cell(x, state)
            decoder_states = decoder_states.write(time, cell_output)
            last_states = last_states.write(time, state)

            # Run the attention mechanism.
            with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                               reuse=True):
                context_vector, attn_dist = attention(state, shared_params)

                if use_dual:
                    dual_context_vector, dual_attn_dist = flat_attention(state, dual_shared_params,
                                                                            scope="Dual_Attention")

            attn_dists = attn_dists.write(time, attn_dist)
            if use_dual:
                dual_attn_dists = dual_attn_dists.write(time, dual_attn_dist)

            if use_dual:
                output = linear([cell_output] + [context_vector] + [dual_context_vector], cell.output_size, True,
                                scope="DualAttnOutputProjection")
            else:
                output = linear([cell_output] + [context_vector], cell.output_size, True, scope="AttnOutputProjection")
            outputs = outputs.write(time, output)
            return (
            time + 1, state, last_states, context_vector, dual_context_vector, decoder_states, attn_dists, dual_attn_dists, outputs)

        initial_time = tf.constant(0)
        initial_outputs = tf.TensorArray(dtype=tf.float32, size=len(decoder_inputs))
        initial_attn_dists = tf.TensorArray(dtype=tf.float32, size=len(decoder_inputs))
        initial_dual_attn_dists = tf.TensorArray(dtype=tf.float32, size=len(decoder_inputs))
        initial_decoder_states = tf.TensorArray(dtype=tf.float32, size=len(decoder_inputs))
        initial_context_vector = array_ops.zeros([batch_size, attn_size])
        initial_context_vector.set_shape([None, attn_size])
        initial_dual_context_vector = array_ops.zeros([batch_size, dual_attn_size])
        initial_dual_context_vector.set_shape([None, dual_attn_size])
        initial_last_states = tf.TensorArray(dtype=tf.float32, size=len(decoder_inputs))

        vec, _ = attention(initial_state, shared_params)
        initial_context_vector = tf.cond(initial_state_attention,
                                         lambda: (vec),
                                         lambda: (initial_context_vector))
        if use_dual:
            initial_dual_context_vector = tf.cond(initial_state_attention,
                                                  lambda: flat_attention(initial_state,
                                                      dual_shared_params,
                                                      scope="Dual_Attention")[0],
                                                  lambda: initial_dual_context_vector)

        time, last_state, last_states, context_vector, dual_context_vector,
        decoder_states, attn_dists, dual_attn_dists, outputs = tf.while_loop(stop_when, one_step,
                          loop_vars=[initial_time, initial_state, initial_last_states,
                                     initial_context_vector, initial_dual_context_vector,
                                     initial_decoder_states,
                                     initial_attn_dists, initial_dual_attn_dists, initial_outputs],
                                     parallel_iterations=32, swap_memory=True)

        outputs = tf.unstack(outputs.stack(), num=len(decoder_inputs))
        decoder_states = tf.stack(
                tf.unstack(decoder_states.stack(), num=len(decoder_inputs)), axis=1)
        attn_dists = tf.unstack(attn_dists.stack(), num=len(decoder_inputs))
        if use_dual:
            dual_attn_dists = tf.unstack(dual_attn_dists.stack(), num=len(decoder_inputs))
        else:
            dual_attn_dists = None
        last_states = tf.unstack(last_states.stack(), num=len(decoder_inputs))
        if pad_last_state:
            padding_inds = tf.cast(tf.reduce_sum(dec_2_padding_mask, axis=1), tf.int32) - 1
            tmp_1 = tf.unstack(last_states, axis=3)
            tmp_2 = tf.unstack(padding_inds)

            tmp_3 = tmp_1[0]
            tmp_4 = tf.transpose(tmp_3, perm=[1, 0, 2, 3])
            tmp_5 = tmp_4[0]

            last_state_0 = [tf.transpose(x, perm=[1, 0, 2, 3])[0][ind] for x, ind in
                            zip(tf.unstack(last_states, axis=3), tf.unstack(padding_inds))]

            last_state_0 = tf.transpose(last_state_0, perm=[1, 0, 2])
            last_state_0 = tf.contrib.rnn.LSTMStateTuple(last_state_0[0], last_state_0[1])

            last_state_1 = [tf.transpose(x, perm=[1, 0, 2, 3])[1][ind] for x, ind in
                            zip(tf.unstack(last_states, axis=3), tf.unstack(padding_inds))]

            last_state_1 = tf.transpose(last_state_1, perm=[1, 0, 2])
            last_state_1 = tf.contrib.rnn.LSTMStateTuple(last_state_1[0], last_state_1[1])

            last_state = (last_state_0, last_state_1)

    return outputs, last_state, decoder_states, attn_dists, dual_attn_dists

def dual_attention_decoder(decoder_inputs, initial_state, encoder_states, dec_2_decoder_states, 
                           enc_padding_mask, cell, dec_2_padding_mask, 
                           initial_state_attention=False, reuse=False, use_dual=False):
    with variable_scope.variable_scope("dual_attention_decoder", reuse=reuse) as scope:
        # if this line fails, it's because the batch size isn't defined
        batch_size = encoder_states.get_shape()[0].value

        # if this line fails, it's because the attention length isn't defined
        attn_size = encoder_states.get_shape()[2].value
        enc_padding_mask = tf.reshape(enc_padding_mask, [batch_size, -1])
        shared_params = flat_attention_init(encoder_states, enc_padding_mask, reuse=reuse)
        attention = flat_attention

        dual_attn_size = dec_2_decoder_states.get_shape()[2].value
        if use_dual:
            dual_shared_params = flat_attention_init(dec_2_decoder_states, dec_2_padding_mask,
                                                    use_dual=True, scope="Dual_Attention")

        def stop_when(time, unused_state, unused_context_vector, unused_dual_context_vector,
                      unused_decoder_states, unused_attn_dists, unused_dual_attn_dists,
                      unused_outputs):
            return tf.less(time, tf.constant(len(decoder_inputs)))

        def one_step(time, state, context_vector, dual_context_vector, decoder_states, attn_dists, dual_attn_dists,
                     outputs):
            inp = tf.gather(decoder_inputs, time)
            # Merge input and previous attentions into one vector x of the same size as inp
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)

            if use_dual:
                x = linear([inp] + [context_vector] + [dual_context_vector], input_size, True,
                           scope='dual_input_transform')
            else:
                x = linear([inp] + [context_vector], input_size, True, scope='input_transform')
            # Run the decoder RNN cell. cell_output = decoder state
            cell_output, state = cell(x, state)
            decoder_states = decoder_states.write(time, cell_output)

            # Run the attention mechanism.
            with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                               reuse=True):
                context_vector, attn_dist = attention(state, shared_params)

                if use_dual:
                    dual_context_vector, dual_attn_dist = flat_attention(state, dual_shared_params,
                                                                            scope="Dual_Attention")

            attn_dists = attn_dists.write(time, attn_dist)
            if use_dual:
                dual_attn_dists = dual_attn_dists.write(time, dual_attn_dist)

            # Concatenate the cell_output (= decoder state) and the context vector, and pass them through a linear layer
            # This is V[s_t, h*_t] + b in the paper
            if use_dual:
                output = linear([cell_output] + [context_vector] + [dual_context_vector], cell.output_size, True,
                                scope="DualAttnOutputProjection")
            else:
                output = linear([cell_output] + [context_vector], cell.output_size, True, scope="AttnOutputProjection")
            outputs = outputs.write(time, output)
            return (
            time + 1, state, context_vector, dual_context_vector, decoder_states, attn_dists, dual_attn_dists, outputs)

        initial_time = tf.constant(0)
        initial_outputs = tf.TensorArray(dtype=tf.float32, size=len(decoder_inputs))
        initial_attn_dists = tf.TensorArray(dtype=tf.float32, size=len(decoder_inputs))
        initial_dual_attn_dists = tf.TensorArray(dtype=tf.float32, size=len(decoder_inputs))
        initial_decoder_states = tf.TensorArray(dtype=tf.float32, size=len(decoder_inputs))
        initial_context_vector = array_ops.zeros([batch_size, attn_size])
        initial_context_vector.set_shape([None, attn_size])
        initial_dual_context_vector = array_ops.zeros([batch_size, dual_attn_size])
        initial_dual_context_vector.set_shape(
            [None, dual_attn_size])


        vec, _ = attention(initial_state, shared_params)
        initial_context_vector = tf.cond(initial_state_attention, \
                                                           lambda: (vec), \
                                                           lambda: (initial_context_vector))
        if use_dual:
            initial_dual_context_vector = tf.cond(initial_state_attention, \
                                                  lambda: flat_attention(initial_state, dual_shared_params,
                                                                         scope="Dual_Attention")[0], \
                                                  lambda: initial_dual_context_vector)

        time, last_state, context_vector, dual_context_vector, decoder_states, attn_dists, dual_attn_dists, outputs = \
            tf.while_loop(stop_when, one_step, \
                          loop_vars=[initial_time, initial_state, initial_context_vector, initial_dual_context_vector,
                                     initial_decoder_states, \
                                     initial_attn_dists, initial_dual_attn_dists, initial_outputs], \
                          parallel_iterations=32, swap_memory=True)

        outputs = tf.unstack(outputs.stack(), num=len(decoder_inputs))
        decoder_states = tf.stack(tf.unstack(decoder_states.stack(), num=len(decoder_inputs)), axis=1)
        attn_dists = tf.unstack(attn_dists.stack(), num=len(decoder_inputs))
        if use_dual:
            dual_attn_dists = tf.unstack(dual_attn_dists.stack(), num=len(decoder_inputs))
        else:
            dual_attn_dists = None

    return outputs, last_state, decoder_states, attn_dists, dual_attn_dists


def linear(args, output_size, bias, bias_start=0.0, scope=None):
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    if isinstance(args, tuple):
        args = args[1]
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(axis=1, values=args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
    return res + bias_term


def flat_attention_init(encoder_states, enc_padding_mask, use_dual=False, reuse=False, scope=None):
    with variable_scope.variable_scope(scope or "Attention"):
        batch_size = encoder_states.get_shape()[0].value  # if this line fails, it's because the batch size isn't defined
        attn_size = encoder_states.get_shape()[2].value  # if this line fails, it's because the attention length isn't defined
        attention_vec_size = attn_size

        # Reshape encoder_states (need to insert a dim)
        encoder_states = tf.expand_dims(encoder_states, axis=2)  # now is shape (batch_size, attn_len, 1, attn_size)

        if use_dual:
            # Get the weight matrix W_h and apply it to each encoder state to get (W_h h_i), the encoder features
            W_h = variable_scope.get_variable("W_d", [1, 1, attn_size, attention_vec_size])
            # Get the weight vectors v
            v = variable_scope.get_variable("v_d", [attention_vec_size])

            encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1], "SAME")

        else:
            # Get the weight matrix W_h and apply it to each encoder state to get (W_h h_i), the encoder features
            W_h = variable_scope.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
            # Get the weight vectors v
            v = variable_scope.get_variable("v", [attention_vec_size])

            encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1],
                                             "SAME")  # shape (batch_size,attn_length,1,attention_vec_size)

        return (batch_size, attn_size, encoder_states, attention_vec_size, encoder_features, v, enc_padding_mask, None)


def flat_attention(decoder_state, shared_params, scope=None):
    (batch_size, _, encoder_states, attention_vec_size, encoder_features, v, enc_padding_mask, _) = shared_params

    with variable_scope.variable_scope(scope or "attention"):
        decoder_features = linear(decoder_state, attention_vec_size, True,
                                  scope="decoder_features")  # shape (batch_size, attention_vec_size)
        decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1),
                                          1)  # reshape to (batch_size, 1, 1, attention_vec_size)

        def masked_attention(e):
            """Take softmax of e then apply enc_padding_mask and re-normalize"""
            attn_dist = nn_ops.softmax(e)  # take softmax. shape (batch_size, attn_length) or (batch_size, num_heads, attn_length) if multihead
            attn_dist *= enc_padding_mask  # apply mask
            masked_sums = tf.reduce_sum(attn_dist, axis=-1,
                                        keep_dims=True)  # shape (batch_size,1) or (batch_size,num_heads,1) if multihead
            return attn_dist / masked_sums


        # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
        e = math_ops.reduce_sum(v * math_ops.tanh(encoder_features + decoder_features), [-1, -2])

        # Calculate attention distribution
        attn_dist = masked_attention(e)


        # Calculate the context vector from attn_dist and encoder_states
        context_vector = math_ops.reduce_sum(array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states,
                                                 [1, 2])  # shape (batch_size, attn_size).

    return context_vector, attn_dist
