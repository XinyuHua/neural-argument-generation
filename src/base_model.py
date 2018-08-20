import os
import time
import numpy as np
import tensorflow as tf
import codecs


def load_embed_txt(embed_file, vocab):
    emb_dict = dict()
    emb_size = None
    with codecs.getreader("utf-8")(tf.gfile.GFile(embed_file, 'rb')) as f:
        for line in f:
            tokens = line.strip().split(" ")
            word = tokens[0]
            if not word in vocab._word_to_id: continue
            vec = list(map(float, tokens[1:]))
            emb_dict[word] = vec
            if emb_size:
                assert emb_size == len(vec), "All embedding size should be same."
            else:
                emb_size = len(vec)
    return emb_dict, emb_size


def _create_pretrained_emb_from_vocab(vocab, embed_file, dtype=tf.float32, name=None):
    trainable_tokens = vocab._word_to_id

    emb_dict, emb_size = load_embed_txt(embed_file, vocab)

    for token in trainable_tokens:
        if token not in emb_dict:
            emb_dict[token] = np.random.normal(size=(200))

    emb_mat = np.array(
        [emb_dict[token] for token in vocab._word_to_id], dtype=dtype.as_numpy_dtype())
    num_trainable_tokens = emb_mat.shape[0]
    emb_size = emb_mat.shape[1]
    emb_mat = tf.constant(emb_mat)
    emb_mat_const = tf.slice(emb_mat, [num_trainable_tokens, 0], [-1, -1])
    # with tf.variable_scope(scope or "pretrain_embeddings", dtype=dtype) as scope:
    emb_mat_var = tf.get_variable(name, [num_trainable_tokens, emb_size])
    return tf.concat([emb_mat_var, emb_mat_const], 0)


class baseModel(object):
    """Base class for seq2seq models."""
    def __init__(self, hps, src_vocab, tgt_vocab):
        self.hps = hps
        self._src_vocab = src_vocab
        self._tgt_vocab = tgt_vocab


    def __call__(self):
        print("[*] Building graph...")
        t0 = time.time()
        self._add_placeholder()
        self._add_model()
        print("  Time to add model: %i seconds" % (time.time() - t0))
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        if self.hps.mode == "train":
            self._add_train_op()
        self.summaries = tf.summary.merge_all()
        print("  Time to build graph: %i seconds" % (time.time() - t0))
        return

    def _add_placeholder(self):
        # encoder part
        self._enc_batch = tf.placeholder(tf.int32, [self.hps.batch_size, None], name='enc_batch')
        self._enc_lens = tf.placeholder(tf.int32, [self.hps.batch_size], name='enc_lens')
        self._enc_padding_mask = tf.placeholder(tf.float32, [self.hps.batch_size, None], name='enc_padding_mask')
        # decoder part
        self._arg_dec_batch = tf.placeholder(tf.int32, [self.hps.batch_size, self.hps.arg_max_dec_steps], name='arg_dec_batch')
        self._arg_target_batch = tf.placeholder(tf.int32, [self.hps.batch_size, self.hps.arg_max_dec_steps], name='arg_target_batch')
        self._arg_dec_padding_mask = tf.placeholder(tf.float32, [self.hps.batch_size, self.hps.arg_max_dec_steps],
                                                name='arg_dec_padding_mask')

        # add placeholder for kp decoder if necessary
        if self.hps.model in  ["sep_dec", "shd_dec"]:
            self._kp_dec_batch = tf.placeholder(tf.int32, [self.hps.batch_size, self.hps.kp_max_dec_steps],
                                                 name='kp_dec_batch')
            self._kp_target_batch = tf.placeholder(tf.int32, [self.hps.batch_size, self.hps.kp_max_dec_steps],
                                                    name='kp_target_batch')
            self._kp_dec_padding_mask = tf.placeholder(tf.float32, [self.hps.batch_size, self.hps.kp_max_dec_steps],
                                                        name='kp_dec_padding_mask')
        self._initial_attention = tf.placeholder_with_default(tf.constant(False), shape=[])

    def _add_model(self):
        with tf.variable_scope("seq2seq_model"):
            # Some initializers
            self.rand_unif_init = tf.random_uniform_initializer(-self.hps.rand_unif_init_mag, self.hps.rand_unif_init_mag,
                                                                seed=123)
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=self.hps.trunc_norm_init_std)

            # Add embedding matrix (shared by the encoder and decoder inputs)
            with tf.variable_scope('embedding'):
                self._add_embedding()

            with tf.variable_scope("encoder"):
                self._add_encoder()
                self._reduce_states()

            with tf.variable_scope("decoder"):
                self._add_decoder()

            with tf.variable_scope("output_projection"):
                self._add_output_projection()
            if self.hps.mode in ["train", "eval"]:
                with tf.variable_scope("loss"):
                    self._loss_arg = tf.contrib.seq2seq.sequence_loss(tf.stack(self.arg_dec_vocab_scores, axis=1),
                                                                    self._arg_target_batch,
                                                                    self._arg_dec_padding_mask)  # this applies softmax internally
                    self._loss = self._loss_arg

                    if self.hps.model in ["sep_dec", "shd_dec"]:
                        self._loss_kp = tf.contrib.seq2seq.sequence_loss(tf.stack(self.kp_dec_vocab_scores, axis=1),
                                                                        self._kp_target_batch,
                                                                        self._kp_dec_padding_mask)  # this applies softmax internally
                        self._loss += self._loss_kp
                        tf.summary.scalar('loss_kp', self._loss_kp)

                    tf.summary.scalar('loss', self._loss)
                    tf.summary.scalar('loss_arg', self._loss_arg)
            else:
                assert len(self.arg_dec_vocab_dists) == 1  # final_dists is a singleton list containing shape (batch_size, extended_vsize)
                topk_probs_arg, self._topk_ids_arg = tf.nn.top_k(self.arg_dec_vocab_dists[0],
                                                             self.hps.batch_size * 2 + 1)  # take the k largest probs. note batch_size=beam_size in decode mode
                self._topk_log_probs_arg = tf.log(topk_probs_arg)

                if self.hps.model in ["sep_dec", "shd_dec"]:
                    assert len(self.kp_dec_vocab_dists) == 1
                    topk_probs_kp, self._topk_ids_kp = tf.nn.top_k(self.kp_dec_vocab_dists[0],
                                                                 self.hps.batch_size * 2 + 1)  # take the k largest probs. note batch_size=beam_size in decode mode

                    self._topk_log_probs_kp = tf.log(topk_probs_kp)




    def _add_embedding(self):
        if os.path.exists(self.hps.embed_path):
          embedding_encoder = _create_pretrained_emb_from_vocab(self._src_vocab, self.hps.embed_path,
                                                              name="embedding_src")
          embedding_decoder = _create_pretrained_emb_from_vocab(self._tgt_vocab, self.hps.embed_path,
                                                              name="embedding_tgt")
        else:
          embedding_encoder = tf.get_variable('embedding_src', [self._src_vocab.size(), self.hps.emb_dim], dtype=tf.float32,
                                              initializer=self.trunc_norm_init)
          embedding_decoder = tf.get_variable('embedding_tgt', [self._tgt_vocab.size(), self.hps.emb_dim], dtype=tf.float32,
                                              initializer=self.trunc_norm_init)

        self.emb_enc_inputs = tf.nn.embedding_lookup(embedding_encoder, self._enc_batch) # tensor with shape (batch_size, max_enc_steps, emb_size)
        self.emb_arg_dec_inputs = [tf.nn.embedding_lookup(embedding_decoder, x) for x in tf.unstack(self._arg_dec_batch, axis=1)] # list length max_dec_steps containing shape (batch_size, emb_size)
        if self.hps.model in ["sep_dec", "shd_dec"]:
            self.emb_kp_dec_inputs = [tf.nn.embedding_lookup(embedding_decoder, x) for x in tf.unstack(self._kp_dec_batch, axis=1)] # list length max_dec_steps containing shape (batch_size, emb_size)


    def _add_train_op(self):
        loss_to_minimize = self._loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

        # Clip the gradients
        grads, global_norm = tf.clip_by_global_norm(gradients, self.hps.max_grad_norm)

        # Add a summary
        tf.summary.scalar('global_norm', global_norm)

        # Apply optimizer
        if self.hps.optimizer == "adam":
            optimizer = tf.train.AdamOptimizer(self.hps.learning_rate)
        else:
            optimizer = tf.train.AdagradOptimizer(0.15, initial_accumulator_value=self.hps.adagrad_init_acc)

        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')

    def _add_output_projection(self):
        with tf.variable_scope('arg_dec_output_projection'):
            w = tf.get_variable('w1', [self.hps.hidden_dim, self._tgt_vocab.size()], dtype=tf.float32, initializer=self.trunc_norm_init)
            v = tf.get_variable('v1', [self._tgt_vocab.size()], dtype=tf.float32, initializer=self.trunc_norm_init)

            arg_dec_flattened = tf.reshape(tf.stack(self.arg_dec_outputs), [-1, self.hps.hidden_dim])
            arg_dec_vocab_scores = tf.nn.xw_plus_b(arg_dec_flattened, w, v)
            arg_dec_vocab_scores = tf.reshape(arg_dec_vocab_scores, [-1, self.hps.batch_size, self._tgt_vocab.size()])
            arg_dec_vocab_dists = tf.nn.softmax(arg_dec_vocab_scores)

            self.arg_dec_vocab_scores = tf.unstack(arg_dec_vocab_scores)
            self.arg_dec_vocab_dists = tf.unstack(arg_dec_vocab_dists)

        if self.hps.model in ["sep_dec", "shd_dec"]:
            with tf.variable_scope('kp_dec_output_projection'):
                kp_dec_flattened = tf.reshape(tf.stack(self.kp_dec_outputs), [-1, self.hps.hidden_dim])
                kp_dec_vocab_scores = tf.nn.xw_plus_b(kp_dec_flattened, w, v)
                kp_dec_vocab_scores = tf.reshape(kp_dec_vocab_scores,
                                                  [-1, self.hps.batch_size, self._tgt_vocab.size()])
                kp_dec_vocab_dists = tf.nn.softmax(kp_dec_vocab_scores)

                self.kp_dec_vocab_scores = tf.unstack(kp_dec_vocab_scores)
                self.kp_dec_vocab_dists = tf.unstack(kp_dec_vocab_dists)


    def _add_encoder(self):
        cell_fw1 = tf.contrib.rnn.LSTMCell(self.hps.hidden_dim, initializer=self.rand_unif_init,
                                           state_is_tuple=True)
        cell_bw1 = tf.contrib.rnn.LSTMCell(self.hps.hidden_dim, initializer=self.rand_unif_init,
                                           state_is_tuple=True)
        cell_fw2 = tf.contrib.rnn.LSTMCell(self.hps.hidden_dim, initializer=self.rand_unif_init,
                                           state_is_tuple=True)
        cell_bw2 = tf.contrib.rnn.LSTMCell(self.hps.hidden_dim, initializer=self.rand_unif_init,
                                           state_is_tuple=True)
        if self.hps.dropout > 0.0:
            cell_fw1 = tf.contrib.rnn.DropoutWrapper(cell=cell_fw1, input_keep_prob=(1 - self.hps.dropout))
            cell_bw1 = tf.contrib.rnn.DropoutWrapper(cell=cell_bw1, input_keep_prob=(1 - self.hps.dropout))
            cell_fw2 = tf.contrib.rnn.DropoutWrapper(cell=cell_fw2, input_keep_prob=(1 - self.hps.dropout))
            cell_bw2 = tf.contrib.rnn.DropoutWrapper(cell=cell_bw2, input_keep_prob=(1 - self.hps.dropout))
        cell_fw = tf.contrib.rnn.MultiRNNCell([cell_fw1, cell_fw2])
        cell_bw = tf.contrib.rnn.MultiRNNCell([cell_bw1, cell_bw2])
        (bi_outputs, (bi_fw_st, bi_bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.emb_enc_inputs,
                                                                             dtype=tf.float32,
                                                                             sequence_length=self._enc_lens ,
                                                                             swap_memory=True)

        # concatenate state of two layers
        bi_fw_st_conc = tf.concat(axis=2, values=[bi_fw_st[0], bi_fw_st[1]])
        bi_bw_st_conc = tf.concat(axis=2, values=[bi_bw_st[0], bi_bw_st[1]])
        self.bi_fw_st_conc = tf.contrib.rnn.LSTMStateTuple(c=bi_fw_st_conc[0], h=bi_fw_st_conc[1])
        self.bi_bw_st_conc = tf.contrib.rnn.LSTMStateTuple(c=bi_bw_st_conc[0], h=bi_bw_st_conc[1])
        self.encoder_outputs = tf.concat(axis=2, values=bi_outputs)  # concatenate the forwards and backwards states


    def _reduce_states(self):
        w_reduce_c = tf.get_variable('w_reduce_c', [self.hps.hidden_dim * 4, self.hps.hidden_dim * 2], dtype=tf.float32,
                                     initializer=self.trunc_norm_init)
        w_reduce_h = tf.get_variable('w_reduce_h', [self.hps.hidden_dim * 4, self.hps.hidden_dim * 2], dtype=tf.float32,
                                     initializer=self.trunc_norm_init)
        bias_reduce_c = tf.get_variable('bias_reduce_c', [self.hps.hidden_dim * 2], dtype=tf.float32,
                                        initializer=self.trunc_norm_init)
        bias_reduce_h = tf.get_variable('bias_reduce_h', [self.hps.hidden_dim * 2], dtype=tf.float32,
                                        initializer=self.trunc_norm_init)

        # Apply linear layer
        old_c = tf.concat(axis=1, values=[self.bi_fw_st_conc.c, self.bi_bw_st_conc.c])  # Concatenation of fw and bw cell
        old_h = tf.concat(axis=1, values=[self.bi_fw_st_conc.h, self.bi_bw_st_conc.h])  # Concatenation of fw and bw state
        new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)  # Get new cell from old cell
        new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)  # Get new state from old state
        new_c_1, new_c_2 = tf.split(new_c, [self.hps.hidden_dim, self.hps.hidden_dim], 1)
        new_h_1, new_h_2 = tf.split(new_h, [self.hps.hidden_dim, self.hps.hidden_dim], 1)

        self._dec_in_state = tuple([tf.contrib.rnn.LSTMStateTuple(new_c_1, new_h_1),
                      tf.contrib.rnn.LSTMStateTuple(new_c_2, new_h_2)])  # Return new cell and state

    def _add_decoder(self):
        raise NotImplementedError("Subclasses should implement this!")


    def _make_feed_dict(self, batch, just_enc=False):
        feed_dict = {}
        feed_dict[self._enc_batch] = batch.enc_batch
        feed_dict[self._enc_lens] = batch.enc_lens
        feed_dict[self._enc_padding_mask] = batch.enc_padding_mask
        if not just_enc:
            feed_dict[self._arg_dec_batch] = batch.arg_dec_batch
            feed_dict[self._arg_target_batch] = batch.arg_target_batch
            feed_dict[self._arg_dec_padding_mask] = batch.arg_dec_padding_mask
            if self.hps.model in ["sep_dec", "shd_dec"]:
                feed_dict[self._kp_dec_batch] = batch.kp_dec_batch
                feed_dict[self._kp_target_batch] = batch.kp_target_batch
                feed_dict[self._kp_dec_padding_mask] = batch.kp_dec_padding_mask
        return feed_dict

    def run_step(self, sess, batch):
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'summaries': self.summaries,
            'loss': self._loss,
            'global_step': self.global_step,
        }
        if self.hps.mode == "train":
            to_return["train_op"] = self._train_op
        return sess.run(to_return, feed_dict)


    def run_encoder(self, sess, batch):
        feed_dict = self._make_feed_dict(batch, just_enc=True)
        (enc_states, dec_in_state, global_step) = sess.run([self.encoder_outputs, self._dec_in_state, self.global_step],
                                                           feed_dict)

        dec_in_state = (tf.contrib.rnn.LSTMStateTuple(dec_in_state[0].c[0], dec_in_state[0].h[0]),
                        tf.contrib.rnn.LSTMStateTuple(dec_in_state[1].c[0], dec_in_state[1].h[0]))
        return enc_states, dec_in_state

    def decode_onestep(self, sess, batch, latest_tokens, enc_states, kp_dec_states, dec_init_states,
                       arm, first_step=False):

        beam_size = len(dec_init_states)

        # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
        cells_0 = [np.expand_dims(state[0].c, axis=0) for state in dec_init_states]
        hiddens_0 = [np.expand_dims(state[0].h, axis=0) for state in dec_init_states]
        new_c_0 = np.concatenate(cells_0, axis=0)  # shape [batch_size,hidden_dim]
        new_h_0 = np.concatenate(hiddens_0, axis=0)  # shape [batch_size,hidden_dim]
        cells_1 = [np.expand_dims(state[1].c, axis=0) for state in dec_init_states]
        hiddens_1 = [np.expand_dims(state[1].h, axis=0) for state in dec_init_states]
        new_c_1 = np.concatenate(cells_1, axis=0)  # shape [batch_size,hidden_dim]
        new_h_1 = np.concatenate(hiddens_1, axis=0)  # shape [batch_size,hidden_dim]
        new_dec_in_state = (
        tf.contrib.rnn.LSTMStateTuple(new_c_0, new_h_0), tf.contrib.rnn.LSTMStateTuple(new_c_1, new_h_1))

        feed = {
            self.encoder_outputs: enc_states,
            self._enc_padding_mask: batch.enc_padding_mask,
            self._dec_in_state: new_dec_in_state,
            self._initial_attention: not first_step,
        }

        if arm == 2 and self.hps.model in ["sep_dec", "shd_dec"]:  # aux task
            feed[self._kp_dec_batch] = np.transpose(np.array([latest_tokens]))

            to_return = {
                "ids": self._topk_ids_kp,
                "probs": self._topk_log_probs_kp,
                "last_states": self._dec_out_state[1],
                "dec_states": self.kp_dec_states,
                "attn_dists": self.attn_dists[1]
            }

        else:

            feed[self._arg_dec_batch] = np.transpose(np.array([latest_tokens]))
            if self.hps.model == "sep_dec":
                feed[self._kp_dec_batch] = np.transpose(np.array([latest_tokens]))
            elif self.hps.model == "shd_dec":
                feed[self._dec_out_state[1]] = new_dec_in_state
                # feed[self._kp_dec_out_state] = new_dec_in_state
            to_return = {
                "ids": self._topk_ids_arg,
                "probs": self._topk_log_probs_arg,
                "last_states": self._dec_out_state[0],
                "attn_dists": self.attn_dists[0]
            }

            if self.hps.attention == "dual" and self.hps.model in ["sep_dec", "shd_dec"]:
                feed[self.kp_states] = kp_dec_states[0]
                feed[self.kp_dec_padding_mask] = kp_dec_states[1]
                to_return['dual_attn_dists'] = self.attn_dists[2]


        results = sess.run(to_return, feed_dict=feed)
        new_states = [
            (tf.contrib.rnn.LSTMStateTuple(results['last_states'][0].c[i, :], results['last_states'][0].h[i, :]),
             tf.contrib.rnn.LSTMStateTuple(results['last_states'][1].c[i, :], results['last_states'][1].h[i, :])) for i
            in range(beam_size)]


        if "dec_states" in results:
            dec_states = results['dec_states'].tolist()
        else:
            dec_states = [None] * beam_size

        assert len(results['attn_dists']) == 1
        attn_dists = results['attn_dists'][0].tolist()

        if "dual_attn_dists" in results:
            assert len(results['dual_attn_dists']) == 1
            dual_attn_dists = results['dual_attn_dists'][0].tolist()
            attn_dists = zip(attn_dists, dual_attn_dists)

        return results['ids'], results['probs'], new_states, attn_dists, dec_states

