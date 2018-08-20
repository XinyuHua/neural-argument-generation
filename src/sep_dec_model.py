import tensorflow as tf
from base_model import baseModel
from attention import dual_attention_decoder

class separateDecoderModel(baseModel):

    def _add_decoder(self):
        # kp decoder
        with tf.variable_scope("kp_decoder"):
            kp_dec_cell1 = tf.contrib.rnn.LSTMCell(self.hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)
            kp_dec_cell2 = tf.contrib.rnn.LSTMCell(self.hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)
            kp_dec_cell = tf.contrib.rnn.MultiRNNCell([kp_dec_cell1, kp_dec_cell2])
            self.kp_states = tf.ones((self.hps.batch_size, self.hps.kp_max_dec_steps, self.hps.hidden_dim), dtype=tf.float32)
            self.kp_dec_padding_mask = tf.ones((self.hps.batch_size, self.hps.kp_max_dec_steps), dtype=tf.float32)

            self.kp_dec_outputs, kp_dec_out_state, self.kp_dec_states, kp_attn_dists, _ = \
                dual_attention_decoder(self.emb_kp_dec_inputs, self._dec_in_state,
                                       self.encoder_outputs, self.kp_states,
                                       self._enc_padding_mask, kp_dec_cell,
                                       self.kp_dec_padding_mask, initial_state_attention=self._initial_attention)

            if self.hps.attention == "dual" and self.hps.mode in ["train", "eval"]:
                self.kp_states = self.kp_dec_states
                self.kp_dec_padding_mask = self._kp_dec_padding_mask

        with tf.variable_scope('arg_decoder'):
            arg_dec_cell1 = tf.contrib.rnn.LSTMCell(self.hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)
            arg_dec_cell2 = tf.contrib.rnn.LSTMCell(self.hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)
            arg_dec_cell = tf.contrib.rnn.MultiRNNCell([arg_dec_cell1, arg_dec_cell2])
            self.arg_dec_outputs, arg_dec_out_state, arg_dec_states, arg_attn_dists, dual_attn_dists = \
                dual_attention_decoder(self.emb_arg_dec_inputs, self._dec_in_state,
                                       self.encoder_outputs, self.kp_states,
                                       self._enc_padding_mask, arg_dec_cell,
                                       self.kp_dec_padding_mask, initial_state_attention=self._initial_attention,
                                       use_dual=(self.hps.attention == "dual"))

        self._dec_out_state = (arg_dec_out_state, kp_dec_out_state)
        self.attn_dists = (arg_attn_dists, kp_attn_dists, dual_attn_dists)
        return

