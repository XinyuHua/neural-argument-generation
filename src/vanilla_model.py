import tensorflow as tf
from base_model import baseModel
from attention import attention_decoder


class vanillaSeq2seqModel(baseModel):

    def _add_decoder(self):
        cell1 = tf.contrib.rnn.LSTMCell(self.hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)
        cell2 = tf.contrib.rnn.LSTMCell(self.hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)
        cell = tf.contrib.rnn.MultiRNNCell([cell1, cell2])

        self.arg_dec_outputs, arg_dec_out_state, arg_attn_dists = \
            attention_decoder(self.emb_arg_dec_inputs, self._dec_in_state, self.encoder_outputs,
                              self._enc_padding_mask, cell, initial_state_attention=(self.hps.mode == "decode"))
        self._dec_out_state = (arg_dec_out_state, [])
        self.attn_dists = (arg_attn_dists, [], [])
        return

