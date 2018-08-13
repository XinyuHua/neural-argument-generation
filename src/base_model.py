import os
import time
import numpy as np
import tensorflow as tf
import codecs
from attention_decoder import attention_decoder
from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = tf.app.flags.FLAGS


def load_embed_txt(embed_file, vocab):
  """
  Load embed_file into a python dictionary.
  Note: the embed_file should be a Glove formated txt file. Assuming
  embed_size=5, for example:
  the -0.071549 0.093459 0.023738 -0.090339 0.056123
  to 0.57346 0.5417 -0.23477 -0.3624 0.4037
  and 0.20327 0.47348 0.050877 0.002103 0.060547
  Args:
    embed_file: file path to the embedding file.
  Returns:
    a dictionary that maps word to vector, and the size of embedding dimensions.
  """
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

def _create_pretrained_emb_from_vocab(
    vocab, embed_file, dtype=tf.float32, name=None):
  """Load pretrain embeding from embed_file, and return an embedding matrix.
  Args:
    embed_file: Path to a Glove formated embedding txt file.
  """
  trainable_tokens = vocab._word_to_id

  emb_dict, emb_size = load_embed_txt(embed_file, vocab)

  for token in trainable_tokens:
    #utils.print_out('    %s' % token)
    if token not in emb_dict:
      #emb_dict[token] = [0.0] * emb_size
      emb_dict[token] = np.random.normal(size=(200))

  emb_mat = np.array(
      [emb_dict[token] for token in vocab._word_to_id], dtype=dtype.as_numpy_dtype())
  num_trainable_tokens = emb_mat.shape[0]
  emb_size = emb_mat.shape[1]
  emb_mat = tf.constant(emb_mat)
  emb_mat_const = tf.slice(emb_mat, [num_trainable_tokens, 0], [-1, -1])
  #with tf.variable_scope(scope or "pretrain_embeddings", dtype=dtype) as scope:
  emb_mat_var = tf.get_variable(name, [num_trainable_tokens, emb_size])
  return tf.concat([emb_mat_var, emb_mat_const], 0)


class SummarizationModel(object):
  """A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode, and coverage"""

  def __init__(self, hps, src_vocab, tgt_vocab):
    self._hps = hps
    self._src_vocab = src_vocab
    self._tgt_vocab = tgt_vocab

  def _add_placeholders(self):
    """Add placeholders to the graph. These are entry points for any input data."""
    hps = self._hps

    # encoder part
    self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch')
    self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size], name='enc_lens')
    self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='enc_padding_mask')
    # decoder part
    self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
    self._target_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
    self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps], name='dec_padding_mask')
    if hps.mode == "decode" and hps.coverage:
      self.prev_coverage = tf.placeholder(tf.float32, [hps.batch_size, None], name='prev_coverage')


  def _make_feed_dict(self, batch, just_enc=False):
    """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

    Args:
      batch: Batch object
      just_enc: Boolean. If True, only feed the parts needed for the encoder.
    """
    feed_dict = {}
    feed_dict[self._enc_batch] = batch.enc_batch
    feed_dict[self._enc_lens] = batch.enc_lens
    feed_dict[self._enc_padding_mask] = batch.enc_padding_mask
    if not just_enc:
      feed_dict[self._dec_batch] = batch.dec_batch
      feed_dict[self._target_batch] = batch.target_batch
      feed_dict[self._dec_padding_mask] = batch.dec_padding_mask
    return feed_dict

  def _add_encoder(self, encoder_inputs, seq_len):
    """Add a single-layer bidirectional LSTM encoder to the graph.

    Args:
      encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
      seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

    Returns:
      encoder_outputs:
        A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
      fw_state, bw_state:
        Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    """
    with tf.variable_scope('encoder'):
      cell_fw1 = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      cell_bw1 = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      cell_fw2 = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      cell_bw2 = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      if self._hps.dropout > 0.0:
        cell_fw1 = tf.contrib.rnn.DropoutWrapper(cell=cell_fw1, input_keep_prob=(1 - self._hps.dropout))
        cell_bw1 = tf.contrib.rnn.DropoutWrapper(cell=cell_bw1, input_keep_prob=(1 - self._hps.dropout))
        cell_fw2 = tf.contrib.rnn.DropoutWrapper(cell=cell_fw2, input_keep_prob=(1 - self._hps.dropout))
        cell_bw2 = tf.contrib.rnn.DropoutWrapper(cell=cell_bw2, input_keep_prob=(1 - self._hps.dropout))
      cell_fw = tf.contrib.rnn.MultiRNNCell([cell_fw1, cell_fw2])
      cell_bw = tf.contrib.rnn.MultiRNNCell([cell_bw1, cell_bw2])
      (bi_outputs, (bi_fw_st, bi_bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)

      # concatenate state of two layers
      bi_fw_st_conc = tf.concat(axis=2, values=[bi_fw_st[0], bi_fw_st[1]])
      bi_bw_st_conc = tf.concat(axis=2, values=[bi_bw_st[0], bi_bw_st[1]])
      bi_fw_st_conc = tf.contrib.rnn.LSTMStateTuple(c=bi_fw_st_conc[0], h=bi_fw_st_conc[1])
      bi_bw_st_conc = tf.contrib.rnn.LSTMStateTuple(c=bi_bw_st_conc[0], h=bi_bw_st_conc[1])
      encoder_outputs = tf.concat(axis=2, values=bi_outputs)  # concatenate the forwards and backwards states

    return encoder_outputs, bi_fw_st_conc, bi_bw_st_conc

  def _reduce_states(self, fw_st, bw_st):
    """Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not.

    Args:
      fw_st: LSTMStateTuple with hidden_dim units.
      bw_st: LSTMStateTuple with hidden_dim units.

    Returns:
      state: LSTMStateTuple with hidden_dim units.
    """
    hidden_dim = self._hps.hidden_dim
    with tf.variable_scope('reduce_final_st'):

      # Define weights and biases to reduce the cell and reduce the state
      w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 4, hidden_dim *2], dtype=tf.float32, initializer=self.trunc_norm_init)
      w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 4, hidden_dim * 2], dtype=tf.float32, initializer=self.trunc_norm_init)
      bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim * 2], dtype=tf.float32, initializer=self.trunc_norm_init)
      bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim * 2], dtype=tf.float32, initializer=self.trunc_norm_init)

      # Apply linear layer
      old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c]) # Concatenation of fw and bw cell
      old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h]) # Concatenation of fw and bw state
      new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c) # Get new cell from old cell
      new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h) # Get new state from old state
      new_c_1, new_c_2 = tf.split(new_c, [hidden_dim, hidden_dim], 1)
      new_h_1, new_h_2 = tf.split(new_h, [hidden_dim, hidden_dim], 1)
      return tuple([tf.contrib.rnn.LSTMStateTuple(new_c_1, new_h_1), tf.contrib.rnn.LSTMStateTuple(new_c_2, new_h_2)]) # Return new cell and state
      #return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

  def _add_decoder(self, inputs):
    """Add attention decoder to the graph. In train or eval mode, you call this once to get output on ALL steps. In decode (beam search) mode, you call this once for EACH decoder step.

    Args:
      inputs: inputs to the decoder (word embeddings). A list of tensors shape (batch_size, emb_dim)

    Returns:
      outputs: List of tensors; the outputs of the decoder
      out_state: The final state of the decoder
      attn_dists: A list of tensors; the attention distributions
      p_gens: A list of tensors shape (batch_size, 1); the generation probabilities
      coverage: A tensor, the current coverage vector
    """
    hps = self._hps
    cell1 = tf.contrib.rnn.LSTMCell(hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)
    cell2 = tf.contrib.rnn.LSTMCell(hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)
    cell = tf.contrib.rnn.MultiRNNCell([cell1, cell2])
    prev_coverage = self.prev_coverage if hps.mode == "decode" and hps.coverage else None  # In decode mode, we run attention_decoder one step at a time and so need to pass in the previous step's coverage vector each time

    outputs, out_state, attn_dists, coverage = attention_decoder(inputs, self._dec_in_state, self._enc_states, self._enc_padding_mask, cell, initial_state_attention=(hps.mode=="decode"), use_coverage=hps.coverage, prev_coverage=prev_coverage)

    return outputs, out_state, attn_dists, coverage


  def _add_emb_vis(self, embedding_var):
    """Do setup so that we can view word embedding visualization in Tensorboard, as described here:
    https://www.tensorflow.org/get_started/embedding_viz
    Make the vocab metadata file, then make the projector config file pointing to it."""
    train_dir = os.path.join(FLAGS.log_root, "train")
    vocab_metadata_path = os.path.join(train_dir, "vocab_metadata.tsv")
    self._src_vocab.write_metadata(vocab_metadata_path) # write metadata file
    summary_writer = tf.summary.FileWriter(train_dir)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = vocab_metadata_path
    projector.visualize_embeddings(summary_writer, config)

  def _add_seq2seq(self):
    """Add the whole sequence-to-sequence model to the graph."""
    hps = self._hps
    src_vsize = self._src_vocab.size() # size of the vocabulary
    tgt_vsize = self._tgt_vocab.size()  # size of the vocabulary

    with tf.variable_scope('seq2seq'):
      # Some initializers
      self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
      self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)

      # Add embedding matrix (shared by the encoder and decoder inputs)
      with tf.variable_scope('embedding'):
        if os.path.exists(self._hps.embed_path):
          embedding_encoder = _create_pretrained_emb_from_vocab(self._src_vocab, self._hps.embed_path,
                                                              name="embedding_src")
          embedding_decoder = _create_pretrained_emb_from_vocab(self._tgt_vocab, self._hps.embed_path,
                                                              name="embedding_tgt")
        else:
          embedding_encoder = tf.get_variable('embedding_src', [src_vsize, hps.emb_dim], dtype=tf.float32,
                                              initializer=self.trunc_norm_init)
          embedding_decoder = tf.get_variable('embedding_tgt', [tgt_vsize, hps.emb_dim], dtype=tf.float32,
                                              initializer=self.trunc_norm_init)
        if hps.mode=="train": self._add_emb_vis(embedding_encoder) # add to tensorboard
        emb_enc_inputs = tf.nn.embedding_lookup(embedding_encoder, self._enc_batch) # tensor with shape (batch_size, max_enc_steps, emb_size)
        emb_dec_inputs = [tf.nn.embedding_lookup(embedding_decoder, x) for x in tf.unstack(self._dec_batch, axis=1)] # list length max_dec_steps containing shape (batch_size, emb_size)

      # Add the encoder.
      enc_outputs, fw_st, bw_st = self._add_encoder(emb_enc_inputs, self._enc_lens)
      self._enc_states = enc_outputs

      # Our encoder is bidirectional and our decoder is unidirectional so we need to reduce the final encoder hidden state to the right size to be the initial decoder hidden state
      self._dec_in_state = self._reduce_states(fw_st, bw_st)

      # Add the decoder.
      with tf.variable_scope('decoder'):
        decoder_outputs, self._dec_out_state, self.attn_dists, self.coverage = self._add_decoder(emb_dec_inputs)

      # Add the output projection to obtain the vocabulary distribution
      with tf.variable_scope('output_projection'):
        w = tf.get_variable('w', [hps.hidden_dim, tgt_vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
        w_t = tf.transpose(w)
        v = tf.get_variable('v', [tgt_vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
        
        decoder_flattened = tf.reshape(tf.stack(decoder_outputs), [-1, hps.hidden_dim])
        vocab_scores = tf.nn.xw_plus_b(decoder_flattened, w, v)
        vocab_scores = tf.reshape(vocab_scores, [-1, hps.batch_size, tgt_vsize])
        vocab_dists = tf.nn.softmax(vocab_scores)

        vocab_scores = tf.unstack(vocab_scores)
        vocab_dists = tf.unstack(vocab_dists)
        '''
        vocab_scores = [] # vocab_scores is the vocabulary distribution before applying softmax. Each entry on the list corresponds to one decoder step
        for i,output in enumerate(decoder_outputs):
          if i > 0:
            tf.get_variable_scope().reuse_variables()
          vocab_scores.append(tf.nn.xw_plus_b(output, w, v)) # apply the linear layer

        vocab_dists = [tf.nn.softmax(s) for s in vocab_scores] # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
        '''
      final_dists = vocab_dists
      self.vocab_dists = vocab_dists

      if hps.mode in ['train', 'eval', 'eval_ppl']:
        # Calculate the loss
        with tf.variable_scope('loss'):
          self._loss = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1), self._target_batch, self._dec_padding_mask) # this applies softmax internally
          tf.summary.scalar('loss', self._loss)
        if hps.coverage:
          with tf.variable_scope('coverage_loss'):
            self._coverage_loss = _coverage_loss(self.attn_dists, self._dec_padding_mask)
            tf.summary.scalar('coverage_loss', self._coverage_loss)
            self._total_loss = self._loss + hps.cov_loss_wt * self._coverage_loss
            tf.summary.scalar('total_loss', self._total_loss)
    if hps.mode == "decode":
      # We run decode beam search mode one decoder step at a time
      assert len(final_dists)==1 # final_dists is a singleton list containing shape (batch_size, extended_vsize)
      final_dists = final_dists[0]
      if not hps.sampling_decoding:
        topk_probs, self._topk_ids = tf.nn.top_k(final_dists, hps.batch_size*2) # take the k largest probs. note batch_size=beam_size in decode mode
      else:
        topk_probs, self._topk_ids = tf.nn.top_k(final_dists, 100)
      self._topk_log_probs = tf.log(topk_probs)


  def _add_train_op(self):
    """Sets self._train_op, the op to run for training."""
    # Take gradients of the trainable variables w.r.t. the loss function to minimize
    loss_to_minimize = self._total_loss if self._hps.coverage else self._loss
    tvars = tf.trainable_variables()
    gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

    # Clip the gradients
    with tf.device("/gpu:" + self._hps.gpu_id):
      grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)

    # Add a summary
    tf.summary.scalar('global_norm', global_norm)

    # Apply adagrad optimizer
    if self._hps.optimizer == "adam":
      optimizer = tf.train.AdamOptimizer(self._hps.lr)
    else:
      optimizer = tf.train.AdagradOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)
    
    with tf.device("/gpu:" + self._hps.gpu_id):
      self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')


  def build_graph(self):
    """Add the placeholders, model, global step, train_op and summaries to the graph"""
    tf.logging.info('Building graph...')
    t0 = time.time()
    self._add_placeholders()
    with tf.device("/gpu:" + self._hps.gpu_id):
      self._add_seq2seq()
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    if self._hps.mode == 'train':
      self._add_train_op()
    self._summaries = tf.summary.merge_all()
    t1 = time.time()
    tf.logging.info('Time to build graph: %i seconds', t1 - t0)

  def run_train_step(self, sess, batch):
    """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'train_op': self._train_op,
        'summaries': self._summaries,
        'loss': self._loss,
        'global_step': self.global_step,
    }
    if self._hps.coverage:
      to_return['coverage_loss'] = self._coverage_loss
    return sess.run(to_return, feed_dict)

  def run_eval_step(self, sess, batch):
    """Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'summaries': self._summaries,
        'loss': self._loss,
        'global_step': self.global_step,
        'vocab_dists': self.vocab_dists,
        'target_batch': self._target_batch,
    }
    if self._hps.coverage:
      to_return['coverage_loss'] = self._coverage_loss
    return sess.run(to_return, feed_dict)

  def run_encoder(self, sess, batch):
    """For beam search decoding. Run the encoder on the batch and return the encoder states and decoder initial state.

    Args:
      sess: Tensorflow session.
      batch: Batch object that is the same example repeated across the batch (for beam search)

    Returns:
      enc_states: The encoder states. A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim].
      dec_in_state: A LSTMStateTuple of shape ([1,hidden_dim],[1,hidden_dim])
    """
    feed_dict = self._make_feed_dict(batch, just_enc=True) # feed the batch into the placeholders
    (enc_states, dec_in_state, global_step) = sess.run([self._enc_states, self._dec_in_state, self.global_step], feed_dict) # run the encoder

    # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    # Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
    # dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state[1].c[0], dec_in_state[1].h[0])
    dec_in_state = (tf.contrib.rnn.LSTMStateTuple(dec_in_state[0].c[0], dec_in_state[0].h[0]), tf.contrib.rnn.LSTMStateTuple(dec_in_state[1].c[0], dec_in_state[1].h[0]))
    return enc_states, dec_in_state


  def decode_onestep(self, sess, batch, latest_tokens, enc_states, dec_init_states,  prev_coverage):
    """For beam search decoding. Run the decoder for one step.

    Args:
      sess: Tensorflow session.
      batch: Batch object containing single example repeated across the batch
      latest_tokens: Tokens to be fed as input into the decoder for this timestep
      enc_states: The encoder states.
      dec_init_states: List of beam_size LSTMStateTuples; the decoder states from the previous timestep
      prev_coverage: List of np arrays. The coverage vectors from the previous timestep. List of None if not using coverage.

    Returns:
      ids: top 2k ids. shape [beam_size, 2*beam_size]
      probs: top 2k log probabilities. shape [beam_size, 2*beam_size]
      new_states: new states of the decoder. a list length beam_size containing
        LSTMStateTuples each of shape ([hidden_dim,],[hidden_dim,])
      attn_dists: List length beam_size containing lists length attn_length.
      p_gens: Generation probabilities for this step. A list length beam_size. List of None if in baseline mode.
      new_coverage: Coverage vectors for this step. A list of arrays. List of None if coverage is not turned on.
    """

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
    new_dec_in_state = (tf.contrib.rnn.LSTMStateTuple(new_c_0, new_h_0), tf.contrib.rnn.LSTMStateTuple(new_c_1, new_h_1))

    feed = {
        self._enc_states: enc_states,
        self._enc_padding_mask: batch.enc_padding_mask,
        self._dec_in_state: new_dec_in_state,
        self._dec_batch: np.transpose(np.array([latest_tokens])),
    }

    to_return = {
      "ids": self._topk_ids,
      "probs": self._topk_log_probs,
      "states": self._dec_out_state,
      "attn_dists": self.attn_dists
    }

    if self._hps.coverage:
      feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
      to_return['coverage'] = self.coverage

    results = sess.run(to_return, feed_dict=feed) # run the decoder step

    # Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
    new_states = [
      (tf.contrib.rnn.LSTMStateTuple(results['states'][0].c[i, :], results['states'][0].h[i, :]),
       tf.contrib.rnn.LSTMStateTuple(results['states'][1].c[i, :], results['states'][1].h[i, :]))  for i in xrange(beam_size)]

    # Convert singleton list containing a tensor to a list of k arrays
    assert len(results['attn_dists'])==1
    attn_dists = results['attn_dists'][0].tolist()
    if FLAGS.coverage:
      new_coverage = results['coverage'].tolist()
      assert len(new_coverage) == beam_size
    else:
      new_coverage = [None for _ in xrange(beam_size)]
    return results['ids'], results['probs'], new_states, attn_dists, new_coverage


def _mask_and_avg(values, padding_mask):
  """Applies mask to values then returns overall average (a scalar)
  Args:
    values: a list length max_dec_steps containing arrays shape (batch_size).
    padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.
  Returns:
    a scalar
  """

  dec_lens = tf.reduce_sum(padding_mask, axis=1) # shape batch_size. float32
  values_per_step = [v * padding_mask[:,dec_step] for dec_step,v in enumerate(values)]
  values_per_ex = sum(values_per_step)/dec_lens # shape (batch_size); normalized value for each batch member
  return tf.reduce_mean(values_per_ex) # overall average


def _coverage_loss(attn_dists, padding_mask):
  """Calculates the coverage loss from the attention distributions.
  Args:
    attn_dists: The attention distributions for each decoder p. A list length max_dec_steps containing shape (batch_size, attn_length)
    padding_mask: shape (batch_size, max_dec_steps).
  Returns:
    coverage_loss: scalar
  """
  coverage = tf.zeros_like(attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
  covlosses = [] # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
  for a in attn_dists:
    covloss = tf.reduce_sum(tf.minimum(a, coverage), [1]) # calculate the coverage loss for this step
    covlosses.append(covloss)
    coverage += a # update the coverage vector
  coverage_loss = _mask_and_avg(covlosses, padding_mask)
  return coverage_loss
