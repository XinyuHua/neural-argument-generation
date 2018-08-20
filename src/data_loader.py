from random import shuffle
import queue
import tensorflow as tf
import time
import utils
import numpy as np
from threading import Thread

class DataSample(object):

    def __init__(self, op_txt, evd_txt, kp_sents, arg_sents, src_vocab, tgt_vocab, hps):
        self.hps = hps

        # Get ids of special tokens
        arg_start_decoding = tgt_vocab.word2id(utils.ARG_START_DECODING)
        arg_stop_decoding = tgt_vocab.word2id(utils.ARG_STOP_DECODING)
        kp_start_decoding = tgt_vocab.word2id(utils.KP_START_DECODING)
        kp_stop_decoding = tgt_vocab.word2id(utils.KP_STOP_DECODING)

        # tokenize the source input
        op_lst = op_txt.split()
        evd_lst = evd_txt.split()
        src_lst = op_lst
        if hps.model == "base_enc_evd":
            src_lst += ["<ctx>"] + evd_lst
        elif hps.model == "base_enc_kp":
            src_lst += ["<ctx>", "<sent_ctx>"] + "</sent_ctx> <sent_ctx>".join(kp_sents).split() + ["</sent_ctx>"]

        if len(src_lst) > self.hps.max_enc_steps:
            src_lst = src_lst[: self.hps.max_enc_steps]
        self.enc_len = len(src_lst)
        self.enc_input = [src_vocab.word2id(tok) for tok in src_lst]

        # tokenize the target output
        arg_ids = [tgt_vocab.word2id(tok) for tok in ' '.join(arg_sents).split()]
        self.arg_dec_input, self.target_arg = self.get_dec_inp_targ_seqs(arg_ids, hps.arg_max_dec_steps,
                                                                     arg_start_decoding, arg_stop_decoding)
        self.dec_arg_len = len(self.arg_dec_input)

        if hps.model in ["sep_dec", "shd_dec"]:
            kp_ids = [tgt_vocab.word2id(tok) for tok in ['<sent_cs>'] + '</sent_cs> <sent_cs>'.join(kp_sents).split() + ['</sent_cs>']]
            self.kp_dec_input, self.target_kp = self.get_dec_inp_targ_seqs(kp_ids, hps.kp_max_dec_steps,
                                                                             kp_start_decoding, kp_stop_decoding)
            self.dec_kp_len = len(self.kp_dec_input)

        self.original_op = op_txt
        self.original_evd = evd_txt
        self.original_src = " ".join(src_lst)
        self.original_arg = " ".join(arg_sents)
        self.original_arg_sents = arg_sents
        self.original_kp = " ".join(kp_sents)
        self.original_kp_sents = kp_sents
        return

    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        """Given the reference summary as a sequence of tokens, return the input sequence for the decoder, and the target sequence which we will use to calculate loss. The sequence will be truncated if it is longer than max_len. The input sequence must start with the start_id and the target sequence must end with the stop_id (but not if it's been truncated).

        Args:
          sequence: List of ids (integers)
          max_len: integer
          start_id: integer
          stop_id: integer

        Returns:
          inp: sequence length <=max_len starting with start_id
          target: sequence same length as input, ending with stop_id only if there was no truncation
        """
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len: # truncate
          inp = inp[:max_len]
          target = target[:max_len] # no end_token
        else: # no truncation
          target.append(stop_id) # end token
        assert len(inp) == len(target)
        return inp, target

    def pad_decoder_inp_targ(self, dec_1_max_len, dec_2_max_len, pad_id):
        """Pad decoder input and target sequences with pad_id up to max_len."""
        while len(self.arg_dec_input) < dec_1_max_len:
            self.arg_dec_input.append(pad_id)
        while len(self.target_arg) < dec_1_max_len:
            self.target_arg.append(pad_id)
        if self.hps.model in ["sep_dec", "shd_dec"]:
            while len(self.kp_dec_input) < dec_2_max_len:
                self.kp_dec_input.append(pad_id)
            while len(self.target_kp) < dec_2_max_len:
                self.target_kp.append(pad_id)


    def pad_encoder_input(self, max_len, pad_id):
        """Pad the encoder input sequence with pad_id up to max_len."""
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)


class Batch(object):
  """Class representing a minibatch of train/val/test examples for text summarization."""

  def __init__(self, example_list, hps, src_pad_id, tgt_pad_id):
    """Turns the example_list into a Batch object.

    Args:
       example_list: List of Example objects
       hps: hyperparameters
       vocab: Vocabulary object
    """
    self.src_pad_id = src_pad_id # id of the PAD token used to pad sequences
    self.tgt_pad_id = tgt_pad_id
    self.init_encoder_seq(example_list, hps) # initialize the input to the encoder
    self.init_decoder_seq(example_list, hps) # initialize the input and targets for the decoder
    self.store_orig_strings(example_list, hps) # store the original strings

  def init_encoder_seq(self, example_list, hps):
    # Determine the maximum length of the encoder input sequence in this batch
    max_enc_seq_len = max([ex.enc_len for ex in example_list])

    # Pad the encoder input sequences up to the length of the longest sequence
    for ex in example_list:
      ex.pad_encoder_input(max_enc_seq_len, self.src_pad_id)

    # Initialize the numpy arrays
    # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
    self.enc_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
    self.enc_lens = np.zeros((hps.batch_size), dtype=np.int32)
    self.enc_padding_mask = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.float32)

    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
      self.enc_batch[i, :] = ex.enc_input[:]
      self.enc_lens[i] = ex.enc_len
      for j in range(ex.enc_len):
        self.enc_padding_mask[i][j] = 1


  def init_decoder_seq(self, example_list, hps):
    # Pad the inputs and targets
    for ex in example_list:
      ex.pad_decoder_inp_targ(hps.arg_max_dec_steps, hps.kp_max_dec_steps, self.tgt_pad_id)

    # Initialize the numpy arrays.
    # Note: our decoder inputs and targets must be the same length for each batch (second dimension = max_dec_steps) because we do not use a dynamic_rnn for decoding. However I believe this is possible, or will soon be possible, with Tensorflow 1.0, in which case it may be best to upgrade to that.
    self.arg_dec_batch = np.zeros((hps.batch_size, hps.arg_max_dec_steps), dtype=np.int32)
    self.arg_target_batch = np.zeros((hps.batch_size, hps.arg_max_dec_steps), dtype=np.int32)
    self.arg_dec_padding_mask = np.zeros((hps.batch_size, hps.arg_max_dec_steps), dtype=np.float32)

    if hps.model in ["sep_dec", "shd_dec"]:
        self.kp_dec_batch  = np.zeros((hps.batch_size, hps.kp_max_dec_steps), dtype=np.int32)
        self.kp_target_batch = np.zeros((hps.batch_size, hps.kp_max_dec_steps), dtype=np.int32)
        self.kp_dec_padding_mask = np.zeros((hps.batch_size, hps.kp_max_dec_steps), dtype=np.float32)

    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
      self.arg_dec_batch[i, :] = ex.arg_dec_input[:]
      self.arg_target_batch[i, :] = ex.target_arg[:]
      for j in range(ex.dec_arg_len):
        self.arg_dec_padding_mask[i][j] = 1
      if hps.model in ["sep_dec", "shd_dec"]:
          self.kp_dec_batch[i, :] = ex.kp_dec_input[:]
          self.kp_target_batch[i, :] = ex.target_kp[:]
          for j in range(ex.dec_kp_len):
            self.kp_dec_padding_mask[i][j] = 1

  def store_orig_strings(self, example_list, hps):
    self.original_src = [ex.original_src for ex in example_list] # list of lists
    self.original_arg = [ex.original_arg for ex in example_list] # list of lists
    self.original_arg_sents = [ex.original_arg_sents for ex in example_list] # list of list of lists
    if hps.model in ["sep_dec", "shd_dec"]:
        self.original_kp = [ex.original_kp for ex in example_list]  # list of lists
        self.original_kp_sents = [ex.original_kp_sents for ex in example_list]  # list of list of lists

class Batcher(object):
  BATCH_QUEUE_MAX = 100 # max number of batches the batch_queue can hold

  def __init__(self, data_path, src_vocab, tgt_vocab, hps):
    self._data_path = data_path
    self._src_vocab = src_vocab
    self._tgt_vocab = tgt_vocab
    self._hps = hps
    single_pass = True if hps.mode == "decode" else False
    self._single_pass = single_pass

    # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
    self._batch_queue = queue.Queue(self.BATCH_QUEUE_MAX)
    self._example_queue = queue.Queue(self.BATCH_QUEUE_MAX * self._hps.batch_size)

    # Different settings depending on whether we're in single_pass mode or not
    if single_pass:
      self._num_example_q_threads = 1 # just one thread, so we read through the dataset just once
      self._num_batch_q_threads = 1  # just one thread to batch examples
      self._bucketing_cache_size = 1 # only load one batch's worth of examples before bucketing; this essentially means no bucketing
      self._finished_reading = False # this will tell us when we're finished reading the dataset
    else:
      self._num_example_q_threads = 16 # num threads to fill example queue
      self._num_batch_q_threads = 4  # num threads to fill batch queue
      self._bucketing_cache_size = 100 # how many batches-worth of examples to load into cache before bucketing

    # Start the threads that load the queues
    self._example_q_threads = []
    for _ in range(self._num_example_q_threads):
      self._example_q_threads.append(Thread(target=self.fill_example_queue))
      self._example_q_threads[-1].daemon = True
      self._example_q_threads[-1].start()
    self._batch_q_threads = []
    for _ in range(self._num_batch_q_threads):
      self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
      self._batch_q_threads[-1].daemon = True
      self._batch_q_threads[-1].start()

    # Start a thread that watches the other threads and restarts them if they're dead
    if not single_pass: # We don't want a watcher in single_pass mode because the threads shouldn't run forever
      self._watch_thread = Thread(target=self.watch_threads)
      self._watch_thread.daemon = True
      self._watch_thread.start()


  def next_batch(self):
    # If the batch queue is empty, print a warning
    if self._batch_queue.qsize() == 0:
      tf.logging.warning('Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i', self._batch_queue.qsize(), self._example_queue.qsize())
      if self._single_pass and self._finished_reading:
        tf.logging.info("Finished reading dataset in single_pass mode.")
        return None

    batch = self._batch_queue.get() # get the next Batch
    return batch

  def fill_example_queue(self):
    input_gen = self.text_generator(utils.example_generator(self._data_path, self._single_pass))

    while True:
      try:
        (op, evd, kp, arg) = next(input_gen) # read the next example from file. article and abstract are both strings.
      except StopIteration: # if there are no more examples:
        tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
        if self._single_pass:
          tf.logging.info("single_pass mode is on, so we've finished reading dataset. This thread is stopping!!")
          self._finished_reading = True
          break
        else:
          raise Exception("single_pass mode is off but the example generator is out of data; error.")

      kp_sents = [sent.strip() for sent in utils.split_sent(kp, "kp")]
      arg_sents = [sent.strip() for sent in utils.split_sent(arg, "arg")]
      example = DataSample(op, evd, kp_sents, arg_sents, self._src_vocab, self._tgt_vocab, self._hps) # Process into an Example.
      self._example_queue.put(example) # place the Example in the example queue.


  def fill_batch_queue(self):
    while True:
      if self._hps.mode != 'decode':
        # Get bucketing_cache_size-many batches of Examples into a list, then sort
        inputs = []
        for _ in range(self._hps.batch_size * self._bucketing_cache_size):
          inputs.append(self._example_queue.get())
        inputs = sorted(inputs, key=lambda inp: inp.enc_len) # sort by length of encoder sequence

        # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
        batches = []
        for i in range(0, len(inputs), self._hps.batch_size):
          batches.append(inputs[i:i + self._hps.batch_size])
        if not self._single_pass:
          shuffle(batches)
        for b in batches:  # each b is a list of Example objects
          self._batch_queue.put(Batch(b, self._hps, self._src_vocab.word2id(utils.PAD_TOKEN), self._tgt_vocab.word2id(utils.PAD_TOKEN)))

      else: # beam search decode mode
        ex = self._example_queue.get()
        b = [ex for _ in range(self._hps.batch_size)]
        self._batch_queue.put(Batch(b, self._hps, self._src_vocab.word2id(utils.PAD_TOKEN), self._tgt_vocab.word2id(utils.PAD_TOKEN)))


  def watch_threads(self):
    while True:
      time.sleep(60)
      for idx,t in enumerate(self._example_q_threads):
        if not t.is_alive(): # if the thread is dead
          tf.logging.error('Found example queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_example_queue)
          self._example_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()
      for idx,t in enumerate(self._batch_q_threads):
        if not t.is_alive(): # if the thread is dead
          tf.logging.error('Found batch queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_batch_queue)
          self._batch_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()


  def text_generator(self, example_generator):
    while True:
      e = next(example_generator) # e is a tf.Example
      try:
        op_text = e.features.feature['op'].bytes_list.value[0].decode()
        evd_text = e.features.feature['evd'].bytes_list.value[0].decode()
        kp_text = e.features.feature['kp'].bytes_list.value[0].decode()
        arg_text = e.features.feature['arg'].bytes_list.value[0].decode()
      except ValueError:
        tf.logging.error('Failed to get article or abstract from example')
        continue
      if len(op_text)==0:
        tf.logging.warning('Found an example with empty article text. Skipping it.')
      else:
        yield ( op_text, evd_text, kp_text, arg_text )
