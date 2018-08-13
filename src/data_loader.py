# Author: Xinyu Hua <hua.x@husky.neu.edu>
# Last modified: 2018-07-30

from random import shuffle
import Queue
import tensorflow as tf
import time
import os
import sys
import utils
import numpy as np

class DataSample(object):

    def __init__(self, src_txt, tgt_txt, src_vocab, tgt_vocab, hps):
        """
        Args:
            src_txt: tokenized string of source input, each token is separated by one single space.
            tgt_txt: 
            src_vocab: Vocabulary for source input.
            tgt_vocab: Vocabulary for target output.
            hps: hyperparameters.
        """
        self.hps = hps

        start_decoding = tgt_vocab.word2id()
        stop_decoding = tgt_vocab.word2id()

        # tokenize the source input
        src_lst = src_txt.split()
        if len(src_lst) > self.hps.max_enc_steps:
            src_lst = src_lst[: self.hps.max_enc_steps]
        self.enc_lens = len(src_lst)
        self.enc_input_wids = [src_vocab.word2id(tok) for tok in src_lst]

        # tokenize the target output
        tgt_str = ' '.join(tgt_txt)
        tgt_lst = tgt_str.split()
        tgt_ids = [tgt_vocab.word2id(tok) for tok in tgt_lst]

        self.dec_input_wids, self.target = self.get_dec_inp_targ_seqs(tgt_ids, hps.max_dec_steps, start_decoding, stop_decoding)
        self.dec_lens = len(self.dec_input_wids)

        self.original_src = src_txt
        self.original_tgt = tgt_str
        self.original_tgt_sents = tgt_txt

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


    def pad_decoder_inp_targ(self, max_len, pad_id):
        """Pad decoder input and target sequences with pad_id up to max_len."""
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
          self.target.append(pad_id)


    def pad_encoder_input(self, max_len, pad_id):
        """Pad the encoder input sequence with pad_id up to max_len."""
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)




class Batch(object):
  """A class to generate minibatches of data. Buckets examples together based on length of the encoder sequence."""

  BATCH_QUEUE_MAX = 100 # max number of batches the batch_queue can hold

  def __init__(self, data_path, src_vocab, tgt_vocab, hps, single_pass):
    """Initialize the batcher. Start threads that process the data into batches.

    Args:
      data_path: tf.Example filepattern.
      vocab: Vocabulary object
      hps: hyperparameters
      single_pass: If True, run through the dataset exactly once (useful for when you want to run evaluation on the dev or test set). Otherwise generate random batches indefinitely (useful for training).
    """
    self._data_path = data_path
    self._src_vocab = src_vocab
    self._tgt_vocab = tgt_vocab
    self._hps = hps
    self._single_pass = single_pass

    # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
    self._batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
    self._example_queue = Queue.Queue(self.BATCH_QUEUE_MAX * self._hps.batch_size)

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
    for _ in xrange(self._num_example_q_threads):
      self._example_q_threads.append(Thread(target=self.fill_example_queue))
      self._example_q_threads[-1].daemon = True
      self._example_q_threads[-1].start()
    self._batch_q_threads = []
    for _ in xrange(self._num_batch_q_threads):
      self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
      self._batch_q_threads[-1].daemon = True
      self._batch_q_threads[-1].start()

    # Start a thread that watches the other threads and restarts them if they're dead
    if not single_pass: # We don't want a watcher in single_pass mode because the threads shouldn't run forever
      self._watch_thread = Thread(target=self.watch_threads)
      self._watch_thread.daemon = True
      self._watch_thread.start()


  def next_batch(self):
    """Return a Batch from the batch queue.

    If mode='decode' then each batch contains a single example repeated beam_size-many times; this is necessary for beam search.

    Returns:
      batch: a Batch object, or None if we're in single_pass mode and we've exhausted the dataset.
    """
    # If the batch queue is empty, print a warning
    if self._batch_queue.qsize() == 0:
      tf.logging.warning('Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i', self._batch_queue.qsize(), self._example_queue.qsize())
      if self._single_pass and self._finished_reading:
        tf.logging.info("Finished reading dataset in single_pass mode.")
        return None

    batch = self._batch_queue.get() # get the next Batch
    return batch

  def fill_example_queue(self):
    """Reads data from file and processes into Examples which are then placed into the example queue."""

    input_gen = self.text_generator(utils.example_generator(self._data_path, self._single_pass))

    while True:
      try:
        (article, abstract) = input_gen.next() # read the next example from file. article and abstract are both strings.
      except StopIteration: # if there are no more examples:
        tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
        if self._single_pass:
          tf.logging.info("single_pass mode is on, so we've finished reading dataset. This thread is stopping!!")
          self._finished_reading = True
          break
        else:
          raise Exception("single_pass mode is off but the example generator is out of data; error.")

      abstract_sentences = [sent.strip() for sent in utils.abstract2sents(abstract)] # Use the <s> and </s> tags in abstract to get a list of sentences.
      example = DataSample(article, abstract_sentences, self._src_vocab, self._tgt_vocab, self._hps) # Process into an Example.
      self._example_queue.put(example) # place the Example in the example queue.


  def fill_batch_queue(self):
    """Takes Examples out of example queue, sorts them by encoder sequence length, processes into Batches and places them in the batch queue.

    In decode mode, makes batches that each contain a single example repeated.
    """
    while True:
      if self._hps.mode != 'decode':
        # Get bucketing_cache_size-many batches of Examples into a list, then sort
        inputs = []
        for _ in xrange(self._hps.batch_size * self._bucketing_cache_size):
          inputs.append(self._example_queue.get())
        inputs = sorted(inputs, key=lambda inp: inp.enc_len) # sort by length of encoder sequence

        # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
        batches = []
        for i in xrange(0, len(inputs), self._hps.batch_size):
          batches.append(inputs[i:i + self._hps.batch_size])
        if not self._single_pass:
          shuffle(batches)
        for b in batches:  # each b is a list of Example objects
          self._batch_queue.put(Batch(b, self._hps, self._src_vocab.word2id(utils.PAD_TOKEN), self._tgt_vocab.word2id(utils.PAD_TOKEN)))

      else: # beam search decode mode
        ex = self._example_queue.get()
        b = [ex for _ in xrange(self._hps.batch_size)]
        self._batch_queue.put(Batch(b, self._hps, self._src_vocab.word2id(utils.PAD_TOKEN), self._tgt_vocab.word2id(utils.PAD_TOKEN)))


  def watch_threads(self):
    """Watch example queue and batch queue threads and restart if dead."""
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
    """Generates article and abstract text from tf.Example.

    Args:
      example_generator: a generator of tf.Examples from file. See data.example_generator"""
    while True:
      e = example_generator.next() # e is a tf.Example
      try:
        article_text = e.features.feature['article'].bytes_list.value[0] # the article text was saved under the key 'article' in the data files
        abstract_text = e.features.feature['abstract'].bytes_list.value[0] # the abstract text was saved under the key 'abstract' in the data files
      except ValueError:
        tf.logging.error('Failed to get article or abstract from example')
        continue
      if len(article_text)==0: # See https://github.com/abisee/pointer-generator/issues/1
        tf.logging.warning('Found an example with empty article text. Skipping it.')
      else:
        yield (article_text, abstract_text)
