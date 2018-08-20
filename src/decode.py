import os
import time
import tensorflow as tf
import beam_search
import utils

SECS_UNTIL_NEW_CKPT = 60  # max number of seconds before loading new checkpoint


class BeamSearchDecoder(object):

  def __init__(self, model, batcher, src_vocab, tgt_vocab, ckpt_id, output_beams=False):
    self._model = model
    self._model()
    self._batcher = batcher
    self._src_vocab = src_vocab
    self._tgt_vocab = tgt_vocab
    self.output_beams = output_beams
    self._saver = tf.train.Saver() # we use this to load checkpoints for decoding
    self._sess = tf.Session(config=utils.get_config())

    # Load an initial checkpoint to use for decoding
    ckpt_path = utils.load_ckpt(self._saver, self._sess, "train", ckpt_id)

    ckpt_name = "ckpt-" + ckpt_path.split('-')[-1] # this is something of the form "ckpt-123456"
    self._decode_dir = os.path.join(model.hps.model_path, get_decode_dir_name(ckpt_name, model.hps))
    if os.path.exists(self._decode_dir):
      raise Exception("single_pass decode directory %s should not already exist" % self._decode_dir)


    # Make the decode dir if necessary
    if not os.path.exists(self._decode_dir):
      os.mkdir(self._decode_dir)

    # Make directories for decoding output
    self._ref_dir = os.path.join(self._decode_dir, "reference")
    if not os.path.exists(self._ref_dir): os.mkdir(self._ref_dir)
    self._dec_dir = os.path.join(self._decode_dir, "decoded")
    if not os.path.exists(self._dec_dir): os.mkdir(self._dec_dir)
    self._summary_path = os.path.join(self._decode_dir, "summary.txt")


  def decode(self):
    t0 = time.time()
    counter = 0
    summary_file = open(self._summary_path, "w")
    while True:
      batch = self._batcher.next_batch()  # 1 example repeated across batch
      if batch is None: # finished decoding dataset in single_pass mode
        tf.logging.info("Decoder has finished reading dataset for single_pass.")
        tf.logging.info("Output has been saved in %s and %s", self._ref_dir, self._dec_dir)
        return

      src_withunks = utils.show_art_oovs(batch.original_src[0], self._src_vocab) # string
      arg_withunks = utils.show_abs_oovs(batch.original_arg[0], self._tgt_vocab, None) # string

      best_hyp_arg, best_hyp_kp = beam_search.run_beam_search(self._sess, self._model, self._tgt_vocab, batch)
      # Extract the output ids from the hypothesis and convert back to words
      output_ids = [int(t) for t in best_hyp_arg.tokens[1:]]
      decoded_words = utils.outputids2words(output_ids, self._tgt_vocab, None)
      # decoded_words = data.outputids2words_withcopy(output_ids, self._tgt_vocab, original_article, best_hyp.attn_dists)
      # Remove the [STOP] token from decoded_words, if necessary
      try:
        fst_stop_idx = decoded_words.index(utils.ARG_STOP_DECODING)  # index of the (first) [STOP] symbol
        decoded_words = decoded_words[:fst_stop_idx]
      except ValueError:
        decoded_words = decoded_words

      self.write_to_file(batch.original_arg_sents[0], decoded_words, counter, "arg")

      summary_file.write("ID: %d\n" % counter)
      summary_file.write("OP: %s\n" % batch.original_src)
      summary_file.write("ARG: %s\n" % arg_withunks)
      summary_file.write("Generation: %s\n" % " ".join(decoded_words))
      summary_file.write("=" * 50 + "\n")

      if self._model.hps.model in ["sep_dec", "shd_dec"]:
        kp_withunks = utils.show_abs_oovs(batch.original_kp[0], self._tgt_vocab, None) # string
        output_ids = [int(t) for t in best_hyp_kp.tokens[1:]]
        decoded_words = utils.outputids2words(output_ids, self._tgt_vocab, None)
        # decoded_words = data.outputids2words_withcopy(output_ids, self._tgt_vocab, original_article, best_hyp.attn_dists)
        # Remove the [STOP] token from decoded_words, if necessary
        try:
          fst_stop_idx = decoded_words.index(utils.KP_STOP_DECODING)  # index of the (first) [STOP] symbol
          decoded_words = decoded_words[:fst_stop_idx]
        except ValueError:
          decoded_words = decoded_words

        self.write_to_file(batch.original_kp_sents[0], decoded_words,  counter, "kp")
      counter += 1  # this is how many examples we've decoded

    summary_file.close()
    tf.logging.info("Decoding took %.3f seconds", time.time() - t0)

  def write_to_file(self, reference_sents, decoded_words, ex_index, mode="arg"):
    # First, divide decoded output into sentences
    decoded_sents = []
    while len(decoded_words) > 0:
      try:
        fst_period_idx = decoded_words.index("</sent>")
      except ValueError: # there is text remaining that doesn't end in "."
        fst_period_idx = len(decoded_words)
      sent = decoded_words[:fst_period_idx+1] # sentence up to and including the period
      decoded_words = decoded_words[fst_period_idx+1:] # everything else
      decoded_sents.append(' '.join(sent))

    # Write to file
    ref_file = os.path.join(self._ref_dir, "%06d_reference_%s.txt" % (ex_index, mode))
    decoded_file = os.path.join(self._dec_dir, "%06d_decoded_%s.txt" % (ex_index, mode))

    with open(ref_file, "w") as f:
      for idx,sent in enumerate(reference_sents):
        f.write(sent) if idx==len(reference_sents)-1 else f.write(sent+"\n")
    with open(decoded_file, "w") as f:
      for idx,sent in enumerate(decoded_sents):
        f.write(sent) if idx==len(decoded_sents)-1 else f.write(sent+"\n")

    tf.logging.info("Wrote example %i, %s to file" % (ex_index, mode))


def get_decode_dir_name(ckpt_name, opt):
  """Make a descriptive name for the decode dir, including the name of the checkpoint we use to decode. This is called in single_pass mode."""

  if "train" in opt.data_path: dataset = "train"
  elif "val" in opt.data_path: dataset = "val"
  elif "test" in opt.data_path: dataset = "test"
  else: raise ValueError("FLAGS.data_path %s should contain one of train, val or test" % (opt.data_path))
  dirname = "decode_%s_%imaxenc_%ibeam_%imindec_%imaxdec" % (dataset, opt.max_enc_steps, opt.beam_size, opt.min_dec_steps, opt.arg_max_dec_steps)
  if ckpt_name is not None:
    dirname += "_%s" % ckpt_name
  return dirname
