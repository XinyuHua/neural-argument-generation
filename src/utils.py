import tensorflow as tf
import time
import os
import glob
import random
import struct
import numpy as np
from tensorflow.core.example import example_pb2


# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
KP_START_DECODING = '[KP_START]' # This has a vocab id, which is used at the start of every decoder input sequence
KP_STOP_DECODING = '[KP_STOP]' # This has a vocab id, which is used at the end of untruncated target sequences
ARG_START_DECODING = '[ARG_START]' # This has a vocab id, which is used at the start of every decoder input sequence
ARG_STOP_DECODING = '[ARG_STOP]'

# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.

def get_config():
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth=True
  return config

def load_ckpt(args, saver, sess, ckpt_dir="train", ckpt_id=None):
  while True:
    try:
      if ckpt_id is None or ckpt_id == -1 :
        latest_filename = "checkpoint_best" if ckpt_dir=="eval" else None
        ckpt_dir = os.path.join(args.model_path, ckpt_dir)
        ckpt_state = tf.train.get_checkpoint_state(ckpt_dir, latest_filename=latest_filename)
        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        ckpt_path = ckpt_state.model_checkpoint_path
      else:
        ckpt_path = os.path.join(args.model_path, ckpt_dir, "model.ckpt-" + str(ckpt_id))
      saver.restore(sess, ckpt_path)
      return ckpt_path
    except:
      tf.logging.info("Failed to load checkpoint from %s. Sleeping for %i secs...", ckpt_dir, 10)
      time.sleep(10)


class Vocab(object):

  def __init__(self, vocab_file, max_size):
    self._word_to_id = {}
    self._id_to_word = {}
    self._count = 0 # keeps track of total number of words in the Vocab

    # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
    for w in [UNKNOWN_TOKEN, PAD_TOKEN, ARG_START_DECODING, ARG_STOP_DECODING, KP_START_DECODING, KP_STOP_DECODING]:
      self._word_to_id[w] = self._count
      self._id_to_word[self._count] = w
      self._count += 1

    # Read the vocab file and add words up to max_size
    with open(vocab_file, 'r') as vocab_f:
      for line in vocab_f:
        pieces = line.strip().split()

        if len(pieces) != 2:
          print('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
          continue
        w = pieces[0]
        if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN,  ARG_START_DECODING, ARG_STOP_DECODING, KP_START_DECODING, KP_STOP_DECODING]:
          raise Exception('<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
        if w in self._word_to_id:
          raise Exception('Duplicated word in vocabulary file: %s' % w)
        self._word_to_id[w] = self._count
        self._id_to_word[self._count] = w
        self._count += 1
        if max_size != 0 and self._count >= max_size:
          print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
          break

    print("Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count-1]))
    # self.load_stopwords()
    print("top 10 words:")
    for i in range(10):
      print(i, self._id_to_word[i])

  def word2id(self, word):
    if word not in self._word_to_id:
      return self._word_to_id[UNKNOWN_TOKEN]
    return self._word_to_id[word]

  def id2word(self, word_id):
    if word_id not in self._id_to_word:
      raise ValueError('Id not found in vocab: %d' % word_id)
    return self._id_to_word[word_id]

  # def load_stopwords(self):
  #   self.stopword_wids = set()
  #   for line in open("/data/xinyu/general_corpus/stop_words.txt"):
  #     token = line.strip()
  #     if token in self._word_to_id:
  #       self.stopword_wids.add(self._word_to_id[token])
  #   return

  def size(self):
    return self._count



def example_generator(data_path, single_pass):

  while True:
    filelist = glob.glob(data_path) # get the list of datafiles
    assert filelist, ('Error: Empty filelist at %s' % data_path) # check filelist isn't empty
    if single_pass:
      filelist = sorted(filelist)
      print(filelist)
    else:
      random.shuffle(filelist)
    for f in filelist:
      reader = open(f, 'rb')
      while True:
        len_bytes = reader.read(8)
        if not len_bytes: break # finished reading this file
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        yield example_pb2.Example.FromString(example_str)
    if single_pass:
      print("example_generator completed reading all datafiles. No more data.")
      break


def article2ids(article_words, vocab):
  ids = []
  oovs = []
  unk_id = vocab.word2id(UNKNOWN_TOKEN)
  for w in article_words:
    i = vocab.word2id(w)
    if i == unk_id: # If w is OOV
      if w not in oovs: # Add to list of OOVs
        oovs.append(w)
      oov_num = oovs.index(w) # This is 0 for the first article OOV, 1 for the second article OOV...
      ids.append(vocab.size() + oov_num) # This is e.g. 50000 for the first article OOV, 50001 for the second...
    else:
      ids.append(i)
  return ids, oovs


def abstract2ids(abstract_words, vocab, article_oovs):
  ids = []
  unk_id = vocab.word2id(UNKNOWN_TOKEN)
  for w in abstract_words:
    i = vocab.word2id(w)
    if i == unk_id: # If w is an OOV word
      if w in article_oovs: # If w is an in-article OOV
        vocab_idx = vocab.size() + article_oovs.index(w) # Map to its temporary article OOV number
        ids.append(vocab_idx)
      else: # If w is an out-of-article OOV
        ids.append(unk_id) # Map to the UNK token id
    else:
      ids.append(i)
  return ids


def outputids2words(id_list, vocab, article_oovs):
  words = []
  for i in id_list:
    try:
      w = vocab.id2word(i) # might be [UNK]
    except ValueError as e: # w is OOV
      assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
      article_oov_idx = i - vocab.size()
      try:
        w = article_oovs[article_oov_idx]
      except ValueError as e: # i doesn't correspond to an article oov
        raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (i, article_oov_idx, len(article_oovs)))
    words.append(w)
  return words

def outputids2words_withcopy(id_list, vocab, attn_dists, ref, show=True):
    ref = ref.split()
    if len(attn_dists[0]) == 2:
        attn_dists= zip(*attn_dists)[0]
    assert len(attn_dists) == len(id_list)

    words = []
    for ind,i in enumerate(id_list):
        try:
          w = vocab.id2word(i) # might be [UNK]
        except ValueError as e: # w is OOV
          raise ValueError("Error: model produced a word ID that isn't in the vocabulary.")
        if w == UNKNOWN_TOKEN:
            copy = np.argmax(attn_dists[ind])
            w = ref[copy]
            if show:
                w = "__%s__" % w
        words.append(w)
    return words

def abstract2sents(abstract):
  cur = 0
  sents = []
  while True:
    try:
      start_p = abstract.index(SENTENCE_START, cur)
      end_p = abstract.index(SENTENCE_END, start_p + 1)
      cur = end_p + len(SENTENCE_END)
      sents.append(abstract[start_p+len(SENTENCE_START):end_p])
    except ValueError as e: # no more sentences
      return sents

def split_sent(input, arg_or_kp):
  if arg_or_kp == "arg":
    START = "<sent>"
    END = "</sent>"
  else:
    START = "<sent_cs>"
    END = "</sent_cs>"
  cur = 0
  sents = []
  while True:
    try:
      start_p = input.index(START, cur)
      end_p = input.index(END, start_p + 1)
      cur = end_p + len(END)
      sents.append(input[start_p+len(START):end_p])
    except ValueError as e: # no more sentences
      return sents


def show_art_oovs(article, vocab):
  unk_token = vocab.word2id(UNKNOWN_TOKEN)
  words = article.split(' ')
  words = [("__%s__" % w) if vocab.word2id(w)==unk_token else w for w in words]
  out_str = ' '.join(words)
  return out_str

def show_abs_oovs(abstract, vocab, article_oovs):
  unk_token = vocab.word2id(UNKNOWN_TOKEN)
  words = abstract.split(' ')
  new_words = []
  for w in words:
    if vocab.word2id(w) == unk_token: # w is oov
      if article_oovs is None: # baseline mode
        new_words.append("__%s__" % w)
      else: # pointer-generator mode
        if w in article_oovs:
          new_words.append("__%s__" % w)
        else:
          new_words.append("!!__%s__!!" % w)
    else: # w is in-vocab word
      new_words.append(w)
  out_str = ' '.join(new_words)
  return out_str