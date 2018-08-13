import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2


dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

all_train_urls = "url_lists/all_train.txt"
all_val_urls = "url_lists/all_val.txt"
all_test_urls = "url_lists/all_test.txt"

cnn_tokenized_stories_dir = "cnn_stories_tokenized"
dm_tokenized_stories_dir = "dm_stories_tokenized"
finished_files_dir = "dat"
chunks_dir = os.path.join(finished_files_dir, "chunked")

# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506

VOCAB_SIZE = 200000
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data

def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."


def create_bin(dataset, input_dir, out_file, makevocab=False):
  """src input file and tgt input file should be tokenized, one sample per line."""
  if makevocab:
    vocab_counter_src = collections.Counter()
    vocab_counter_tgt = collections.Counter()
  src_input_file_path = input_dir + dataset + ".src"
  tgt_input_file_path = input_dir + dataset + "_arg.tgt"
  src_data = [fix_missing_period(line.strip().lower()) for line in open(src_input_file_path)]
  tgt_data = [SENTENCE_START + " " + fix_missing_period(line.strip().lower()) + " " + SENTENCE_END for line in open(tgt_input_file_path)]
  assert len(src_data) == len(tgt_data), "src and tgt lengths are different!"

  with open(out_file, "wb") as writer:
    for idx, src in enumerate(src_data):
      tgt = tgt_data[idx]

      # Write to tf.Example
      tf_example = example_pb2.Example()
      tf_example.features.feature['article'].bytes_list.value.extend([src])
      tf_example.features.feature['abstract'].bytes_list.value.extend([tgt])
      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))
      # Write the vocab to file, if applicable
      if makevocab:
        art_tokens = src.split()
        abs_tokens = tgt.split()
        abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
        tokens = art_tokens
        tokens = [t.strip() for t in tokens]  # strip
        tokens = [t for t in tokens if t != ""]  # remove empty
        vocab_counter_src.update(tokens)
        tokens = abs_tokens
        tokens = [t.strip() for t in tokens]  # strip
        tokens = [t for t in tokens if t != ""]  # remove empty
        vocab_counter_tgt.update(tokens)
        # write vocab to file
  if makevocab:
      print
      "Writing vocab file..."
      with open(os.path.join(finished_files_dir, "vocab." + dataset + ".src"), 'w') as writer:
        for word, count in vocab_counter_src.most_common(VOCAB_SIZE):
          writer.write(word + ' ' + str(count) + '\n')
      print
      with open(os.path.join(finished_files_dir, "vocab." + dataset + ".tgt"), 'w') as writer:
        for word, count in vocab_counter_tgt.most_common(VOCAB_SIZE):
          writer.write(word + ' ' + str(count) + '\n')
      print
      "Finished writing vocab file"

def process():
    for d_split in ["train", "valid"]:
        dataset = d_split + "_core_sample3"
        input_path = "../data/trainable/"
        create_bin(dataset=dataset, input_dir=input_path, out_file=input_path + "bin/" + dataset + ".bin", makevocab=True)

def process_demo():
    dataset = "train5.demo_toy"
    input_path = "/data/xinyu/cmv/generation/counterarg_generation/trainable/demo_toy/seq2seq_toy/"
    create_bin(dataset=dataset, input_dir=input_path, out_file=input_path + "pg_format/" + dataset + ".bin", makevocab=False)



if __name__ == '__main__':
    process()
