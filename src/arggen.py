# Main program for train/eval/decode on neural argument generation models.

import time
import os
import tensorflow as tf
import numpy as np
from data_loader import Batcher
from vanilla_model import VanillaSeq2seqModel
from sep_dec_model import SeparateDecoderModel
from shd_dec_model import SharedDecoderModel
from decode import BeamSearchDecoder
import utils
import argparse

parser = argparse.ArgumentParser(description="Entry for neural argument generation.")

# path for data and model storage
parser.add_argument("--data_path", type=str, default="dat/trainable/bin/train.bin", help="Path to binarized train/valid/test data.")
parser.add_argument("--src_vocab_path", type=str, default="dat/vocab.src", help="Path to source vocabulary.")
parser.add_argument("--tgt_vocab_path", type=str, default="dat/vocab.tgt", help="Path to target vocabulary.")
parser.add_argument("--embed_path", type=str, default="dat/embedding/glove/glove.6B.200d.txt", help="Path to word embedding.")
parser.add_argument("--model_path", type=str, default="dat/log/vanilla", help="Path to store the model checkpoints.")
parser.add_argument("--exp_name", type=str, default="demo", help="Experiment name under model_path.")

# model setups
parser.add_argument("--model", type=str, choices=["vanilla", "sep_dec", "shd_dec"], help="Different types of models, choose from vanilla, sep_dec, and shd_dec.", default="vanilla")
parser.add_argument("--mode", type=str, choices=["train", "eval", "decode"], help="Whether to run train, eval, or decode", default="train")
parser.add_argument("--attention", type=str, choices=["dual", "flat"], help="Types of attention, whether dual or flat.", default="flat")
parser.add_argument("--max_training_steps", type=int, help="After reached this many steps, stop training.", default=10000)
parser.add_argument("--save_model_seconds", type=int, help="Save model checkpoint every this many seconds.", default=60)
parser.add_argument("--ckpt_id", type=int, help="Load this checkpoint.", default=-1)

# other parameters
parser.add_argument("--emb_dim", type=int, help="Dimension of word embedding.", default=200)
parser.add_argument("--hidden_dim", type=int, help="Dimension of hidden state for each layer of RNN.", default=200)
parser.add_argument("--batch_size", type=int, help="Batch size.", default=32)
parser.add_argument("--max_enc_steps", type=int, help="Maximum number of tokens to encode.", default=250)
parser.add_argument("--arg_max_dec_steps", type=int, help="Maximum number of tokens for argument decoder.", default=100)
parser.add_argument("--kp_max_dec_steps", type=int, help="Maximum number of tokens for keyphrase decoder.", default=100)
parser.add_argument("--min_dec_steps", type=int, help="Minimum number of tokens for decoder.", default=10)
parser.add_argument("--beam_size", type=int, help="Beam size during decoding.", default=5)
parser.add_argument("--src_vocab_size", type=int, help="Source vocabulary size.", default=50000)
parser.add_argument("--tgt_vocab_size", type=int, help="Target vocabulary size.", default=50000)
parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=0.001)
parser.add_argument("--optimizer", type=str, choices=["adam", "adagrad"], default="adam", help="Optimizer to use for gradient descent.")
parser.add_argument("--adagrad_init_acc", type=float, default=0.1)
parser.add_argument("--dropout", type=float, help="Dropout probability.", default=0.2)
parser.add_argument("--rand_unif_init_mag", type=float, default=0.02)
parser.add_argument("--trunc_norm_init_std", type=float,  default=0.0001)
parser.add_argument("--max_grad_norm", type=float, help="Norm for gradient clipping.", default=2.0)

args = parser.parse_args()


def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
  if running_avg_loss == 0:
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
  running_avg_loss = min(running_avg_loss, 12)  # clip
  loss_sum = tf.Summary()
  tag_name = 'running_avg_loss/decay=%f' % (decay)
  loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
  summary_writer.add_summary(loss_sum, step)
  tf.logging.info('running_avg_loss: %f', running_avg_loss)
  return running_avg_loss

def run_eval(model, batcher, ckpt_id):
  model()
  saver = tf.train.Saver(max_to_keep=3)
  sess = tf.Session(config=utils.get_config())
  eval_dir = os.path.join(args.model_path, "eval")
  bestmodel_save_path = os.path.join(eval_dir, 'bestmodel')
  summary_writer = tf.summary.FileWriter(eval_dir)
  running_avg_loss = 0
  best_loss = None

  batch_cnt = 0
  while True:
    if not ckpt_id == -1 and batch_cnt > 100:
        break
    batch_cnt += 1
    _ = utils.load_ckpt(args, saver, sess, "train", ckpt_id)
    batch = batcher.next_batch()

    # run eval on the batch
    t0=time.time()
    results = model.run_step(sess, batch)
    t1=time.time()
    tf.logging.info('seconds for batch: %.2f', t1-t0)

    # print the loss and coverage loss to screen
    loss = results['loss']
    tf.logging.info('batch_id: %d (100)\tloss: %f', batch_cnt, loss)

    # add summaries
    summaries = results['summaries']
    train_step = results['global_step']
    summary_writer.add_summary(summaries, train_step)

    # calculate running avg loss
    running_avg_loss = calc_running_avg_loss(np.asscalar(loss), running_avg_loss, summary_writer, train_step)

    # If running_avg_loss is best so far, save this checkpoint (early stopping).
    # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir
    if best_loss is None or running_avg_loss < best_loss:
      tf.logging.info('Found new best model with %.3f running_avg_loss. Saving to %s', running_avg_loss, bestmodel_save_path)
      saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
      best_loss = running_avg_loss

    # flush the summary writer every so often
    if train_step % 100 == 0:
      summary_writer.flush()

def setup_training(model, data_loader):
  train_dir = os.path.join(args.model_path, "train")
  if not os.path.exists(train_dir): os.makedirs(train_dir)

  model()

  saver = tf.train.Saver(max_to_keep=20)

  sv = tf.train.Supervisor(logdir=train_dir,
                     is_chief=True,
                     saver=saver,
                     summary_op=None,
                     save_summaries_secs=60,
                     save_model_secs=args.save_model_seconds,
                     global_step=model.global_step)
  summary_writer = sv.summary_writer
  tf.logging.info("Preparing or waiting for session...")
  sess_context_manager = sv.prepare_or_wait_for_session(config=utils.get_config())
  tf.logging.info("Created session.")
  try:
    run_training(model, data_loader, sess_context_manager, summary_writer) # this is an infinite loop until interrupted
  except KeyboardInterrupt:
    tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
    sv.stop()


def run_training(model, data_loader, sess_context_manager, summary_writer):
  tf.logging.info("starting run_training")
  with sess_context_manager as sess:
    while True:
      batch = data_loader.next_batch()

      tf.logging.info('running training step...')
      t0=time.time()
      results = model.run_step(sess, batch)
      t1=time.time()
      tf.logging.info('seconds for training step: %.3f', t1-t0)

      loss = results['loss']
      tf.logging.info('loss: %f', loss)

      if not np.isfinite(loss):
        raise Exception("Loss is not finite. Stopping.")

      summaries = results['summaries']
      train_step = results['global_step']

      summary_writer.add_summary(summaries, train_step)
      if train_step % 100 == 0:
        summary_writer.flush()
      if train_step > args.max_training_steps:
        break


def main():

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info('Starting seq2seq_attention in %s mode...', (args.mode))

  args.model_path = os.path.join(args.model_path, args.exp_name)
  if not os.path.exists(args.model_path):
    if args.mode=="train":
      os.makedirs(args.model_path)
    else:
      raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (args.model_path))

  src_vocab = utils.Vocab(args.src_vocab_path, args.src_vocab_size)
  tgt_vocab = utils.Vocab(args.tgt_vocab_path, args.tgt_vocab_size)
  batcher = Batcher(args.data_path, src_vocab, tgt_vocab, args)

  if args.model == "vanilla":
    model_class = VanillaSeq2seqModel
  elif args.model == "sep_dec":
    model_class = SeparateDecoderModel
  elif args.model == "shd_dec":
    model_class = SharedDecoderModel


  tf.set_random_seed(111)

  if args.mode == 'train':
    model = model_class(args, src_vocab, tgt_vocab)
    setup_training(model, batcher)
  elif args.mode == 'eval':
    model = model_class(args, src_vocab, tgt_vocab)
    run_eval(model, batcher, args.ckpt_id)
  elif args.mode == "decode":
    args.batch_size = args.beam_size
    args.arg_max_dec_steps = 1
    args.kp_max_dec_steps = 1
    model = model_class(args, src_vocab, tgt_vocab)
    decoder = BeamSearchDecoder(model, batcher, src_vocab, tgt_vocab, args.ckpt_id)
    decoder.decode()
  else:
    raise ValueError("The 'mode' flag must be one of train/eval/decode")

if __name__ == '__main__':
  main()
