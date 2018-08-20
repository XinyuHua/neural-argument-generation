"""Definition of data samples and batcher."""
import time
from random import shuffle
from threading import Thread
import queue
import numpy as np
import tensorflow as tf
import utils

class DataSample(object):
    """Class represents each data sample."""

    def __init__(self, op_txt, evd_txt, kp_sents, arg_sents, src_vocab, tgt_vocab, hps):
        self.hps = hps
        arg_start_decoding = tgt_vocab.word2id(utils.ARG_START_DECODING)
        arg_stop_decoding = tgt_vocab.word2id(utils.ARG_STOP_DECODING)
        kp_start_decoding = tgt_vocab.word2id(utils.KP_START_DECODING)
        kp_stop_decoding = tgt_vocab.word2id(utils.KP_STOP_DECODING)

        op_lst = op_txt.split()
        evd_lst = evd_txt.split()
        src_lst = op_lst
        if hps.model == "base_enc_evd":
            src_lst += ["<ctx>"] + evd_lst
        elif hps.model == "base_enc_kp":
            src_lst += ["<ctx>", "<sent_ctx>"] \
                       + "</sent_ctx> <sent_ctx>".join(kp_sents).split() \
                       + ["</sent_ctx>"]

        if len(src_lst) > self.hps.max_enc_steps:
            src_lst = src_lst[: self.hps.max_enc_steps]
        self.enc_len = len(src_lst)
        self.enc_input = [src_vocab.word2id(tok) for tok in src_lst]

        arg_ids = [tgt_vocab.word2id(tok) for tok in ' '.join(arg_sents).split()]

        def get_dec_inp_targ_seqs(sequence, max_len, start_id, stop_id):
            inp = [start_id] + sequence[:]
            target = sequence[:]
            if len(inp) > max_len:
                inp = inp[:max_len]
                target = target[:max_len]
            else:
                target.append(stop_id)
            assert len(inp) == len(target)
            return inp, target


        self.arg_dec_input, self.target_arg = \
            get_dec_inp_targ_seqs(arg_ids, hps.arg_max_dec_steps,
                                  arg_start_decoding, arg_stop_decoding)
        self.dec_arg_len = len(self.arg_dec_input)

        if hps.model in ["sep_dec", "shd_dec"]:
            kp_ids = [tgt_vocab.word2id(tok) for tok in
                      ['<sent_cs>']
                      + '</sent_cs> <sent_cs>'.join(kp_sents).split()
                      + ['</sent_cs>']]
            self.kp_dec_input, self.target_kp = \
                get_dec_inp_targ_seqs(kp_ids, hps.kp_max_dec_steps,
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



    def pad_decoder_inp_targ(self, dec_1_max_len, dec_2_max_len, pad_id):
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
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)


class Batch(object):
    """Class representing a minibatch of train/val/test examples for text summarization."""

    def __init__(self, example_list, hps, src_pad_id, tgt_pad_id):
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.init_encoder_seq(example_list, hps)
        self.init_decoder_seq(example_list, hps)
        self.store_orig_strings(example_list, hps)

    def init_encoder_seq(self, example_list, hps):
        max_enc_seq_len = max([ex.enc_len for ex in example_list])

        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.src_pad_id)

        self.enc_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros((hps.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.float32)

        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1


    def init_decoder_seq(self, example_list, hps):
        for ex in example_list:
            ex.pad_decoder_inp_targ(hps.arg_max_dec_steps, hps.kp_max_dec_steps, self.tgt_pad_id)

        self.arg_dec_batch = np.zeros(
            (hps.batch_size, hps.arg_max_dec_steps), dtype=np.int32)
        self.arg_target_batch = np.zeros(
            (hps.batch_size, hps.arg_max_dec_steps), dtype=np.int32)
        self.arg_dec_padding_mask = np.zeros(
            (hps.batch_size, hps.arg_max_dec_steps), dtype=np.float32)

        if hps.model in ["sep_dec", "shd_dec"]:
            self.kp_dec_batch = np.zeros(
                (hps.batch_size, hps.kp_max_dec_steps), dtype=np.int32)
            self.kp_target_batch = np.zeros(
                (hps.batch_size, hps.kp_max_dec_steps), dtype=np.int32)
            self.kp_dec_padding_mask = np.zeros(
                (hps.batch_size, hps.kp_max_dec_steps), dtype=np.float32)

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
        self.original_src = [ex.original_src for ex in example_list]
        self.original_arg = [ex.original_arg for ex in example_list]
        self.original_arg_sents = [ex.original_arg_sents for ex in example_list]
        if hps.model in ["sep_dec", "shd_dec"]:
            self.original_kp = [ex.original_kp for ex in example_list]
            self.original_kp_sents = [ex.original_kp_sents for ex in example_list]

class Batcher(object):
    """Batcher class, creating batches from binarized data and yield batches when called."""
    BATCH_QUEUE_MAX = 100

    def __init__(self, data_path, src_vocab, tgt_vocab, hps):
        self._data_path = data_path
        self._src_vocab = src_vocab
        self._tgt_vocab = tgt_vocab
        self._hps = hps
        single_pass = True if hps.mode == "decode" else False
        self._single_pass = single_pass

        self._batch_queue = queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = queue.Queue(self.BATCH_QUEUE_MAX * self._hps.batch_size)

        if single_pass:
            self._num_example_q_threads = 1
            self._num_batch_q_threads = 1
            self._bucketing_cache_size = 1
            self._finished_reading = False
        else:
            self._num_example_q_threads = 16
            self._num_batch_q_threads = 4
            self._bucketing_cache_size = 100

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

        if not single_pass:
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()


    def next_batch(self):
        if self._batch_queue.qsize() == 0:
            tf.logging.warning('Bucket input queue is empty when calling next_batch. '
                               'Bucket queue size: %i, Input queue size: %i',
                               self._batch_queue.qsize(), self._example_queue.qsize())
            if self._single_pass and self._finished_reading:
                tf.logging.info("Finished reading dataset in single_pass mode.")
                return None

        batch = self._batch_queue.get()
        return batch

    def fill_example_queue(self):
        input_gen = text_generator(utils.example_generator(self._data_path, self._single_pass))

        while True:
            try:
                (op, evd, kp, arg) = next(input_gen)
            except StopIteration:
                tf.logging.info("The example generator for this example "
                                "queue filling thread has exhausted data.")
                if self._single_pass:
                    tf.logging.info("Finished reading dataset. This thread is stopping!!")
                    self._finished_reading = True
                    break
                else:
                    raise Exception("The example generator is out of data; error.")

            kp_sents = [sent.strip() for sent in utils.split_sent(kp, "kp")]
            arg_sents = [sent.strip() for sent in utils.split_sent(arg, "arg")]
            example = DataSample(op, evd, kp_sents, arg_sents,
                                 self._src_vocab, self._tgt_vocab, self._hps)
            self._example_queue.put(example)


    def fill_batch_queue(self):
        while True:
            if self._hps.mode != 'decode':
                inputs = []
                for _ in range(self._hps.batch_size * self._bucketing_cache_size):
                    inputs.append(self._example_queue.get())
                inputs = sorted(inputs, key=lambda inp: inp.enc_len)
                batches = []
                for i in range(0, len(inputs), self._hps.batch_size):
                    batches.append(inputs[i:i + self._hps.batch_size])
                if not self._single_pass:
                    shuffle(batches)
                for b in batches:
                    self._batch_queue.put(Batch(b, self._hps,
                                                self._src_vocab.word2id(utils.PAD_TOKEN),
                                                self._tgt_vocab.word2id(utils.PAD_TOKEN)))

            else:
                ex = self._example_queue.get()
                b = [ex for _ in range(self._hps.batch_size)]
                self._batch_queue.put(Batch(b, self._hps, self._src_vocab.word2id(utils.PAD_TOKEN),
                                            self._tgt_vocab.word2id(utils.PAD_TOKEN)))


    def watch_threads(self):
        while True:
            time.sleep(60)
            for idx, t in enumerate(self._example_q_threads):
                if not t.is_alive():
                    tf.logging.error('Found example queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_example_queue)
                    self._example_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
            for idx, t in enumerate(self._batch_q_threads):
                if not t.is_alive():
                    tf.logging.error('Found batch queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()


def text_generator(example_generator):
    while True:
        e = next(example_generator)
        try:
            op_text = e.features.feature['op'].bytes_list.value[0].decode()
            evd_text = e.features.feature['evd'].bytes_list.value[0].decode()
            kp_text = e.features.feature['kp'].bytes_list.value[0].decode()
            arg_text = e.features.feature['arg'].bytes_list.value[0].decode()
        except ValueError:
            tf.logging.error('Failed to get article or abstract from example')
            continue
        if len(op_text) == 0:
            tf.logging.warning('Found an example with empty article text. Skipping it.')
        else:
            yield (op_text, evd_text, kp_text, arg_text)
