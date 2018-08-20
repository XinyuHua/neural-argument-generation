import tensorflow as tf
import numpy as np
import utils


class Hypothesis(object):
    def __init__(self, tokens, log_probs, state, attn_dists, dec_states=None, tgt_2=None):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.attn_dists = attn_dists
        self.tgt_2 = tgt_2
        self.dec_states = dec_states

    def extend(self, token, log_prob, state, attn_dist, dec_state):
        if dec_state is not None:
            if self.dec_states is not None:
                dec_state = np.vstack([self.dec_states, dec_state])
        else:
            dec_state = self.dec_states

        return Hypothesis(tokens=self.tokens + [token],
                          log_probs=self.log_probs + [log_prob],
                          state=state,
                          attn_dists=self.attn_dists + [attn_dist],
                          dec_states=dec_state,
                          tgt_2=self.tgt_2)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def log_prob(self):
        # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
        return self.log_prob / len(self.tokens)


def beam_search_hyps(sess, model, vocab, batch, dec_in_state, enc_states, dec_2_dec_states, dec_2, max_dec_steps, arm, num_tgt_2_hyps=0):
    # Initialize beam_size-many hyptheses
    start_tok = utils.ARG_START_DECODING if arm == 1 else utils.KP_START_DECODING
    stop_tok = utils.ARG_STOP_DECODING if arm == 1 else utils.KP_STOP_DECODING

    hyps = [Hypothesis(tokens=[vocab.word2id(start_tok)], log_probs=[0.0], state=dec_in_state[i], attn_dists=[],
                       tgt_2=dec_2[i], dec_states=dec_2_dec_states[i]) for i in range(model.hps.beam_size)]
    results = []  # this will contain finished hypotheses (those that have emitted the [STOP] token)
    steps = 0
    while steps < max_dec_steps and len(results) < model.hps.beam_size:
        latest_tokens = [h.latest_token for h in hyps]  # latest token produced by each hypothesis

        states = [h.state for h in hyps]  # list of current decoder states of the hypotheses

        if arm == 1 and hyps[0].dec_states is not None:
            dec_2_dec_states = (np.stack([h.dec_states[0] for h in hyps]), np.stack([h.dec_states[1] for h in hyps]))

        # Run one step of the decoder to get the new info
        (topk_ids, topk_log_probs, new_states, attn_dists, dec_states) = model.decode_onestep(sess=sess,
                                                                                              batch=batch,
                                                                                              latest_tokens=latest_tokens,
                                                                                              enc_states=enc_states,
                                                                                              kp_dec_states=dec_2_dec_states,
                                                                                              dec_init_states=states,
                                                                                              arm=arm,
                                                                                              first_step=len(hyps[0].tokens) == 1)
        # Extend each hypothesis and collect them all in all_hyps
        all_hyps = []
        # On the first step, we only had one original hypothesis (the initial hypothesis). On subsequent steps, all original hypotheses are distinct.
        if steps == 0 and num_tgt_2_hyps == 0:
            num_orig_hyps = 1
        elif steps == 0 and num_tgt_2_hyps > 0:
            num_orig_hyps = num_tgt_2_hyps
        else:
            num_orig_hyps = len(hyps)

        for i in range(num_orig_hyps):
            h, new_state, attn_dist, dec_state = hyps[i], new_states[i], attn_dists[i], dec_states[i] # take the ith hypothesis and new decoder state info
            for j in range(model.hps.beam_size * 2):  # for each of the top 2*beam_size hyps:
                # Extend the ith hypothesis with the jth option
                new_hyp = h.extend(token=topk_ids[i, j],
                                   log_prob=topk_log_probs[i, j],
                                   state=new_state,
                                   attn_dist=attn_dist,
                                   dec_state=dec_state)
                all_hyps.append(new_hyp)

        # Filter and collect any hypotheses that have produced the end token.
        hyps = []
        for h in sort_hyps(all_hyps):
            if h.latest_token == vocab.word2id(stop_tok):
                if steps >= model.hps.min_dec_steps:
                    results.append(h)
            else:
                hyps.append(h)
            if len(hyps) == model.hps.beam_size or len(results) == model.hps.beam_size:
                break
        steps += 1
    return hyps, results


def run_beam_search(sess, model, vocab, batch):
    # Run the encoder to get the encoder hidden states and decoder initial state
    enc_states, dec_in_state = model.run_encoder(sess, batch)
    dec_in_state = [dec_in_state] * model.hps.beam_size
    dec_2_dec_states = [None] * model.hps.beam_size
    dec_2 = np.array([None] * model.hps.beam_size)

    tgt_2_hyps, tgt_2_results = beam_search_hyps(sess, model, vocab, batch, dec_in_state, enc_states, dec_2_dec_states, dec_2,
                                                 model.hps.kp_max_dec_steps, arm=2)

    if len(tgt_2_results) == 0:
        tgt_2_results = tgt_2_hyps
    dec_2[range(len(tgt_2_results))] = range(len(tgt_2_results))
    num_tgt_2_hyps = len(tgt_2_results)

    for i, h in enumerate(tgt_2_results):
        dec_in_state[i] = h.state

    if model.hps.attention == "dual":
        dec_2_padded_states = np.zeros((model.hps.beam_size, model.hps.max_dec_2_steps, model.hps.hidden_dim), dtype=np.float32)
        dec_2_padding_mask = np.zeros((model.hps.beam_size, model.hps.max_dec_2_steps), dtype=np.float32)
        for i, h in enumerate(tgt_2_results):
            for j in range(len(h.dec_states)):
                dec_2_padded_states[i, j:] = h.dec_states[j]
                dec_2_padding_mask[i, j] = 1
        dec_2_dec_states = zip(dec_2_padded_states, dec_2_padding_mask)

    tgt_1_hyps, tgt_1_results = beam_search_hyps(sess, model, vocab, batch, dec_in_state, enc_states, dec_2_dec_states, dec_2,
                                                 model.hps.arg_max_dec_steps, arm=1, num_tgt_2_hyps=num_tgt_2_hyps)
    # At this point, either we've got beam_size results, or we've reached maximum decoder steps
    if len(tgt_1_results) == 0:
        tgt_1_results = tgt_1_hyps

    # Sort hypotheses by average log probability
    tgt_1_hyps_sorted = sort_hyps(tgt_1_results)
    best_tgt_1 = tgt_1_hyps_sorted[0]
    best_tgt_2 = tgt_2_results[best_tgt_1.tgt_2]

    return best_tgt_1, best_tgt_2


def sort_hyps(hyps):
    return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)