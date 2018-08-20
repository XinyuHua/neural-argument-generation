# Author: Xinyu Hua
# Last modified: 2018-08-13
"""
Process plain text data into binary format.
"""
import struct
import tensorflow as tf
from tensorflow.core.example import example_pb2


def create_bin(path, data_split):
    """
    data should be tokenized.
    Args:
        path: The path to plain text data.
        data_split: A string denoting which split to process, either train or valid.
    """
    print("[*] Creating binary file for %s" % data_split)
    enc_src_path = path + data_split + "_core_sample3.src"
    dec_arg_path = path + data_split + "_core_sample3_arg.tgt"
    dec_kp_path = path + data_split + "_core_sample3_kp.tgt"

    # Read encoder and decoder inputs
    enc_dat = [ln.strip().lower() for ln in open(enc_src_path).readlines()]
    enc_op_dat = [ln.split("<ctx>")[0].strip() for ln in enc_dat]
    enc_evd_dat = [ln.split("<ctx>")[1].strip() for ln in enc_dat]
    dec_arg_dat = [ln.strip().lower() for ln in open(dec_arg_path).readlines()]
    dec_kp_dat = [ln.strip().lower() for ln in open(dec_kp_path).readlines()]

    assert len(enc_dat) == len(dec_arg_dat), "Lengths of encoder source and decoder target do not match!"
    assert len(dec_arg_dat) == len(dec_kp_dat), "Lengths of decoder argument and decoder keyphrase do not match!"

    output_path = path + data_split + ".bin"
    with open(output_path, "wb") as fout:
        for idx, op in enumerate(enc_op_dat):
            evd = enc_evd_dat[idx]
            kp = dec_kp_dat[idx]
            arg = dec_arg_dat[idx]

            # Write to tf.Example
            tf_example = example_pb2.Example()
            tf_example.features.feature['op'].bytes_list.value.extend([op.encode()])
            tf_example.features.feature['evd'].bytes_list.value.extend([evd.encode()])
            tf_example.features.feature['kp'].bytes_list.value.extend([kp.encode()])
            tf_example.features.feature['arg'].bytes_list.value.extend([arg.encode()])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            fout.write(struct.pack('q', str_len))
            fout.write(struct.pack('%ds' % str_len, tf_example_str))
            if idx > 10:break
    return


if __name__=='__main__':
    create_bin("dat/trainable/", "train")
    create_bin("dat/trainable/", "valid")