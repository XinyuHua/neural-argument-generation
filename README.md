## About

This repository holds the code for Neural Argument Generation project developed at Northeastern NLP. For details about the framework please read our ACL 2018 paper:

* Xinyu Hua and Lu Wang. [Neural Argument Generation Augmented with Externally Retrieved Evidence.](http://xinyuhua.github.io/resources/acl2018/acl2018.pdf) In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL), 2018.
* [Supplementary material](http://xinyuhua.github.io/resources/acl2018/acl2018_supp.pdf)

## Usage
### Requirement

- python 3.5
- tensorflow  1.4.0
- numpy 1.14

### Data
Please download the dataset from [here](https://drive.google.com/file/d/1qyFP9zi9PMvQI7hERoP-X1YHly1dw9Fw/view?usp=sharing).

The dataset consists of the following 5 parts:

1. cmv\_processed: filtered OP posts and root replies used to create the core dataset

2. wikipedia\_retrieval: wikipedia article titles retrieved as evidence source for OP and root replies

3. reranked\_evidence: selected evidence sentences and extracted keyphrases for OP and root replies

4. trainable: directly trainable dataset

5. test: test set we used for evaluation

(Detailed readme file can be found [here](http://xinyuhua.github.io/resources/acl2018/README.txt).)

### File structure
Please download the corresponding data and put them under dat/ folder. If the folder does not exist please create by hand.
```
mkdir dat/log
mkdir -p dat/trainable/bin
```


```
neural-argument-generation/
 ├── src/
 │   ├── arggen.py
 │   ├── attention.py
 │   ├── base_model.py
 │   ├── beam_search.py
 │   ├── data_loader.py
 │   ├── decode.py
 │   ├── sep_dec_model.py
 │   ├── shd_dec_model.py
 │   ├── utils.py
 │   └── vanilla_model.py
 │
 ├── scripts/
 │   ├──  preprocess.py
 │   └──  evaluation.py (coming soon)
 │
 └── dat/
     ├── vocab.src
     ├── vocab.tgt
     ├── trainable/
     │    ├── train_core_sample3.src
     │    ├── train_core_sample3_arg.tgt
     │    ├── train_core_sample3_kp.tgt
     │    ├── valid_core_sample3.src
     │    ├── valid_core_sample3_arg.tgt
     │    ├── valid_core_sample3_kp.tgt
     │    └── bin/
     └── log/
```


### Preprocessing
This step binarizes the plain text data. Please make sure the plain text data files are in order.

```
python3 scripts/preprocess.py
```

### Training and concurrent validation
Train the model by assigning ```--mode=train```. While the model is training, start another thread by assigning ```--mode=eval``` for concurrent validation. The summaries on loss will be logged into the same exp folder. These results can be visualized by tensorboard.

```
python3 src/arggen.py [--mode={train,eval}] [--model={vanilla,seq_dec,shd_dec}] \
                      [--data_path=PATH_TO_BIN_DATA] \
                      [--model_path=PATH_TO_STORE_MODEL] \
                      [--exp_name=EXP_NAME] \
                      [--batch_size=BS] \
                      [--src_vocab_path=PATH_TO_SRC_VOCAB] \
                      [--tgt_vocab_path=PATH_TO_TGT_VOCAB] \
```


### Inference
After the model is trained, decode on binarized data using the following command. Note that the default for ```--ckpt_id``` is -1, which indicates the newest (not necessarily the best) checkpoint.
```
python3 src/arggen.py [--mode=decode] [--model={vanilla,seq_dec,shd_dec}] \
                      [--data_path=PATH_TO_BIN_DATA] \
                      [--model_path=PATH_TO_STORE_MODEL] \
                      [--exp_name=EXP_NAME] \
                      [--ckpt_id=CKPT_ID] \
                      [--beam_size=BS] \
                      [--src_vocab_path=PATH_TO_SRC_VOCAB] \
                      [--tgt_vocab_path=PATH_TO_TGT_VOCAB] \
```

### Evaluation

[coming soon]

### Support or Contact

Please contact Xinyu Hua (hua.x@husky.neu.edu) for any questions about this repository.

### Acknowledgement

Part of this codebase is based on [Pointer-generator](https://github.com/abisee/pointer-generator). The dual attention implementation is adapted from [Lisa Fan](https://github.com/lisafan).

