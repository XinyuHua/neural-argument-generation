### About

This repository holds the code and data for Neural Argument Generation project at Northeastern NLP. For details about the framework please read our ACL 2018 paper:

```
Neural Argument Generation Augmented with Externally Retrieved Evidence

Xinyu Hua and Lu Wang, Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL), 2018.

[link](http://xinyuhua.github.io/resources/acl2018/acl2018.pdf)
```

### Usage
#### Requirement

- python 3.5
- tensorflow  1.4.0
- numpy 1.14
- nltk 3.2.5

#### Data
Please download the dataset from [here](https://drive.google.com/file/d/1qyFP9zi9PMvQI7hERoP-X1YHly1dw9Fw/view?usp=sharing).

The dataset consists of the following 5 parts:

1. cmv\_processed: filtered OP posts and root replies used to create the core dataset

2. wikipedia\_retrieval: wikipedia article titles retrieved as evidence source for OP and root replies

3. reranked\_evidence: selected evidence sentences and extracted keyphrases for OP and root replies

4. trainable: directly trainable dataset

5. test: test set we used for evaluation

(Detailed readme file can be found [here](http://xinyuhua.github.io/resources/acl2018/README.txt).)


#### Preprocessing

(Note: we include processed trainable and test dataset, to train with processed dataset please skip this step.)
(coming soon)

#### Training
(coming soon)

#### Inference
(coming soon)

### Support or Contact

Please contact Xinyu Hua (hua.x@husky.neu.edu) for any problems with this repository.
