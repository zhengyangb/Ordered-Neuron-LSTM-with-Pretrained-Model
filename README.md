# <small> DS-GA 1012 Course Project </small>

# Ordered Neuron LSTM with Pretrained Model

[![GitHub contributors](https://img.shields.io/github/contributors/zhengyangb/NLU2019.svg)](https://github.com/zhengyangb/NLU2019/graphs/contributors/)

Syntactic parsing, the task of learning grammar structure, acts as the backbone of naturallanguage understanding and has been shown to benefit many downstream natural language processing tasks.  Toward  this  pursuit,  we  introduce two different word representations to ON-LSTM model and evaluate their grammar induction performances.

## Requirements

To run the code, you will need an environment with the following:

- Python (>3.6)
- PyTorch
- CUDA (strongly recommended)
- NLTK

To run the evaluation script, you will also need Penn Treebank database. 

## Quick Start

1. Please make sure you have the install NLTK PTB package and have the PTB corpus in the directory. For more details, please refer to [ON-LSTM repo](https://github.com/yikangshen/Ordered-Neurons) and find instructions. 

2. To run the language modeling training, use the following command

   ```
   python main_gpt.py --cuda --mode GPT --learning_rate 1e-6 --lr 10 --batch_size 20 --dropoute 0.0 --dropout 0.45 --dropouth 0.3 --dropouti 0.0 --wdrop 0.45 --chunk_size 10 --seed 141 --epoch 1000
   ```

3. To test the model on the unsupervised parsing task, please use 

   ```
   python test_phrase_grammar.py --cuda
   ```

## References

- Shen, Yikang, et al. "Ordered Neurons: Integrating Tree Structures into Recurrent Neural Networks." *Proceedings of ICLR* (2019).
- Code for the paper "Ordered Neurons: Integrating Tree Structures into Recurrent Neural Networks" https://github.com/yikangshen/Ordered-Neurons
- The Big-&-Extending-Repository-of-Transformers: Pretrained PyTorch models for Google's BERT, OpenAI GPT & GPT-2, Google/CMU Transformer-XL. https://github.com/huggingface/pytorch-pretrained-BERT









<!--
run with `--cuda --mode GPT --learning_rate 1e-6 --lr 10 --batch_size 20 --dropoute 0.0 --dropout 0.45 --dropouth 0.3 --dropouti 0.0 --wdrop 0.45 --chunk_size 10 --seed 141 --epoch 1000`  

- Distinguish GPT optimizer 
- Return GPT LM result
- Weighted loss
- Hyper-parameter
  - Turn off embedding drop out
  - decrease learning rate  

April 8  
tools/id2gptid  
util.get_batch_gpt  
GPT_model  
main_gpt  

TODO:  
Resume setting
Check accuracy   
Inprove efficiency  
Experiments design  
-->

<!--
# ON-LSTM

This repository contains the code used for word-level language model and unsupervised parsing experiments in 
[Ordered Neurons: Integrating Tree Structures into Recurrent Neural Networks](https://arxiv.org/abs/1810.09536) paper, 
originally forked from the 
[LSTM and QRNN Language Model Toolkit for PyTorch](https://github.com/salesforce/awd-lstm-lm).
If you use this code or our results in your research, we'd appreciate if you cite our paper as following:

```
@article{shen2018ordered,
  title={Ordered Neurons: Integrating Tree Structures into Recurrent Neural Networks},
  author={Shen, Yikang and Tan, Shawn and Sordoni, Alessandro and Courville, Aaron},
  journal={arXiv preprint arXiv:1810.09536},
  year={2018}
}
```

## Software Requirements
Python 3.6, NLTK and PyTorch 0.4 are required for the current codebase.

## Steps

1. Install PyTorch 0.4 and NLTK


2. Download PTB data. Note that the two tasks, i.e., language modeling and unsupervised parsing share the same model 
strucutre but require different formats of the PTB data. For language modeling we need the standard 10,000 word 
[Penn Treebank corpus](https://github.com/pytorch/examples/tree/75e435f98ab7aaa7f82632d4e633e8e03070e8ac/word_language_model/data/penn) data, 
and for parsing we need [Penn Treebank Parsed](https://catalog.ldc.upenn.edu/LDC99T42) data.

3. Scripts and commands

  	+  Train Language Modeling
   	 	```python -u main.py --cuda --batch_size 20 --dropout 0.45 --dropouth 0.3 --dropouti 0.5 --wdrop 0.45 --chunk_size 10 --seed 141 --epoch 1000 --data data/penn```

  	    + Remove the cuda flag if to run without cuda

  	+ Test Unsupervised Parsing

    ```python test_phrase_grammar.py --cuda```
    
    The default setting in `main.py` achieves a perplexity of approximately `56.17` on PTB test set 
    and unlabeled F1 of approximately `47.7` on WSJ test set.

-->