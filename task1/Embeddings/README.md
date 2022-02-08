# Pretraining models for sentiment project
## Establish the environment 
Fisrt, you should build the `Python` virtual environment.
```bash
python -m venv pretrainenv
source bgaenv/bin/activate
```
Then, you can train your model,including `Skip-Gram`,`CBOW`,`Glove` and `FastText`model.
## Train the model
1. Word2Vec
+ SkipGram
```bash
python pretrain.py --model SkipGram 
```
+ CBOW
```bash
python pretrain.py --model CBOW 
```
2. Glove
```bash
python pretrain.py --model Glove 
```
The detail configuration is listed in `config.py`.