# The BGANet for sentiment classification
## How to establish the environment
First, you should create the virtual environment in `Python`  as follows:
```bash
python -m venv bgaenv
source bgaenv/bin/activate
```
And then install the required package in `bgaenv` virtual environment with pip command:
```bash
pip install -r requirements.txt
```
## How to run and save the model result
1. BGANet
The model should configure the some parameters, you should run the following command:
+ BGANet(LSTM):
```bash
python train.py --gate-flag --rnn-type lstm --model BGANet --epoch-times 50
```
+ BGANet(GRU):
```bash
python train.py --gate-flag --rnn-type gru --model BGANet --epoch-times 50
```
2. BGANetNoneGate
+ BGANetNoneGate(LSTM):
```bash
python train.py --rnn-type lstm --model BGANet --epoch-times 50
```
+ BGANetNoneGate(GRU):
```bash
python train.py --rnn-type gru --model BGANet --epoch-times 50
```
And you can also choose the multiple attention version with `--multiple` flag in BGANet.
3. TCHNN
+ TCHNN(LSTM):
```bash
python train.py --rnn-type lstm --model TCHNN --epoch-times 50
```
+ TCHNN(GRU):
```bash
python train.py --rnn-type gru --model TCHNN --epoch-times 50 
```
4. TCHNNGate
+ TCHNNGate(LSTM):
```bash
python train.py --rnn-type lstm --model TCHNN --epoch-times 50 --gate-flag
```
+ TCHNNGate(GRU):
```bash
python train.py --rnn-type gru --model TCHNN --epoch-times 50 --gate-flag
```
And you can also choose the multiple attention version with `--bi-multiple` or `--cap-multiple` flag in BGANet.
5. BertCNN
```bash
python train.py --model BertCNN --epoch-times 50
```
6. BertRNN
+ BertRNN(LSTM)
```bash
python train.py --rnn-type lstm --model BertRNN --epoch-times 50
```
+ BertRNN(GRU)
```bash
python train.py --rnn-type lstm --model BertRNN --epoch-times 50
```
And you can also choose the multiple attention version with `--multiple` in BertRNN.
7. Bert
+ Bert
```bash
python train.py --model Bert --epoch-times 50
```
And you can also choose the multiple attention version with `--multiple` in Bert.

**notes:** You should add `--dataset` in your command ,which denotes the specific dataset(`student`,`hotel` and `restaurant`) for training the model, the default dataset is `student`.
## The datasets for training the model
+ Student sentiment dataset
+ [Hotel sentiment dataset](https://https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/ChnSentiCorp_htl_all/intro.ipynb)
+ [Restaurant sentiment dataset](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/waimai_10k/intro.ipynb)
