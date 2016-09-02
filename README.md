# tf-lstm-char-cnn

Tensorflow port of Yoon Kim's [Torch7 code](https://github.com/yoonkim/lstm-char-cnn). See also similar project [here](https://github.com/carpedm20/lstm-char-cnn-tensorflow) that failed to reproduce Kim's results and was apparently abandoned by the author. Many pieces of code are borrowed from it.

## Installation
You need tensorflow version 0.10 and python (obviously).

## Running

```sh
python train.py -h
python evaluate.py -h
```

## Training

```sh
python train.py
```
will train large model from Yoon Kim's paper.

## Evaluate

```sh
python evaluate.py --load_model cv/epoch024_4.4962.model
```
evaluates this model on the test dataset

## Generate

```sh
python generate.py --load_model cv/epoch024_4.4962.model
```
generates random text from the loaded model

## Model

![model](https://cloud.githubusercontent.com/assets/14280777/17991383/13990c56-6b0c-11e6-8a9f-f4de07a6984f.png)

`Model_1` graph is used for inference (computing validation loss and perplexity during training)

`Model` graph is used for training.


## Results
Does **NOT** reproduce the original result. Training large character model yields test perplexity 86. Yoon Kim got to 79. Reason unknown.

```
 33143: 24 [ 1295/ 1327], train_loss/perplexity = 3.91299558/50.0486526 secs/batch = 2.8760s, grad.norm=16.99129105
 33148: 24 [ 1300/ 1327], train_loss/perplexity = 4.05615807/57.7520065 secs/batch = 3.7823s, grad.norm=15.83820534
 33153: 24 [ 1305/ 1327], train_loss/perplexity = 4.15003490/63.4362144 secs/batch = 3.0995s, grad.norm=16.83547401
 33158: 24 [ 1310/ 1327], train_loss/perplexity = 4.36305475/78.4965591 secs/batch = 2.9983s, grad.norm=16.57718658
 33163: 24 [ 1315/ 1327], train_loss/perplexity = 4.19097710/66.0873337 secs/batch = 2.8248s, grad.norm=17.13934517
 33168: 24 [ 1320/ 1327], train_loss/perplexity = 4.13271141/62.3467407 secs/batch = 3.3823s, grad.norm=16.16269875
 33173: 24 [ 1325/ 1327], train_loss/perplexity = 4.11810350/61.4426079 secs/batch = 3.2843s, grad.norm=16.78089714
        > validation loss = 4.67386150, perplexity = 107.11054993
        > validation loss = 4.61193180, perplexity = 100.67845154
        > validation loss = 4.57779837, perplexity = 97.29994202
        > validation loss = 4.56800795, perplexity = 96.35198212
        > validation loss = 4.72813463, perplexity = 113.08441925
        > validation loss = 4.67605066, perplexity = 107.34529114
        > validation loss = 4.59290695, perplexity = 98.78116608
        > validation loss = 4.43686056, perplexity = 84.50921631
        > validation loss = 4.26628971, perplexity = 71.25675964
        > validation loss = 4.36400318, perplexity = 78.57103729
        > validation loss = 4.57795095, perplexity = 97.31478882
        > validation loss = 4.53907728, perplexity = 93.60439301
        > validation loss = 4.45130253, perplexity = 85.73854828
        > validation loss = 4.29314089, perplexity = 73.19600677
        > validation loss = 4.21168852, perplexity = 67.47036743
        > validation loss = 4.27082157, perplexity = 71.58042145
        > validation loss = 4.68130589, perplexity = 107.91089630
        > validation loss = 4.21844149, perplexity = 67.92753601
        > validation loss = 4.75229788, perplexity = 115.85018921
        > validation loss = 4.56119919, perplexity = 95.69817352
        > validation loss = 4.32170248, perplexity = 75.31674194
at the end of epoch: 24
train loss = 4.03567644, perplexity = 56.58118088
validation loss = 4.49624329, perplexity = 89.67959731
Saved model cv/epoch024_4.4962.model
validation perplexity did not improve enough, decay learning rate
learning rate was: 0.00390625
new learning rate is: 0.001953125
```

![image](https://cloud.githubusercontent.com/assets/14280777/18205786/ec622ffa-70f1-11e6-810d-013bc39ff72f.png)
