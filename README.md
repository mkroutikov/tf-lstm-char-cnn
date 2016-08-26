# tf-lstm-char-cnn

Tensorflow port of Yoon Kim's [Torch7 code](https://github.com/yoonkim/lstm-char-cnn). See also similar project [here](https://github.com/carpedm20/lstm-char-cnn-tensorflow) that failed to reproduce Kim's results and was apparently abandoned by the author. Many pieces of code are borrowed from it.

## Installation
You need tensorflow version 0.10 and python (obviously).

## Training

```
python train.py
```
will train large model from Yoon Kim's paper.

## Evaluate

```
python eval.py --model model --dataset dataset
```
evaluates this model on the given dataset

## Model

![model](https://cloud.githubusercontent.com/assets/14280777/17991383/13990c56-6b0c-11e6-8a9f-f4de07a6984f.png)

`Model_1` graph is used for inference (computing validation loss and perplexity during training)

`Model` graph is used for training.
