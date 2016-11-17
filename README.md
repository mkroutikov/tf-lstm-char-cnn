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

### Results

![training](https://cloud.githubusercontent.com/assets/14280777/20392288/24afe002-aca5-11e6-8729-edc3e4dccc55.png)

| Learning rate  |  Train/Valid/Test loss  |  Train/Valid/Test perplexity  |
|:--------------:|:-----------------------:|:------------------------------|
| 1.0            | 3.815 / 4.407 / 4.370   | 35.40 / 82.02 / **79.00**        |

Note that model **DOES** reproduce the published result.

### Training times

Timings were recorded on AWS EC2 instancies:

1. `c4.8xlarge` - 32 CPUs, no GPUs
2. `g2.2xlarge` - 8 CPU, 1 GPU (K520)
3. `p2.xlarge`  - 4 CPU, 1 GPU (K80)

|   Timing        | `c4.8xlarge` | `g2.2xlarge` | `p2.xlarge` |
|-----------------|--------------|--------------|-------------|
| Secs per batch  | 0.98         | 2.85         | 0.32        |
| Secs per epoch  | 1404         | 3779         | 428         |

Takes 3 hours to complete training (25 epochs) on `p2.xlarge` machine

## Generating random text
```
default wo n't <unk> down to keep a bit investment bankers trust in the <unk> of the company and as a <unk> a ford 's bill and long-term in the promoting a huge administrative 

in N treasury secretary robert stein as far below the like a major bankers fannie mae of publicly under the <unk> from a retired newly merged by a <unk> the mackenzie <unk> <unk> N <unk> in N people profit at the tremor and the cuban <unk> and the <unk> and a real estate rate increases from the technique when it was the <unk> which mr. gillett <unk> and it 's financial services <unk> about $ N <unk> most of the transportation and <unk> mcalpine energy 

the mainframe 

though none of <unk> the <unk> <unk> democratic leader adams those in the purchase as <unk> far more than he <unk> <unk> buy <unk> when the dingell the <unk> and mr. <unk> is organized improvements of the difference between N N of the american a <unk> and <unk> <unk> costs of the root to endure to judge and no solution to the same <unk> in one much greater than the company 's <unk> to be named to settle of the <unk> by discouraging oils a <unk> <unk> the chicago 's osaka 's fate in new york like such research firm with double-digit level 

the ad agency and <unk> in need to seismic of <unk> to hedge of with a one-hour sellers the <unk> tanks at least <unk> of the pediatric direct access to keep fasb policy claims against the wave of welcome who has an injury the <unk> s&ls will not as saying little more big board of the foreign currencies 

so far less king in the company 's <unk> while the book pieces
```

## Contributors

[Nicole](https://github.com/hejunqing)

[David](https://github.com/pythonner)
