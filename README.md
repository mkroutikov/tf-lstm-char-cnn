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
| 1.0            | 3.815 / 4.407 / 4.370   | 35.40 / 82.02 / 79.00         |

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
default targets for <unk> may be settled at least half of minnesota 's N <unk> group and the company 's <unk> director of remains n't winning more than the national convenience and its own N N N million 's immediate deficits and other companies at least to buy $ N N N hours of the company officials who did n't immediately funded by the company microsystems

<unk> N <unk> in N pages for all the seismic public revenue disappointments

it was a good news agency <unk> 's blessing equipment and discuss during the company and <unk> by N trillion <unk> and said john wilbur n. <unk> research services corp. N million of the massachusetts and <unk> downturn in N N million

but she added

the company

democrat bush rubicam around N N who was <unk> which is described by <unk> <unk> talks

richard <unk> smith arabia <unk> and a cloud a corporation money to keep their own N N years ago to be <unk> and <unk> <unk> without its new jersey and bebear 's big board and stevenson corp. a lawyer

by contrast for instance about N N N N N N N years ago

and <unk> & gamble stock 's <unk> <unk> a model to preclude a 190-point homes are resolved from any cyclical deals with depressed finance as a provision and the national air lines of bikers were <unk> to conform to provide more than prompted the <unk> midday by physical <unk> of the kaiser & hedges by a pile during a toledo <unk> texas air lines in the company recently on the vaccine is n't the singapore acceptance of effective at very heavy damage to <unk> $ N million libel information that high rates
```

## Contributors

[Nicole (hejunqing)](https://github.com/hejunqing)
[David Nadeau (pythonner)](https://github.com/pythonner)
