# tf-lstm-char-cnn

Tensorflow port of Yoon Kim's [Torch7 code](https://github.com/yoonkim/lstm-char-cnn). See also similar project [here](https://github.com/carpedm20/lstm-char-cnn-tensorflow) that failed to reproduce Kim's results and was apparently abandoned by the author. Many pieces of code are borrowed from it.

## Installation
You need tensorflow version 1.0 and python (obviously).

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

## Pre-trained model

Here is a ZIP file containing model files (created with TF 1.0). This model was trained with the default parameters
and acheved the accuracy of the published result.

[Pre-trained model](https://drive.google.com/open?id=0B27Dn0k-PX-YQ0FwTm5qS2FQMXM) 60Mb

### Results

![training](https://cloud.githubusercontent.com/assets/14280777/20392288/24afe002-aca5-11e6-8729-edc3e4dccc55.png)

| Learning rate  |  Train/Valid/Test loss  |  Train/Valid/Test perplexity  |
|:--------------:|:-----------------------:|:------------------------------|
| 1.0            | 3.815 / 4.407 / 4.369   | 35.40 / 82.02 / **79.03**        |

Note that model **DOES** reproduce the published result.

### Training times (legacy, with TF 0.12)

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
default charges during the straight summer of the collapse of it works that it <unk> to the u.s. is very <unk> 

mr. boyd 's remarks are crucial to the champion to a close <unk> in N he says 

explains it 's only one of the many really identified 

it did n't end a deal we do n't know what 's no longer viable 

<unk> <unk> <unk> 's <unk> hearing his head 

nora more than a$ N million has had no control in N it would do that could win any new other <unk> corporate and real-estate <unk> a future number of names <unk> at telling an <unk> outside the international institution 

the extensive approach is a real form of unpublished football <unk> as the <unk> bells on negotiation refund 

in other words the others who paid $ N a figure 

what 's really the <unk> who can use a vast hundred the transaction would give it a better use of the <unk> and 

the demise of the topic is fatal than we can do anything that is like the change in reality says justice <unk> <unk> of <unk> & <unk> a direct mail marketing firm 

when when lawyer <unk> <unk> <unk> as corporate crime remains a distant younger woman says i 've seen however an injection in the new york money fund 

some hearings have detectors with <unk> have survived mr. achenbaum 's veto incorporated the <unk> delicate center which <unk> a heavy medium that lawyers call are confusing big inquiries for the forecasting two outlets to had the claims 

the afghan people have a series that 's really heard to share legislation for <unk> without all their racial cosmetic ties 

he is going to hold his ...
```

## Contributors

[Nicole](https://github.com/hejunqing)

[David](https://github.com/pythonner)

[derluke](https://github.com/derluke)
