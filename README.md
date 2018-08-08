## margin-centre-face
**Improved face recognition for video.**

Our baseline model  is [InsightFace](https://github.com/deepinsight/insightface)  which introducted Angluar-Margin loss and performed well on many face recognition benchmarks. The code is based on  [InsightFace_TF](https://github.com/auroua/InsightFace_TF)  and we test its Model D.

## Angular Centre Loss 

We add a centre term to the loss function to make the embeddings closer with-in class.

The original Angular Margin is 

![](https://github.com/cnzeki/margin-centre-face/blob/master/image/am-loss.jpg)

This loss only forces the margin between intra and inter class to be large. As a result, embedings have a large intra-class variance. To learn a model with small intra-class variance, we checked the loss function and add a **centre** term:

![](https://github.com/cnzeki/margin-centre-face/blob/master/image/centre-loss.jpg)

In a hyper-sphere, an embeding vector is a point on the surface. It is easy to carry out that the **W_yi** should be center of all points in class **yi**. So **cosθ_yi** just represents how close is the embeding point to the center. To make is a loss funtion to minimize,  the average **cosθ_yi**  is subtracted by **1** Since there is a scalar **s** in **L_am** , we apply it to the centre loss too.

Then the final loss is a weighted sum of these two plus the weight decay term :

![](https://github.com/cnzeki/margin-centre-face/blob/master/image/total-loss.jpg)

In this experiment we use ![](https://github.com/cnzeki/margin-centre-face/blob/master/image/loss-param.jpg)

## Steps

### Training data

Just go [InsightFace Dataset-Zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo) and get the datasets you want to use. Here we used MS1M and VGGFace2.

### Test data

Download [YTF](http://www.cs.tau.ac.il/~wolf/ytfaces/)  do alignment and crop to 112x112,  use the alignment codes from [InsightFace](https://github.com/deepinsight/insightface) 

Check [here](https://github.com/cnzeki/face-datasets/tree/master/LFW) to get LFW test dataset.

### Training

```
cp config.ini.example config.ini
```

Edit `config.ini` to make dataset paths right.

#### Train Margin centre from scratch

In the `dataset` module , we provide codes to combine multiple datasets.  By default VGG and MS1M were used for training.

A pretrained model is here [vgg-ms1m/iter_258000](https://pan.baidu.com/s/1hPdVrwylXI0FZSDRCxGjyw) 

```
python -m train_centre.py
```

#### Fine tune with Triplet loss

Noticed that the YTF dataset has large intra-class variance , to train a model prefer large intra-class variance we must choose a proper training dataset. VGGFace2 is better than MS1M for this task.

```
python -m train_triplet.py --model_path=<pretrained_model_ckpt>
```

## Evaluation

### LFW

```
python -m eval/test_lfw --data=PATH/TO/lfw.np --model_path=/YOUR/MODEL/PATH
```

### YTF

Notice: We find that even the  corrected version of [YTF split pair file](http://www.cs.tau.ac.il/~wolf/ytfaces/splits_corrected.txt) contains some errors, a list of wrong video names is here [ytf-error.txt](https://github.com/cnzeki/margin-centre-face/blob/master/dataset/ytf-error.txt)

```
python -m eval/test_ytf --model_path=/YOUR/MODEL/PATH
```

### Results

|                  model                   | image size |         LFW          |         YTF          | YTF-corrected        |
| :--------------------------------------: | :--------: | :------------------: | :------------------: | -------------------- |
|         vgg-triplet/iter_426000          |   96x112   | **0.99467+-0.00386** | **0.96040+-0.00946** | 0.97202+-0.00819     |
| [vgg-ms1m/iter_258000](https://pan.baidu.com/s/1hPdVrwylXI0FZSDRCxGjyw) |   96x112   |   0.99300+-0.00323   |   0.95960+-0.00958   | **0.97531+-0.00537** |
| [InsightFace_TF/D](https://github.com/auroua/InsightFace_TF#model-d) |  112X112   |   0.99350+-0.00369   |   0.94920+-0.01078   | 0.96296+-0.00807     |


