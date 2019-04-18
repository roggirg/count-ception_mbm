# count-ception_mbm

Pytorch implementation of [count-ception](https://arxiv.org/abs/1703.08710) on the MBM dataset.
Please refer to the [original repository] (https://github.com/ieee8023/countception) (with Theano and Lasagna deep learning frameworks) for more details on other datasets.

## Requirements

- Pytorch
- Scikit-Image

## Preparing dataset

I included a Pickle file of the dataset similar to how it was prepared in the [original repository's MBM code](https://github.com/ieee8023/countception/blob/master/count-ception-mbm.ipynb).
To re-generate the pickle file, you would need to run 'create_datafiles.py' making sure to specify the [dataset directory](https://github.com/roggirg/count-ception_mbm/blob/64d552255f8042dd0efe9a6d9380a10d8713f5ca/utils/create_datafiles.py#L71).

## Training

To train a model, run the following command: 

`python train.py --pkl-file 'utils/MBM-dataset.pkl' --batch-size 2 --epochs 1000 --lr 0.001`

To test the model, run the following command:

`python test.py --pkl-file 'utils/MBM-dataset.pkl' --batch-size 1 --ckpt 'checkpoints/after_950_epochs.model'`



## Citation:

Count-ception: Counting by Fully Convolutional Redundant Counting<br>
JP Cohen, G Boucher, CA Glastonbury, HZ Lo, Y Bengio<br>
International Conference on Computer Vision (ICCV) Workshop on Bioimage Computing

```
@inproceedings{Cohen2017,
title = {Count-ception: Counting by Fully Convolutional Redundant Counting},
author = {Cohen, Joseph Paul and Boucher, Genevieve and Glastonbury, Craig A. and Lo, Henry Z. and Bengio, Yoshua},
booktitle = {International Conference on Computer Vision Workshop on BioImage Computing},
url = {http://arxiv.org/abs/1703.08710},
year = {2017}
}
