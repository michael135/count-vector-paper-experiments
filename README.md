# Process Monitoring on Sequences of System Call Count Vectors

This repository contains the code and part of the data used in the paper *[Process Monitoring on Sequences of System Call Count Vectors](https://arxiv.org/abs/1707.03821)* by Dymshits et. al.

## Install requirements

`pip install -r requirements.txt`

To install tensorflow with GPU support, replace `tensorflow` by `tensorflow-gpu` in `requirements.txt`.

## Unzip the data

`tar -xf data/laboratory_data.bzip2 -C data/`

Beware that once unzipped, the file weights approximately 1.8gb.

## Reproduce laboratory setup results

`python train.py data/laboratory_data.pkl`
