# Process Monitoring on Sequences of System Call Count Vectors

This repository contains the code and part of the data used in the paper *Process Monitoring on Sequences of System Call Count Vectors* by Dymshits et. al.

## Install requirements

To install tensorflow with GPU support, replace `tensorflow` by `tensorflow-gpu` in `requirements.txt`.

`pip install -r requirements.txt`

## Unzip the data

`tar -xf data/laboratory_data.bzip2 data/`

Beware that once unzipped, the file weights approximately 1.8gb.

## Reproduce Laboratory Setup Results

`python train.py data/laboratory_data.pkl`
