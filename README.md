# Multivariate Time Series Forecasting with Graph Neural Networks
Natalie Koh, Zachary Laswick, Daiwei Shen

## Datasets
- [MotionSense](https://github.com/mmalekzadeh/motion-sense)
- [MHealth](https://archive.ics.uci.edu/ml/datasets/mhealth+dataset)

## Architectures Used
- [STEP](https://arxiv.org/abs/2206.09113)
- [Graph WaveNet](https://arxiv.org/pdf/1906.00121)
- Simple graph convolutional network with LSTM layer implemented in Keras

## Scripts
- For data pre-processing, see `PruneDatasets_SingleSubject.ipynb`.
- To run STEP on the datasets, use scripts in STEP/ModifiedSTEPCode.
- To run Graph WaveNET, `cd` into the WaveNet directory and run `python trainpy --gcn_bool`.
- To run the simple GCN implemented in Keras, use `KerasGNNwLSTM.ipynb`.

