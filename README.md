# fbpicker
## Multi-pattern deep neural network algorithm for first-break picking employing an open-source library

Here we propose a deep-learning solution to address the issue of multi-level first-break (FB) analysis using a time-sequence pattern-recognition process implemented in the newest open-source machine learning library (Keras). We use well-established methods such as STA/LTA, entropy or fractal dimension to provide patterns required by generative model training with deep neural networks (DNN). FB picking is cast as the binary classification that requires a model to differentiate FB sample from all other samples in a seismic trace. Our algorithm (provided freely as a Jupyter Notebook) is robust and flexible in a way of adding new pattern generators that might contribute to even better performance, while already trained models can be saved and re-used for another dataset collected with similar acquisition parameters (e.g., in multi-line surveys). Application to real seismic data showed that the model trained on only 1000 manually-picked FB onsets was able to predict FB on 470 thousands of traces with the success rate of nearly 90% as verified by the manually-derived picks, which were not used in the training process. 

This repository contains IPython Notebook with sample code and arbitrary seismic traces, complementing research artictle about automatic first-break picking.

You can view **"training.ipynb"** and **"prediction.ipynb"** directly on github, or clone the repository, install dependencies listed in the notebook and play with code locally.

# Training
![Training](./images/training.png "Training")

# Prediction
![Prediction](./images/prediction.png "Prediction")
