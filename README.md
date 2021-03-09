# AutoEncoder for Deep learning study 2020

MNISTデータを利用したシンプルなAutoEncoderの学習と外れ値検出，ノイズ除去

## Before running script

    mkdir models
    mkdir models/autoencoder
    mkdir models/outlier_detection
    mkdir models/denoising
    mkdir figures
    mkdir figures/autoencoder
    mkdir figures/outlier_detection
    mkdir figures/denoising
    
    
## Run Autoencoder
training:

    python autoencoder.py -hidden 20 -epochs 50 -stack
    python autoencoder.py -hidden 2 -epochs 50 -stack
    # model is saved at models/autoencoder
    # reconstruct images are saved at figures/autoencoder

reconstruct from trained models:

    python autoencoder.py -hidden 20 -model [MODEL_PATH] -stack
    python autoencoder.py -hidden 2 -model [MODEL_PATH] -stack
    

## Run Outlier Detection
training:

    python outlier_detection.py -hidden 20 -epochs 50
    python outlier_detection.py -hidden 2 -epochs 50
    # model is saved at models/outlier_detection
    # reconstruct images are saved at figures/outlier_detection

## Run Denoising
training:

    python denoising.py -hidden 20 -epochs 50
    python denoising.py -hidden 2 -epochs 50
    # model is saved at models/denoising
    # reconstruct images are saved at figures/denoising
