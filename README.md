# Apple Stock Price Prediction Using Ensemble RNN Models

## Overview

This project focuses on developing an **Ensemble Recurrent Neural Network (RNN)** model to predict **Apple's stock price**. It integrates various RNN variants, including **Gated Recurrent Units (GRUs)** and **Long Short-Term Memory (LSTM)** networks, aimed at capturing short, medium, and long-term patterns in stock market data to improve predictive accuracy.

RNNs, GRUs, and LSTMs are well-suited for time-series analysis:
- **RNNs** capture sequential data but struggle with long-term dependencies due to vanishing gradients.
- **GRUs** improve on this by using **gates** to manage medium-term dependencies more effectively.
- **LSTMs** employ **cell states** and gates to capture long-term dependencies, overcoming gradient issues.

### Ensemble Model

The project introduces an **ensemble method** that combines predictions from 15 RNN models (3 RNN variants applied to 5 different sequence lengths: 5, 10, 20, 40, and 80 days). A **Single-Layer Perceptron (SLP)** is used to optimally weigh the predictions, resulting in a final stock price prediction. This ensemble approach leverages diverse sequence lengths and model architectures to create a more robust model.

## Key Methods Utilised

- **Recurrent Neural Network (RNN)**: Captures patterns over time by maintaining hidden states. Implemented using PyTorch's `RNN()` function with Mean Squared Error (MSE) as the loss function and Stochastic Gradient Descent (SGD) as the optimiser.
  
- **Gated Recurrent Unit (GRU)**: Regulates past information through reset and update gates. Implemented with PyTorch's `GRU()` function, the GRU models are trained similarly to RNNs and designed to handle varying sequence lengths.
  
- **Long Short-Term Memory (LSTM)**: Uses forget, input, and output gates to retain long-term dependencies. Implemented using PyTorch's `LSTM()` function, with MSE loss and SGD optimisation. LSTMs handle longer time sequences in stock prediction.

- **Single Layer Perceptron (SLP)**: A simple neural network combining predictions from RNN, GRU, and LSTM models. It takes 15 input features (3 variants × 5 sequence lengths) and produces a single output, trained with MSE and SGD in PyTorch.

The models are evaluated using **MSE**, **Mean Absolute Error (MAE)**, and **Root Mean Squared Error (RMSE)** to assess stock price prediction accuracy.

## Data

The data was sourced from **Yahoo Finance**, containing Apple stock data from **2000-01-01 to 2023-01-01**. Key stock features include:
- Open
- High
- Low
- Close
- Adjusted Close
- Volume

**Feature engineering** was conducted to introduce 30-day and 90-day moving averages, standard deviation, and Bollinger Bands, bringing the total features to 14. The data was **min-max scaled** to ensure all features contributed equally during model training.

The dataset was split as follows:
- **Training data**: Pre-2018
- **Validation data**: 2018–2021
- **Testing data**: 2022–2023

## Results

Surprisingly, RNN models outperformed GRU and LSTM models across most configurations. The **RNN with a 40-day sequence** length achieved the best performance with a validation loss of 0.00096. Conversely, **LSTMs**, expected to perform well in long-term dependencies, struggled and had the highest loss across all configurations. 

The **SLP ensemble method** performed well, with a validation loss of 0.00235, MAE of 0.03809, and RMSE of 0.04850, ranking 7th overall. The RNN 40, GRU 20, and LSTM 20 models contributed most to the ensemble, though the inclusion of LSTM was unexpected given its weaker performance. While promising, the ensemble method did not outperform the best individual RNN and GRU models.

For detailed insights and analysis, refer to the complete research report available as **`Research_Report.pdf`**.

## Technologies Used

- Python
- PyTorch
- Numpy
- Pandas
- Matplotlib
- yfinance
- Scikit-learn
