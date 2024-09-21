# Apple Stock Price Prediction Using Ensemble RNN Models

## Overview

This project aims to explore the development of an **Ensemble Recurrent Neural Network (RNN)** model designed for predicting **Apple's stock price**. The report investigates the effectiveness of using an ensemble approach that integrates various RNN variants, including **Gated Recurrent Units (GRUs)** and **Long Short-Term Memory (LSTM)** networks. These models were employed to capture short, medium, and long-term dependencies within stock market data, with the goal of improving predictive accuracy.

RNNs, GRUs, and LSTMs are well-suited for sequential data processing, making them ideal for time-series analysis like stock market forecasting. However, each model has its strengths and weaknesses:
- **RNNs** utilise hidden states that carry information through time, allowing them to process sequences. However, they face challenges with long-term dependencies due to vanishing and exploding gradients.
- **GRUs** introduce **gates** to control how much of the previous hidden state is passed forward. The update and reset gates allow GRUs to manage medium-term dependencies more effectively than standard RNNs.
- **LSTMs** are more complex and introduce **cell states** alongside hidden states. The forget, input, and output gates allow LSTMs to capture long-term dependencies while mitigating gradient issues.

## Ensemble Model
To address the limitations of individual RNN models, this project introduces an **ensemble method** that integrates predictions from multiple RNN variants using a **Single-Layer Perceptron (SLP)**. The ensemble leverages **Stacked Generalization**, a technique designed to optimally combine outputs from different models.

In this approach, **15 RNN models** are created by applying **RNN**, **GRU**, and **LSTM** variants across **five different sequence lengths** (5, 10, 20, 40, and 80 days). The **SLP** then learns to assign optimal weights to the predictions from each model, producing a final combined prediction. By incorporating a diverse set of sequence lengths and model types, this ensemble captures a wide range of temporal patterns, creating a more comprehensive model for predicting stock prices.

## Data
The dataset used for this project was sourced from the **Yahoo Finance API**, containing stock data for Apple from **2000-01-01 to 2023-01-01**. The data includes stock features such as:
- Open
- High
- Low
- Close
- Adjusted Close
- Volume

Feature Engineering was also conducted, introducing **30-day and 90-day moving averages**, **standard deviation**, and **Bollinger Bands**, bringing the total number of features to 14.

Additional preprocessing steps include:
- **Min-Max Scaling** was applied to normalise the data between -1 and 1.
- Data was split into training, validation, and testing sets, ensuring that the time-series nature was preserved.
    - Training data: before 2018
    - Validation data: 2018–2021
    - Testing data: 2022–2023

## Technologies Used
- Python
- Pytorch
- Numpy
- Pandas
- Matplotlib
- yfinance
- Scikit-learn

## Full Report
For a more in-depth explanation of the methodology, results, and conclusions, please refer to the full research report, available in the repository as Research_Report.pdf.
