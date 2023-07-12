# Variable length motif discovery

## Introduction

This codebase contains a method for discovering variable length motifs in time series data. The method is explained in the paper "Variable length motif discovery in time series data".

## How to use

Test data can be found in the **Data/** folder. The following datasets are grouped in their respective folders:

- Synthetic data: A synthetic dataset where motifs are added to a noisy signal
- Numenta Taxi: The NY taxi dataset of the [Numenta Anomaly Benchmark](https://github.com/numenta/NAB). It shows the passenger count of a taxi service over time. A weekly pattern is visible.
- Skyline: A real world KPI dataset provided by Skyline Communications. The dataset includes important metrics from Telecom providers where motifs are present.

Code can be found in the **Code/** folder. The code to test the proposed method and VALMOD on the provided datasets is in the jupyter notebook files. The proposed method implementation can be found in the **Helper/** subdirectory, while the [VALMOD implementation](https://www.sciencedirect.com/science/article/abs/pii/S0952197620300087) can be found in the **distancematrix/** subdirectory.

The jupyter notebooks will output their results in the **Results/**  directory as pdf files. The **result.pdf** file shows an overview of the top motif of each processed file. Additionally, the found motifs of each processed file are saved under **output\_<filename>.pdf**.