# Transformer Inter-Turn Fault Detection using Artificial Intelligence

This repository contains a machine learning-based solution for detecting both the severity and location of inter-turn faults in transformers using Artificial Intelligence. The project involves simulating transformer faults, capturing and processing current waveforms using Continuous Wavelet Transformers (CWT), and feeding them into a Convolutional Neural Network (CNN) for analysis. The detailed findings of this project can be found [here](link).

## Repository Structure

The project is organized into two main folders:

### 1. Fault Generation

This folder contains MATLAB Live Script files and functions to simulate faults on each phase of the transformer, process current waveforms, and save them as scalogram images. The primary files and their purposes are as follows:

- **Phase A Code.mlx**: Simulates faults in Phase A of the transformer.
- **Phase B Code.mlx**: Simulates faults in Phase B of the transformer.
- **Phase C Code.mlx**: Simulates faults in Phase C of the transformer.
- **processdata.mlx**: Processes waveform data and saves it as scalogram images.
- **faultgenmain.mlx**: Main script to run the entire project.
- **newfinal.slx**: Simulink 630 kVA transformer model

Additionally, the `functions` folder within this directory contains essential functions for saving scalogram images, saving data as CSV files, plotting scalogram images, and creating necessary folders for the project's operation.

### 2. ML

This folder contains Python scripts for dataset preparation and prediction using a CNN model. Here are the key files and their roles:

- **dataset_preparation.py**: Extracts saved scalogram images from disk, attaches labels, and binarizes them into a .npz file for efficient storage.
- **predictor.py**: Contains code to extract images from the .npz file and feed them into a CNN for training. The model can be operated in both training and testing modes, allowing you to evaluate its performance on generated faults.

## Getting Started

To get started with this project, follow these steps:

1. Clone this repository to your local machine:

   ```
   git clone <repository-url>
   ```

2. Set up your environment:

   - Ensure you have MATLAB installed for running the fault generation scripts.
   - Install the required Python libraries by running:

     ```
     pip install -r requirements.txt
     ```

3. Edit the code to change the directory locations for saving and retrieving scalogram images.

4. Generate and process fault data using MATLAB scripts in the "Fault Generation" folder.

5. Prepare the dataset using `dataset_preparation.py`. This script will create a .npz file containing your dataset.

6. Train and test the CNN model by using `predictor.py`. Adjust the model parameters as needed for your specific use case.

7. Analyze the results and use the model for transformer inter-turn fault detection.

## Conclusion

This project provides a comprehensive solution for transformer inter-turn fault detection using deep learning. The combination of fault generation, data preprocessing, and CNN-based prediction enables accurate detection of fault severity and location.
