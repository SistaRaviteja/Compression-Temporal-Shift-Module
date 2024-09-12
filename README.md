# Compression Temporal Shift Module (CTSM)

This repository contains the implementation of the Compression Temporal Shift Module (CTSM), a novel spatio-temporal architecture designed for compressing surgical videos and generating semantically rich latent representations while maintaining reasonable computational costs.

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Usage](#usage)
   - [Data Preparation](#data-preparation)
   - [Training the Model](#training-the-model)
   - [Inference](#inference)
4. [License](#license)

## Installation

To use the code in this repository, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/SistaRaviteja/Compression-Temporal-Shift-Module.git
   ```

2. **Create a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv ctsm-env
   source ctsm-env/bin/activate  # On Windows, use ctsm-env\Scripts\activate`
   ```

3. **Install Required Packages:**

   Install the required Python packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

   Ensure all dependencies are installed correctly, including PyTorch, torchvision, numpy, and other libraries specified in `requirements.txt`.

## Configuration

The `config.py` file contains various settings and parameters required for training and inference. Some key parameters include:

- `tr_batch_size`: Batch size for training.
- `infer_batch_size`: Batch size for inference
- `lr`: Learning rate for the optimizer.
- `epochs`: Number of training epochs.
- `pre_trained_model_path`: Path to save or load the model weights.
- `train`: Enable to train the model.
- `generate_results`: Enable to do inference.
- `depth` : Switch between models of 2, 3 and 4 depths.
- `sequence length`: Switch between 3 and 10 sequence lengths.
- `overlap`: Switch between overlap and non-overlap sequences.
- `additional_files`: path to the dataset and corresponding datapaths files.

Make sure to update the configuration file and datapath files according to the naming used in `config.py` as per your requirements.

## Usage

### Data Preparation

1. **Dataset Download:**

   - Prepare the dataset (e.g., Cholec80) and place it in the desired directory.
   - The dataset should be organized properly, and a path to the dataset should be provided in the configuration file.

### Training the Model

To train the CTSM model on the prepared dataset:

1. **Set Configuration:**

   - Update the `config.py` file to specify the necessary configurations, such as `train=True`, dataset path, model save path, batch size, number of epochs, and learning rate.
   - Change the depth value to shift between models of `depth = 2, 3 and 4`.
   - Change the sequence length value to shift between `sequence length = 3 and 10`.
   - Chnage the overlap value to shift between `overlap = seq and nvseq`, overlapping and non-overlapping sequence respectively.

2. **Run Training Script:**

   - Set `train = True` in `config.py`
   - Execute the training script to start training the CTSM model:

   ```bash
   python /path/to/training.py
   ```

   - The training results, including model checkpoints, will be saved to the directory specified in `config.py`.

### Inference

To perform inference:

1. **Load Pre-trained Model:**
   
   - Set `pre_trained_model_weigths = True`
   - Update the `pre_trained_model_path` in the `config.py` file to point to the pre-trained model.

2. **Run Inference Script:**

   - Set `generate_results = True` and `train = False`
   - Use the `training.py` script to run the inference on the test dataset:

   ```bash
   python /path/to/training.py
   ```

   - The python scripts for inference are in [infer_utils](https://github.com/SistaRaviteja/Compression-Temporal-Shift-Module/tree/main/src/infer_utils)
   - The results will be saved in the specified output directory.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.