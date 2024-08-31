# LiquidLM: Time-Aware Reservoir Computing with Self-Attention for Language Modeling

By David Hamner

## Overview

LiquidLM is an experimental language model that combines reservoir computing with time-aware dynamics and self-attention mechanisms for code generation. This project explores a novel architecture that integrates a Time-Aware Reservoir Layer with self-attention layers to create a unique approach to language modeling.

## Features

- Time-Aware Reservoir Layer for dynamic computations
- Self-attention layers for capturing long-range dependencies
- Positional encoding for sequence awareness
- Gradient accumulation and learning rate scheduling
- Early stopping mechanism for efficient training
- Sample generation to showcase model capabilities

## Requirements

- Python 3.7+
- PyTorch
- Transformers library
- Datasets library
- NumPy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/ruapotato/LiquidLM
   cd liquidlm
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To train the model:
```
python train.py
```

To generate text using a trained model:
```
python generate.py --model_path path/to/best_model.pth --prompt "Your prompt here"
```

## Model Architecture

LiquidLM consists of the following key components:

1. Embedding Layer
2. Time-Aware Reservoir Layer
3. Positional Encoding
4. Multiple Self-Attention Layers
5. Layer Normalization
6. Dropout for regularization
7. Linear output layer

The Time-Aware Reservoir Layer introduces time-dependent dynamics to the model, while the self-attention layers allow it to capture long-range dependencies in the input data. The positional encoding ensures the model is aware of the sequence order.

## Training

The model is trained on the CodeParrot dataset, which contains a large corpus of Python code. The training process includes:

- Gradient accumulation for effective batch processing
- Learning rate scheduling with warm-up
- Early stopping to prevent overfitting
- Regular generation of sample outputs to monitor progress

## Results

The model shows promising results in generating code-like text, including function definitions and class structures. The addition of self-attention layers has improved the model's ability to maintain context over longer sequences. However, as an experimental project, the output may not always be syntactically correct or fully coherent.

## Future Work

- Fine-tune hyperparameters for improved performance
- Experiment with different reservoir sizes and attention mechanisms
- Implement more advanced regularization techniques
- Explore applications beyond code generation
- Investigate the impact of different positional encoding schemes

## Acknowledgements

This project was developed with the assistance of Claude, an AI language model created by Anthropic, PBC.

## License

GPL3
