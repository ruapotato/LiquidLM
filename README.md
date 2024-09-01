# LiquidLM: Time-Aware Reservoir Computing with Self-Attention for Language Modeling
By David Hamner

## Overview
LiquidLM is an experimental language model that combines reservoir computing with time-aware dynamics and self-attention mechanisms for code generation. This project explores a novel architecture that integrates a Time-Aware Reservoir Layer with self-attention layers to create a unique approach to language modeling.

## Latest Update
- Integrated self-attention layers with Time-Aware Reservoir for enhanced context modeling

## Features
- Time-Aware Reservoir Layer for dynamic computations
- Self-attention layers for capturing long-range dependencies
- Positional encoding for sequence awareness
- Gradient accumulation and learning rate scheduling
- Early stopping mechanism for efficient training
- Sample generation to showcase model capabilities
- Improved context modeling through integrated self-attention and reservoir computing

## Requirements
- Python 3.7+
- PyTorch
- Transformers library
- Datasets library
- NumPy
- tqdm

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

1. Embedding Layer: Converts input tokens to dense vector representations.
2. Positional Encoding: Adds information about the position of tokens in the sequence.
3. Multiple LiquidLayers, each containing:
   a. Multi-Head Attention: Allows the model to attend to different parts of the input.
   b. Time-Aware Reservoir Layer: Introduces time-dependent dynamics to the model.
   c. Feed-Forward Network (MLP): Further processes the information.
4. Linear output layer: Projects the final hidden states to vocabulary size for token prediction.

The core of the model is the LiquidLayer, which implements an Attention -> Reservoir -> MLP flow:

1. Multi-Head Attention: 
   - Processes the input through self-attention mechanism.
   - Followed by layer normalization and residual connection.

2. Time-Aware Reservoir:
   - Applies a sparse, frozen linear transformation (the reservoir).
   - Adds time-dependent sinusoidal "ripples" to the reservoir output.
   - Projects the reservoir output back to the model dimension.
   - Followed by layer normalization and residual connection.

3. Feed-Forward Network (MLP):
   - Consists of two linear transformations with a ReLU activation in between.
   - Followed by layer normalization and residual connection.

This architecture combines the strengths of attention mechanisms for capturing long-range dependencies, reservoir computing for introducing dynamic temporal features, and traditional feed-forward networks for non-linear transformations. The Time-Aware Reservoir Layer is a novel component that aims to capture temporal patterns in code sequences, which could be particularly useful for understanding the structure and flow of programming languages.

The model uses a causal mask in the attention mechanism to ensure that predictions for a given token can only depend on the tokens that come before it, making it suitable for autoregressive language modeling tasks.

## Training
The model is trained on the CodeParrot dataset, which contains a large corpus of Python code. The training process includes:
- Gradient accumulation for effective batch processing
- Learning rate scheduling with warm-up
- Early stopping to prevent overfitting
- Regular generation of sample outputs to monitor progress
- Adaptive gradient scaling for stable training on GPUs

## Results
The model shows promising results in generating code-like text, including function definitions and class structures. The addition of self-attention layers has improved the model's ability to maintain context over longer sequences. The integration of self-attention with the Time-Aware Reservoir has further enhanced the model's capacity to capture both short-term and long-term dependencies in code. However, as an experimental project, the output may not always be syntactically correct or fully coherent.

## Future Work
- Fine-tune hyperparameters for improved performance
- Experiment with different reservoir sizes and attention mechanisms
- Implement more advanced regularization techniques
- Explore applications beyond code generation
- Investigate the impact of different positional encoding schemes
- Analyze the interplay between reservoir dynamics and self-attention mechanisms

## Acknowledgements
This project was developed with the assistance of Claude, an AI language model created by Anthropic, PBC.

## License
GPL3
