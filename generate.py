import torch
import torch.nn as nn
from transformers import RobertaTokenizerFast
import argparse
import logging
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TimeAwareReservoirLayer(nn.Module):
    def __init__(self, input_size, reservoir_size, sparsity=0.95):
        super().__init__()
        self.reservoir = nn.Linear(input_size, reservoir_size, bias=False)
        self.reservoir_size = reservoir_size
        self.frequencies = nn.Parameter(torch.randn(reservoir_size) * 0.01, requires_grad=False)

    def forward(self, x, t):
        r = self.reservoir(x)
        time_factor = torch.sin(self.frequencies * t.unsqueeze(-1)) + torch.cos(self.frequencies * t.unsqueeze(-1))
        return torch.tanh(r + time_factor)

class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        return src

class LiquidLayer(nn.Module):
    def __init__(self, input_size, reservoir_size, hidden_size, output_size, num_layers=8, nhead=16, dropout_rate=0.2):
        super().__init__()
        self.reservoir = TimeAwareReservoirLayer(input_size, reservoir_size)
        self.pos_encoder = PositionalEncoding(reservoir_size)
        self.attention_layers = nn.ModuleList([SelfAttentionLayer(reservoir_size, nhead, dropout_rate) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(reservoir_size)
        self.output_layer = nn.Linear(reservoir_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, t):
        reservoir_out = self.reservoir(x, t)
        reservoir_out = self.pos_encoder(reservoir_out)
        reservoir_out = self.dropout(reservoir_out)
        
        for layer in self.attention_layers:
            reservoir_out = layer(reservoir_out)
        
        output = self.layer_norm(reservoir_out)
        output = self.dropout(output)
        output = self.output_layer(output)
        return output

class LiquidLM(nn.Module):
    def __init__(self, vocab_size, reservoir_size, hidden_size, num_layers=8, nhead=16, dropout_rate=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.liquid_layer = LiquidLayer(hidden_size, reservoir_size, hidden_size, vocab_size, num_layers, nhead, dropout_rate)

    def forward(self, x, t):
        embedded = self.embedding(x)
        output = self.liquid_layer(embedded, t)
        return output

def generate_text(model, tokenizer, start_string, length=2000, temperature=0.7):
    model.eval()
    device = next(model.parameters()).device
    input_ids = torch.tensor(tokenizer.encode(start_string)).unsqueeze(0).to(device)
    generated_text = start_string

    with torch.no_grad():
        for i in range(length):
            t = torch.arange(input_ids.size(1)).unsqueeze(0).float().to(device)
            predictions = model(input_ids, t)
            predictions = predictions[:, -1, :] / temperature
            next_token_id = torch.multinomial(torch.softmax(predictions, dim=-1), num_samples=1).squeeze()

            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=-1)
            next_token = tokenizer.decode([next_token_id.item()])
            generated_text += next_token

    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Generate text using LiquidLM")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--prompt", type=str, default="def main():", help="Starting prompt for text generation")
    parser.add_argument("--length", type=int, default=2000, help="Length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for text generation")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load the tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")

    # Hyperparameters (should match the training script)
    reservoir_size = 3072
    hidden_size = 1024
    num_layers = 8
    nhead = 16
    dropout_rate = 0.2

    # Initialize the model
    model = LiquidLM(tokenizer.vocab_size, reservoir_size, hidden_size, num_layers, nhead, dropout_rate).to(device)

    # Load the trained model
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Model loaded from {args.model_path}")

    # Generate text
    generated_text = generate_text(model, tokenizer, args.prompt, args.length, args.temperature)
    print(f"Generated text:\n{generated_text}")

if __name__ == "__main__":
    main()
