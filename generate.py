import torch
import torch.nn as nn
from transformers import RobertaTokenizerFast
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

class LiquidLayer(nn.Module):
    def __init__(self, input_size, reservoir_size, hidden_size, output_size, num_layers=4, dropout_rate=0.4):
        super().__init__()
        self.reservoir = TimeAwareReservoirLayer(input_size, reservoir_size)
        self.recurrent_layer = nn.LSTM(reservoir_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, hidden_state, t):
        reservoir_out = self.reservoir(x, t)
        reservoir_out = self.dropout(reservoir_out)
        output, new_hidden_state = self.recurrent_layer(reservoir_out, hidden_state)
        output = self.layer_norm(output)
        output = self.dropout(output)
        output = self.output_layer(output)
        return output, new_hidden_state

class LiquidLM(nn.Module):
    def __init__(self, vocab_size, reservoir_size, hidden_size, num_layers=4, dropout_rate=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.liquid_layer = LiquidLayer(hidden_size, reservoir_size, hidden_size, vocab_size, num_layers, dropout_rate)

    def forward(self, x, hidden_state, t):
        embedded = self.embedding(x)
        output, new_hidden_state = self.liquid_layer(embedded, hidden_state, t)
        return output, new_hidden_state

    def init_hidden(self, batch_size):
        return (torch.zeros(self.liquid_layer.recurrent_layer.num_layers, batch_size, self.liquid_layer.recurrent_layer.hidden_size),
                torch.zeros(self.liquid_layer.recurrent_layer.num_layers, batch_size, self.liquid_layer.recurrent_layer.hidden_size))

def generate_text(model, tokenizer, start_string, length=2000, temperature=0.7):
    model.eval()
    device = next(model.parameters()).device
    hidden = model.init_hidden(1)
    hidden = tuple(h.to(device) for h in hidden)
    input_ids = torch.tensor(tokenizer.encode(start_string)).unsqueeze(0).to(device)
    generated_text = start_string

    with torch.no_grad():
        for i in range(length):
            t = torch.tensor([i]).float().to(device)
            predictions, hidden = model(input_ids, hidden, t)
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
    reservoir_size = 4096
    hidden_size = 1024
    num_layers = 4
    dropout_rate = 0.4

    # Initialize the model
    model = LiquidLM(tokenizer.vocab_size, reservoir_size, hidden_size, num_layers, dropout_rate).to(device)

    # Load the trained model
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Model loaded from {args.model_path}")

    # Generate text
    generated_text = generate_text(model, tokenizer, args.prompt, args.length, args.temperature)
    print(f"Generated text:\n{generated_text}")

if __name__ == "__main__":
    main()
