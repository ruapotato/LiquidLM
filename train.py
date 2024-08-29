import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
import numpy as np
import random
import math
from transformers import get_linear_schedule_with_warmup, RobertaTokenizerFast
import signal
import sys
import logging
import gc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class TimeAwareReservoirLayer(nn.Module):
    def __init__(self, input_size, reservoir_size, sparsity=0.95):
        super().__init__()
        self.reservoir = nn.Linear(input_size, reservoir_size, bias=False)
        self.reservoir_size = reservoir_size

        # Initialize the reservoir weights randomly and make them sparse
        weights = torch.randn(reservoir_size, input_size) * 0.1
        mask = torch.rand(reservoir_size, input_size) > sparsity
        self.reservoir.weight.data = weights * mask

        # Freeze the reservoir weights
        self.reservoir.weight.requires_grad = False

        # Initialize time-dependent frequencies for ripples
        self.frequencies = nn.Parameter(torch.randn(reservoir_size) * 0.01, requires_grad=False)

    def forward(self, x, t):
        # Apply the reservoir transformation
        r = self.reservoir(x)

        # Create time-dependent ripples using sinusoidal encoding
        time_factor = torch.sin(self.frequencies * t.unsqueeze(-1)) + torch.cos(self.frequencies * t.unsqueeze(-1))

        # Combine reservoir output with time-dependent ripples
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

def train(model, train_loader, val_loader, tokenizer, epochs, lr, device, accumulation_steps=8):
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Learning rate scheduler with warm-up
    total_steps = len(train_loader) * epochs // accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 20, num_training_steps=total_steps)

    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0

    def signal_handler(sig, frame):
        logging.info("Ctrl+C detected. Stopping training...")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, signal_handler)

    try:
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            optimizer.zero_grad()

            for batch_idx, data in enumerate(train_loader):
                x = data['input_ids'].to(device)
                y = x.clone().roll(-1, dims=1)
                y[:, -1] = tokenizer.pad_token_id

                hidden = model.init_hidden(x.size(0))
                hidden = tuple(h.to(device) for h in hidden)

                # Create a time tensor for the sequence
                t = torch.arange(x.size(1)).float().to(device)

                output, _ = model(x, hidden, t)
                loss = criterion(output.reshape(-1, output.size(-1)), y.reshape(-1))
                loss = loss / accumulation_steps
                loss.backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * accumulation_steps

                if batch_idx % 100 == 0:
                    logging.info(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item() * accumulation_steps:.4f}')

                # Clear CUDA cache to free up memory
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            avg_loss = total_loss / len(train_loader)
            logging.info(f'Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}')

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data in val_loader:
                    x = data['input_ids'].to(device)
                    y = x.clone().roll(-1, dims=1)
                    y[:, -1] = tokenizer.pad_token_id

                    hidden = model.init_hidden(x.size(0))
                    hidden = tuple(h.to(device) for h in hidden)
                    t = torch.arange(x.size(1)).float().to(device)
                    output, _ = model(x, hidden, t)
                    loss = criterion(output.reshape(-1, output.size(-1)), y.reshape(-1))
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            logging.info(f'Validation Loss: {avg_val_loss:.4f}')

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f'Early stopping triggered after {epoch+1} epochs')
                    break

            # Generate sample text
            sample_text = generate_text(model, tokenizer, "def main():", length=500)
            logging.info(f'Sample generated text:\n{sample_text}')

            # Clear CUDA cache after each epoch
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        logging.info("Training interrupted. Saving the model...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_val_loss,
        }, 'interrupted_model.pth')
        raise

    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")
        raise

def tokenize_function(examples, tokenizer):
    return tokenizer(examples['content'], truncation=True, padding='max_length', max_length=512)

def collate_fn(batch):
    return {
        'input_ids': torch.tensor([item['input_ids'] for item in batch])
    }

# Main execution
if __name__ == "__main__":
    try:
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)

        # Hyperparameters
        reservoir_size = 4096
        hidden_size = 1024
        batch_size = 32
        epochs = 60
        lr = 0.00005
        dropout_rate = 0.4
        accumulation_steps = 16
        sequence_length = 512
        num_layers = 4

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")

        # Load dataset
        dataset = load_dataset("codeparrot/codeparrot-clean-valid")

        # Create a tokenizer
        tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")

        # Tokenize the dataset
        tokenized_datasets = dataset.map(
            lambda examples: tokenize_function(examples, tokenizer),
            batched=True,
            remove_columns=dataset["train"].column_names,
            num_proc=4,
        )

        # Split the dataset into train and validation
        train_val_dataset = tokenized_datasets["train"]
        train_size = int(0.9 * len(train_val_dataset))
        val_size = len(train_val_dataset) - train_size
        train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

        # Initialize the model
        model = LiquidLM(tokenizer.vocab_size, reservoir_size, hidden_size, num_layers, dropout_rate).to(device)

        # Train the model
        train(model, train_loader, val_loader, tokenizer, epochs, lr, device, accumulation_steps)

        # Load the best model
        checkpoint = torch.load('best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])

        # Generate some text
        generated_text = generate_text(model, tokenizer, "def fibonacci(n):", length=2000)
        logging.info(f"Generated text:\n{generated_text}")

    except KeyboardInterrupt:
        logging.info("Script interrupted by user. Exiting...")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        raise
