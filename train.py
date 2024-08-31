import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
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
    def __init__(self, input_size, reservoir_size, hidden_size, output_size, num_layers=4, nhead=16, dropout_rate=0.2):
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

class CodeParrotDataset(Dataset):
    def __init__(self, tokenizer, max_length=2048, max_samples=50000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples

        # Load a subset of the data
        self.data = load_dataset("codeparrot/codeparrot-clean-valid", split="train")

        # Filter out long examples
        self.data = self.data.filter(lambda example: self.check_length(example))

        # Limit the number of samples
        if len(self.data) > max_samples:
            self.data = self.data.select(range(max_samples))

        logging.info(f"Total examples after filtering: {len(self.data)}")

    def check_length(self, example):
        return len(self.tokenizer.encode(example['content'])) < self.max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = self.tokenizer.encode_plus(
            item['content'],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
        }

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

def train(model, dataset, tokenizer, epochs, lr, device, accumulation_steps=4):
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = len(dataset) * epochs // accumulation_steps
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
            # Resample the dataset for each epoch
            indices = torch.randperm(len(dataset)).tolist()[:50000]
            epoch_dataset = Subset(dataset, indices)
            
            train_size = int(0.9 * len(epoch_dataset))
            val_size = len(epoch_dataset) - train_size
            train_dataset, val_dataset = random_split(epoch_dataset, [train_size, val_size])

            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=4)

            model.train()
            total_loss = 0
            optimizer.zero_grad()

            for batch_idx, data in enumerate(train_loader):
                x = data['input_ids'].to(device)
                y = x.clone().roll(-1, dims=1)
                y[:, -1] = tokenizer.pad_token_id

                t = torch.arange(x.size(1)).unsqueeze(0).repeat(x.size(0), 1).float().to(device)

                output = model(x, t)
                loss = criterion(output.reshape(-1, output.size(-1)), y.reshape(-1))
                loss = loss / accumulation_steps
                loss.backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * accumulation_steps

                if batch_idx % 100 == 0:
                    logging.info(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item() * accumulation_steps:.4f}')

                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            avg_loss = total_loss / len(train_loader)
            logging.info(f'Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}')

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data in val_loader:
                    x = data['input_ids'].to(device)
                    y = x.clone().roll(-1, dims=1)
                    y[:, -1] = tokenizer.pad_token_id

                    t = torch.arange(x.size(1)).unsqueeze(0).repeat(x.size(0), 1).float().to(device)
                    output = model(x, t)
                    loss = criterion(output.reshape(-1, output.size(-1)), y.reshape(-1))
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            logging.info(f'Validation Loss: {avg_val_loss:.4f}')

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

            sample_text = generate_text(model, tokenizer, "def fibonacci(n):", length=500)
            logging.info(f'Sample generated text:\n{sample_text}')

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

if __name__ == "__main__":
    try:
        torch.autograd.set_detect_anomaly(True)

        reservoir_size = 3072
        hidden_size = 1024
        batch_size = 4
        epochs = 60
        lr = 0.00005
        dropout_rate = 0.2
        accumulation_steps = 4
        sequence_length = 2048
        num_layers = 8
        nhead = 16

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")

        tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")

        dataset = CodeParrotDataset(tokenizer, max_length=sequence_length, max_samples=50000)

        model = LiquidLM(tokenizer.vocab_size, reservoir_size, hidden_size, num_layers, nhead, dropout_rate).to(device)

        train(model, dataset, tokenizer, epochs, lr, device, accumulation_steps)

        checkpoint = torch.load('best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])

        generated_text = generate_text(model, tokenizer, "def fibonacci(n):", length=2000)
        logging.info(f"Generated text:\n{generated_text}")

    except KeyboardInterrupt:
        logging.info("Script interrupted by user. Exiting...")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        raise
