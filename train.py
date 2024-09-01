import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
import numpy as np
import random
import math
from transformers import get_linear_schedule_with_warmup, RobertaTokenizerFast
import logging
from tqdm import tqdm
import os
import multiprocessing

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Set the multiprocessing start method to 'spawn'
multiprocessing.set_start_method('spawn', force=True)

# Set the device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# Increase the file descriptor limit
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

# Set environment variable to disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        weights = torch.randn(reservoir_size, input_size) * 0.02
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
        time_factor = torch.sin(self.frequencies[None, None, :] * t[:, :, None])

        # Combine reservoir output with time-dependent ripples
        return torch.tanh(r + time_factor * 0.1)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(1)  # Add batch and head dimensions
            scores = scores.masked_fill(mask == float('-inf'), float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(attention_output)

class LiquidLayer(nn.Module):
    def __init__(self, d_model, reservoir_size, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.reservoir = TimeAwareReservoirLayer(d_model, reservoir_size)
        self.reservoir_proj = nn.Linear(reservoir_size, d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t, mask=None):
        # Multi-head attention with mask
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Time-aware reservoir
        reservoir_output = self.reservoir(x, t)
        reservoir_output = self.reservoir_proj(reservoir_output)
        x = self.norm2(x + self.dropout(reservoir_output))

        # Feed-forward network
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x

class LiquidLM(nn.Module):
    def __init__(self, vocab_size, d_model, reservoir_size, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([LiquidLayer(d_model, reservoir_size, num_heads, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, t):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)

        mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)

        for layer in self.layers:
            x = layer(x, t, mask)

        return self.fc_out(x)

class CodeParrotDataset(Dataset):
    def __init__(self, tokenizer, max_length=2048, max_samples=50000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples

        dataset = load_dataset("codeparrot/codeparrot-clean-valid", split="train")
        self.data = dataset.select(range(min(len(dataset), max_samples)))
        self.data = self.data.map(self.preprocess, remove_columns=dataset.column_names)
        self.data = self.data.filter(lambda x: len(x['input_ids']) > 0)

    def preprocess(self, example):
        content = example['content']
        encoded = self.tokenizer.encode_plus(
            content,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze().tolist(),
            'attention_mask': encoded['attention_mask'].squeeze().tolist(),
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long)
        }

def generate_text(model, tokenizer, start_string, length=100, temperature=0.7):
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(start_string)).unsqueeze(0).to(device)
    generated_text = start_string

    with torch.no_grad():
        for i in range(length):
            t = torch.arange(input_ids.size(1)).unsqueeze(0).float().to(device)
            
            predictions = model(input_ids, t)
            predictions = predictions[:, -1, :] / temperature
            
            # Handle potential NaN values
            if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                logging.warning("NaN or Inf detected in predictions. Stopping generation.")
                break
            
            # Use softmax with careful normalization
            probs = torch.softmax(predictions - predictions.max(dim=-1, keepdim=True).values, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).squeeze()

            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=-1)
            next_token = tokenizer.decode([next_token_id.item()])
            generated_text += next_token

    return generated_text

def train(model, dataset, tokenizer, epochs, lr, batch_size=32, accumulation_steps=4):
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

    total_steps = len(train_loader) * epochs // accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, data in enumerate(progress_bar):
            x = data['input_ids'].to(device)
            y = x.clone().roll(-1, dims=1)
            y[:, -1] = tokenizer.pad_token_id

            t = torch.arange(x.size(1)).unsqueeze(0).repeat(x.size(0), 1).float().to(device)

            with torch.amp.autocast('cuda'):
                output = model(x, t)
                loss = criterion(output.reshape(-1, output.size(-1)), y.reshape(-1))
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            progress_bar.set_postfix({'loss': loss.item() * accumulation_steps})

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

        sample_text = generate_text(model, tokenizer, "import pygame:", length=100)
        logging.info(f'Sample generated text:\n{sample_text}')

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Model parameters
    vocab_size = 50265  # RobertaTokenizer default vocab size
    d_model = 768
    reservoir_size = 2048
    num_heads = 12
    num_layers = 6
    dropout_rate = 0.1
    lr = 5e-5

    batch_size = 16
    epochs = 30
    accumulation_steps = 4
    sequence_length = 512

    logging.info(f"Initializing model with parameters: d_model={d_model}, reservoir_size={reservoir_size}, num_heads={num_heads}, num_layers={num_layers}")

    tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")

    dataset = CodeParrotDataset(tokenizer, max_length=sequence_length, max_samples=50000)
    model = LiquidLM(
        vocab_size, d_model, reservoir_size, 
        num_heads, num_layers, dropout_rate
    ).to(device)

    logging.info("Starting training...")
    train(model, dataset, tokenizer, epochs, lr, batch_size, accumulation_steps)

    logging.info("Training completed. Loading best model...")
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    logging.info("Generating sample text...")
    generated_text = generate_text(model, tokenizer, "import pygame", length=200)
    logging.info(f"Generated text:\n{generated_text}")
