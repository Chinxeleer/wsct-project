import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
import argparse

# =============================================================================
# Vocabulary Definition
# =============================================================================

COLORS = ["red", "blue", "green", "yellow"]
SHAPES = ["circle", "square", "star", "cross"]
QUANTITIES = ["1", "2", "3", "4"]

# Build vocabulary
TOKEN_TO_IDX = {}
IDX_TO_TOKEN = {}

def build_vocab():
    """Build vocabulary mapping for cards, categories, and special tokens."""
    idx = 0
    # Card tokens (0-63): ordered by color, shape, quantity
    for color in COLORS:
        for shape in SHAPES:
            for qty in QUANTITIES:
                token = f"{color}|{shape}|{qty}"
                TOKEN_TO_IDX[token] = idx
                IDX_TO_TOKEN[idx] = token
                idx += 1
    
    # Category tokens (64-67)
    for i, cat in enumerate(["C1", "C2", "C3", "C4"]):
        TOKEN_TO_IDX[cat] = idx + i
        IDX_TO_TOKEN[idx + i] = cat
    idx += 4
    
    # Special tokens (68-69)
    TOKEN_TO_IDX["SEP"] = idx
    IDX_TO_TOKEN[idx] = "SEP"
    TOKEN_TO_IDX["EOS"] = idx + 1
    IDX_TO_TOKEN[idx + 1] = "EOS"

build_vocab()
VOCAB_SIZE = len(TOKEN_TO_IDX)  # 70

# =============================================================================
# Data Processing
# =============================================================================

def parse_card(token_str: str):
    """Parse a card token from various formats."""
    # Remove brackets, quotes, and split by common delimiters
    cleaned = re.sub(r"[\[\]'\"]", "", token_str)
    parts = re.split(r"[,|\s]+", cleaned)
    parts = [p.strip() for p in parts if p.strip()]
    
    if len(parts) == 3:
        color, shape, qty = parts
        if color in COLORS and shape in SHAPES and qty in QUANTITIES:
            return f"{color}|{shape}|{qty}"
    return None

def parse_line(line: str) -> List[int]:
    """Parse a line from the dataset into token indices."""
    line = line.strip()
    if not line:
        return []
    
    # Check if already integer indices
    if re.match(r'^[\d\s]+$', line):
        return [int(x) for x in line.split()]
    
    # Parse token by token
    indices = []
    # Split by array notation or whitespace
    tokens = re.findall(r"array\([^\)]+\)|SEP|EOS|C[1-4]", line)
    
    for token in tokens:
        token = token.strip()
        
        # Handle array notation
        if token.startswith("array"):
            card = parse_card(token)
            if card and card in TOKEN_TO_IDX:
                indices.append(TOKEN_TO_IDX[card])
        # Handle special tokens
        elif token in TOKEN_TO_IDX:
            indices.append(TOKEN_TO_IDX[token])
    
    return indices

# =============================================================================
# Dataset
# =============================================================================

class WCSTDataset(Dataset):
    """Dataset for WCST sequences."""
    
    def __init__(self, sequences: List[List[int]], max_len: int = 128):
        self.max_len = max_len
        self.sequences = []
        
        for seq in sequences:
            if len(seq) > 1:  # Need at least 2 tokens for input/target pairs
                padded = self._pad_sequence(seq)
                self.sequences.append(padded)
    
    def _pad_sequence(self, seq: List[int]) -> List[int]:
        """Pad or truncate sequence to max_len."""
        if len(seq) >= self.max_len:
            return seq[:self.max_len]
        # Pad with EOS tokens
        return seq + [TOKEN_TO_IDX["EOS"]] * (self.max_len - len(seq))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        # Return input (all but last) and target (all but first)
        return seq[:-1], seq[1:]

def load_dataset(filepath: str, max_len: int = 128) -> WCSTDataset:
    """Load dataset from text file."""
    sequences = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    indices = parse_line(line)
                    if indices:
                        sequences.append(indices)
                except Exception as e:
                    print(f"Warning: Failed to parse line: {line[:50]}... Error: {e}")
    
    print(f"Loaded {len(sequences)} sequences")
    return WCSTDataset(sequences, max_len)

# =============================================================================
# Model Components
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input."""
        return x + self.pe[:x.size(1)]

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Linear projections and split into heads
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        
        return output, attn_weights

class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """Transformer decoder block with self-attention and feed-forward."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out, attn_weights = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x, attn_weights

class WCSTTransformer(nn.Module):
    """Decoder-only transformer for WCST."""
    
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 512,
        max_len: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small random values."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        # Embed tokens and add positional encoding
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Apply transformer blocks
        attn_maps = []
        for block in self.blocks:
            x, attn_weights = block(x, mask)
            attn_maps.append(attn_weights)
        
        # Final normalization and output projection
        x = self.norm(x)
        logits = self.output(x)
        
        return logits, attn_maps

# =============================================================================
# Training Utilities
# =============================================================================

def create_causal_mask(seq_len: int, device: torch.device):
    """Create causal mask for autoregressive generation."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return (mask == 0).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Create causal mask
        mask = create_causal_mask(inputs.size(1), device)
        
        # Forward pass
        logits, _ = model(inputs, mask)
        
        # Compute loss
        loss = criterion(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * inputs.size(0)
        
        predictions = logits.argmax(dim=-1)
        mask_valid = targets != TOKEN_TO_IDX["EOS"]
        total_correct += ((predictions == targets) & mask_valid).sum().item()
        total_tokens += mask_valid.sum().item()
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            mask = create_causal_mask(inputs.size(1), device)
            logits, _ = model(inputs, mask)
            
            loss = criterion(logits.view(-1, VOCAB_SIZE), targets.view(-1))
            total_loss += loss.item() * inputs.size(0)
            
            predictions = logits.argmax(dim=-1)
            mask_valid = targets != TOKEN_TO_IDX["EOS"]
            total_correct += ((predictions == targets) & mask_valid).sum().item()
            total_tokens += mask_valid.sum().item()
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    return avg_loss, accuracy

# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train transformer on WCST")
    parser.add_argument('--data', type=str, required=True, help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=512, help='Feed-forward dimension')
    parser.add_argument('--max_len', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_path', type=str, default='wcst_model.pt', help='Model save path')
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    
    # Load dataset
    dataset = load_dataset(args.data, args.max_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model
    model = WCSTTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_len=args.max_len,
        dropout=args.dropout
    ).to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=TOKEN_TO_IDX["EOS"])
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, dataloader, optimizer, criterion, args.device)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'accuracy': train_acc,
                'args': args
            }, args.save_path)
            print(f"  Saved best model (loss: {best_loss:.4f})")
    
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Model saved to: {args.save_path}")

if __name__ == "__main__":
    main()
