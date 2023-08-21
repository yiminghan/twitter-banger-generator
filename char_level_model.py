########
#
# Character Level Training
# Based on Karpathy's lecture #6
# https://www.youtube.com/watch?v=kCc8FmEb1nY
#
########

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import argparse

# hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 128  # what is the maximum context length for predictions?
max_iters = 100000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "mps"
n_embed = 128
n_head = 4
eval_iters = 200
n_layer = 4  # How much multi head attention layers
dropout = 0.2
# ------------

outdir = "./checkpoints/char_level_models"

torch.set_default_device(device)

# ls ./scrap |  cut -f1 -d'.'
poaster_ids = [
    # "10x_er",
    # "1a1n1d1y",
    # "BasedBeffJezos",
    # "Ryan_Gasoline",
    # "Soul0Engineer",
    # "TheWeebDev",
    # "anammostarac",
    # "ctjlewis",
    # "fabiankunick",
    "goth600",
    # "growing_daniel",
    # "kosenjuu",
    # "powerbottomdad1",
    # "realGeorgeHotz",
    # "shauseth",
    # "skooookum",
    # "t3dotgg",
    # "tekbog",
    # "tszzl",
    # "var_epsilon",
    # "wagieeacc",
    # "wireless_anon",
    # "xlr8harder",
    # "yacineMTB",
]


class FeedForwardBlock(nn.Module):
    def __init__(self, n_embed) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttentionHead(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        wei = q @ k.transpose(-2, -1) * 16**-0.5
        # float('-inf') blocks the communication from future to past
        # but we can actually adjust this later on if we want to leak future info to past
        wei = wei.masked_fill(self.tril[:block_size, :block_size] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfAttentionHead(head_size) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(x))


class AttentionBlock(nn.Module):
    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed // n_heads
        self.attn = MultiHeadAttention(n_heads, head_size)
        self.ff = FeedForwardBlock(n_embed)
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.attn(self.layer_norm1(x))
        x = x + self.ff(self.layer_norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # language model head
        self.attn_blocks = nn.Sequential(
            *[AttentionBlock(n_embed, n_heads=n_head) for _ in range(n_layer)],
            nn.LayerNorm(n_embed),
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        token_emb = self.token_embedding_table(idx)  # (B,T, n_embed)
        pos_emb = self.position_embedding_table(torch.arange(T))  # (T, n_embed)
        x = token_emb + pos_emb  # (B,T, n_embed)
        x = self.attn_blocks(x)  # (B,T, n_embed)
        logits = self.lm_head(x)  # (B,T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # (B, T)
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


def train(poaster_id):
    print(f"=== Training for {poaster_id} ===")
    with open(f"./scrap/{poaster_id}.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [
        stoi[c] for c in s
    ]  # encoder: take a string, output a list of integers
    decode = lambda l: "".join(
        [itos[i] for i in l]
    )  # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    # data loading
    def get_batch(split):
        # generate a small batch of data of inputs x and targets y
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    model = Transformer(vocab_size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")

    m = model.to(device)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        # sample a batch of data
        xb, yb = get_batch("train")

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    print(f"saving checkpoint to {outdir}")
    torch.save(checkpoint, os.path.join(outdir, f"{poaster_id}.pt"))
    # generate from the model
    context = torch.zeros((1, block_size), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


def sample(poaster_id):
    with open(f"./scrap/{poaster_id}.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # create a mapping from characters to integers
    itos = {i: ch for i, ch in enumerate(chars)}
    decode = lambda l: "".join(
        [itos[i] for i in l]
    )  # decoder: take a list of integers, output a string

    model = Transformer(vocab_size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    model_path = os.path.join(outdir, f"{poaster_id}.pt")
    model_dict = torch.load(model_path)["model"]
    model.load_state_dict(model_dict)
    print(f"====== Load From {model_path} =====")
    m = model.to(device)

    context = torch.zeros((1, block_size), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()).strip())


parser = argparse.ArgumentParser(
    prog="Twitter Bangers",
    description="Char level transformer model for generating tweets from highbie poasters",
)

parser.add_argument("twitter_id")  # positional argument
parser.add_argument("-t", "--train", action="store_true")
parser.add_argument("-s", "--seed", default=1337)


args = parser.parse_args()
torch.manual_seed(args.seed)
print(args)


if args.train == True:
    # Training parameters are default for now
    train(args.twitter_id)

else:
    sample(args.twitter_id)


# for p in poaster_ids:
# train(p)
