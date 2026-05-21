#!/usr/bin/env python3
"""
Transformer Ablation Experiments
Task: Character-level language modeling
Reproduces core ideas from "Attention Is All You Need" (Vaswani et al., 2017)
and studies the necessity of each component.
"""

import math, time, json, random, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────── Reproducibility ────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ──────────────────────────── Hyper-params ──────────────────────────────
CFG = dict(
    d_model   = 64,
    n_heads   = 4,
    d_ff      = 128,
    n_layers  = 2,
    seq_len   = 64,
    batch     = 64,
    n_iters   = 1200,   # reduced for CPU speed
    lr        = 1e-3,
    dropout   = 0.1,
    log_every = 200,
)

# ─────────────────────────── Corpus (Alice in Wonderland, public domain) ─
CORPUS = """
Alice was beginning to get very tired of sitting by her sister on the bank,
and of having nothing to do: once or twice she had peeped into the book her
sister was reading, but it had no pictures or conversations in it, and what
is the use of a book, thought Alice, without pictures or conversation? So she
was considering in her own mind as well as she could, for the hot day made her
feel very sleepy and stupid, whether the pleasure of making a daisy-chain would
be worth the trouble of getting up and picking the daisies, when suddenly a
White Rabbit with pink eyes ran close by her. There was nothing so very
remarkable in that; nor did Alice think it so very much out of the way to hear
the Rabbit say to itself, Oh dear! Oh dear! I shall be late! When she thought
it over afterwards, it occurred to her that she ought to have wondered at this,
but at the time it all seemed quite natural; but when the Rabbit actually took
a watch out of its waistcoat-pocket, and looked at it, and then hurried on,
Alice started to her feet, for it flashed across her mind that she had never
before seen a rabbit with either a waistcoat-pocket, or a watch to take out
of it, and burning with curiosity, she ran across the field after it, and
fortunately was just in time to see it pop down a large rabbit-hole under the
hedge. In another moment down went Alice after it, never once considering how
in the world she was to get out again. The rabbit-hole went straight on like a
tunnel for some way, and then dipped suddenly down, so suddenly that Alice had
not a moment to think about stopping herself before she found herself falling
down a very deep well. Either the well was very deep, or she fell very slowly,
for she had plenty of time as she went down to look about her and to wonder
what was going to happen next. First, she tried to look down and make out what
she was coming to, but it was too dark to see anything; then she looked at the
sides of the well, and noticed that they were filled with cupboards and
book-shelves; here and there she saw maps and pictures hung upon pegs. She
took down a jar from one of the shelves as she passed; it was labelled
ORANGE MARMALADE but to her great disappointment it was empty; she did not
like to drop the jar for fear of killing somebody underneath, so managed to put
it into one of the cupboards as she fell past it. Well, thought Alice to
herself, so after such a fall as this, I shall think nothing of tumbling down
stairs! How brave they will all think me at home! Why, I would not say anything
about it, even if I fell off the top of the house! Which was very likely true.
Down, down, down. Would the fall never come to an end? I wonder how many miles
I have fallen by this time? she said aloud. I must be getting somewhere near
the centre of the earth. Let me see: that would be four thousand miles down,
I think, for you see, Alice had learnt several things of this sort in her
lessons in the schoolroom, and though this was not a very good opportunity for
showing off her knowledge, as there was no one to listen to her, still it was
good practice to say it over; and she went on, yes, that is about the right
distance, but then I wonder what Latitude or Longitude I have got to? Alice
had no idea what Latitude was, or Longitude either, but thought they were nice
grand words to say. Presently she began again. I wonder if I shall fall right
through the earth! How funny it will seem to come out among the people that
walk with their heads downward! The Antipathies, I think, she was rather glad
there was no one listening, this time, as it did not sound at all the right
word, but she could not remember the right word for it. Yes, that is quite
the right distance; but then I wonder what Latitude or Longitude I have got
to? And she tried to fancy what the flame of a candle looks like after the
candle is blown out, for she could not remember ever having seen such a thing.
After a while, finding that nothing more happened, she decided on going into
the garden at once; but, alas for poor Alice! when she got to the door, she
found she had forgotten the little golden key, and when she went back to the
table for it, she found she could not possibly reach it: she could see it
quite plainly through the glass, and she tried her best to climb up one of
the legs of the table, but it was too slippery; and when she had tired herself
out with trying, the poor little thing sat down and cried. Come, there is no
use in crying like that! said Alice to herself, rather sharply. I advise you
to leave off this minute! She generally gave herself very good advice, though
she very seldom followed it, and sometimes she scolded herself so severely as
to bring tears into her eyes; and once she remembered trying to box her own
ears for having cheated herself in a game of croquet she was playing against
herself, for this curious child was very fond of pretending to be two people.
But it is no use now, thought poor Alice, to pretend to be two people! Why,
there is hardly enough of me left to make one respectable person!
""" * 4  # repeat for more data


# ═══════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════

class CharDataset:
    def __init__(self, text, seq_len=64):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.seq_len = seq_len
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def get_batch(self, batch_size):
        ix = torch.randint(len(self.data) - self.seq_len - 1, (batch_size,))
        x = torch.stack([self.data[i:i + self.seq_len] for i in ix])
        y = torch.stack([self.data[i + 1:i + self.seq_len + 1] for i in ix])
        return x, y


# ═══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODINGS
# ═══════════════════════════════════════════════════════════════════════

class SinusoidalPE(nn.Module):
    """Original paper: fixed sine/cosine encoding."""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class LearnedPE(nn.Module):
    """Learned absolute positional embeddings (common modern default)."""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.drop(x + self.emb(pos))


class LinearPE(nn.Module):
    """Simple linear encoding: position / max_len broadcast across d_model."""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.max_len = max_len

    def forward(self, x):
        T = x.size(1)
        pos = (torch.arange(T, device=x.device).float() / self.max_len)
        pos = pos.unsqueeze(0).unsqueeze(-1).expand(x.size(0), T, x.size(2))
        return self.drop(x + pos)


class NoPE(nn.Module):
    """No positional encoding — ablation baseline."""
    def __init__(self, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(x)


# ═══════════════════════════════════════════════════════════════════════
#  ATTENTION VARIANTS
# ═══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """Standard Q, K, V multi-head attention (Vaswani et al.)."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.Wq(x).view(B, T, self.h, self.dk).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.h, self.dk).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.h, self.dk).transpose(1, 2)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        if mask is not None:
            s = s.masked_fill(mask == 0, -1e9)
        a = self.drop(F.softmax(s, dim=-1))
        out = (a @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.Wo(out)


class MergedKVAttention(nn.Module):
    """Ablation 2.2: K and V are the same projection (V = K)."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)  # V will reuse this
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.Wq(x).view(B, T, self.h, self.dk).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.h, self.dk).transpose(1, 2)
        v = k  # KEY DIFFERENCE: V = K
        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        if mask is not None:
            s = s.masked_fill(mask == 0, -1e9)
        a = self.drop(F.softmax(s, dim=-1))
        out = (a @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.Wo(out)


class GatedMultiHeadAttention(nn.Module):
    """
    Experiment 2.5 — Proposed improvement: Content-Dependent Head Gating.

    Each token position computes a soft gate vector over the h attention heads
    based on its own embedding.  This lets the network dynamically suppress or
    amplify specific heads depending on the *content* at that position, rather
    than treating all heads equally at every step.

      gate_i(x_t) = sigmoid(x_t @ w_gate_i)   ∈ (0,1)
      output = W_O [ gate_1 * head_1 ; ... ; gate_h * head_h ]

    Extra params: d_model (gate projection), negligible overhead.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, n_heads)  # content-to-gate
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.Wq(x).view(B, T, self.h, self.dk).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.h, self.dk).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.h, self.dk).transpose(1, 2)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        if mask is not None:
            s = s.masked_fill(mask == 0, -1e9)
        a = self.drop(F.softmax(s, dim=-1))
        heads = (a @ v)  # (B, h, T, dk)
        # gates: (B, T, h) -> (B, h, T, 1)
        gates = torch.sigmoid(self.gate_proj(x)).permute(0, 2, 1).unsqueeze(-1)
        gated = (gates * heads).transpose(1, 2).contiguous().view(B, T, C)
        return self.Wo(gated)


# ═══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD
# ═══════════════════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════
#  TRANSFORMER BLOCK  (residual toggle for 2.3)
# ═══════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1,
                 use_residual=True, attn_cls=MultiHeadAttention):
        super().__init__()
        self.attn = attn_cls(d_model, n_heads, dropout)
        self.ff   = FeedForward(d_model, d_ff, dropout)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.use_residual = use_residual

    def forward(self, x, mask=None):
        if self.use_residual:
            x = x + self.attn(self.ln1(x), mask)
            x = x + self.ff(self.ln2(x))
        else:
            x = self.attn(self.ln1(x), mask)
            x = self.ff(self.ln2(x))
        return x


# ═══════════════════════════════════════════════════════════════════════
#  TRANSFORMER LANGUAGE MODEL
# ═══════════════════════════════════════════════════════════════════════

def _make_causal_mask(T, device):
    return torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, d_ff=128,
                 n_layers=2, max_len=512, dropout=0.1,
                 pe_type='sinusoidal', use_residual=True,
                 attn_cls=MultiHeadAttention):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

        pe_map = {
            'sinusoidal': lambda: SinusoidalPE(d_model, max_len, dropout),
            'learned':    lambda: LearnedPE(d_model, max_len, dropout),
            'linear':     lambda: LinearPE(d_model, max_len, dropout),
            'none':       lambda: NoPE(dropout),
        }
        self.pe = pe_map[pe_type]()

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, use_residual, attn_cls)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        mask = _make_causal_mask(x.size(1), x.device)
        x = self.pe(self.embed(x))
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln_f(x))


# ═══════════════════════════════════════════════════════════════════════
#  CNN LANGUAGE MODEL  (Experiment 2.4 comparison)
# ═══════════════════════════════════════════════════════════════════════

class CausalConv(nn.Module):
    """Left-padded causal 1-D convolution."""
    def __init__(self, ch, kernel, dilation=1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(ch, ch, kernel, dilation=dilation, padding=self.pad)

    def forward(self, x):  # x: (B, C, T)
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # trim right padding


class CNNBlock(nn.Module):
    def __init__(self, d_model, kernel=3, dilation=1, dropout=0.1):
        super().__init__()
        self.c1   = CausalConv(d_model, kernel, dilation)
        self.c2   = CausalConv(d_model, kernel, dilation)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):          # x: (B, T, C)
        r = x
        h = x.transpose(1, 2)     # (B, C, T)
        h = F.gelu(self.c1(h))
        h = self.drop(self.c2(h))
        h = h.transpose(1, 2)     # (B, T, C)
        return self.norm(r + h)


class CNNLM(nn.Module):
    """Dilated causal CNN with positional encoding — Experiment 2.4."""
    def __init__(self, vocab_size, d_model=64, n_layers=6,
                 max_len=512, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe    = SinusoidalPE(d_model, max_len, dropout)
        dilations  = [1, 2, 4, 8, 1, 2, 4, 8][:n_layers]
        self.blocks = nn.ModuleList([
            CNNBlock(d_model, kernel=3, dilation=d, dropout=dropout)
            for d in dilations
        ])
        self.ln   = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.pe(self.embed(x))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))


# ═══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, dataset, n_iters=2000, batch=64, lr=1e-3,
          log_every=200, label="model"):
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_iters)
    history = []
    t0 = time.time()
    model.train()
    for i in range(1, n_iters + 1):
        x, y = dataset.get_batch(batch)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if i % log_every == 0:
            history.append({'iter': i, 'loss': round(loss.item(), 4)})
            elapsed = time.time() - t0
            print(f"  [{label}] iter {i:4d}/{n_iters}  loss={loss.item():.4f}"
                  f"  elapsed={elapsed:.1f}s", flush=True)
    return history


# ═══════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════

def plot_group(results_dict, title, filename, out_dir):
    plt.figure(figsize=(8, 5))
    for label, hist in results_dict.items():
        iters  = [h['iter'] for h in hist]
        losses = [h['loss'] for h in hist]
        plt.plot(iters, losses, marker='o', markersize=3, label=label)
    plt.title(title, fontsize=13)
    plt.xlabel('Iteration')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved plot: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def save_results(all_results, param_counts, out_dir):
    """Save intermediate results so progress is never lost."""
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'results': all_results, 'params': param_counts,
                   'cfg': CFG}, f, indent=2, ensure_ascii=False)
    print(f"  [checkpoint] results.json saved.", flush=True)


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = CharDataset(CORPUS, seq_len=CFG['seq_len'])
    V = dataset.vocab_size
    print(f"\nCorpus size: {len(CORPUS):,} chars | vocab: {V}\n", flush=True)

    def make_transformer(**kw):
        return TransformerLM(V, CFG['d_model'], CFG['n_heads'], CFG['d_ff'],
                             CFG['n_layers'], dropout=CFG['dropout'], **kw)

    def run(model, label):
        torch.manual_seed(SEED)
        print(f"\n{'='*60}", flush=True)
        print(f"  Training: {label}  ({count_params(model):,} params)", flush=True)
        print('='*60, flush=True)
        return train(model, dataset,
                     n_iters=CFG['n_iters'], batch=CFG['batch'],
                     lr=CFG['lr'], log_every=CFG['log_every'], label=label)

    all_results = {}
    param_counts = {}

    # ── Experiment 2.1: Positional Encoding ──────────────────────────
    print("\n\n" + "━"*60)
    print("  EXPERIMENT 2.1  Position Encoding Variants")
    print("━"*60)
    exp21 = {}
    for pe in ['sinusoidal', 'learned', 'linear', 'none']:
        m = make_transformer(pe_type=pe)
        label = f"PE={pe}"
        exp21[label] = run(m, label)
        param_counts[label] = count_params(m)
    all_results['exp21'] = exp21
    plot_group(exp21, "Exp 2.1 – Position Encoding Comparison",
               "exp21_position_encoding.png", out_dir)
    save_results(all_results, param_counts, out_dir)

    # ── Experiment 2.2: Q / K / V Necessity ─────────────────────────
    print("\n\n" + "━"*60)
    print("  EXPERIMENT 2.2  Q/K/V Necessity (Merged K=V vs Standard)")
    print("━"*60)
    exp22 = {}
    m_std = make_transformer(attn_cls=MultiHeadAttention)
    exp22["Standard Q,K,V"] = run(m_std, "Standard Q,K,V")
    param_counts["Standard Q,K,V"] = count_params(m_std)

    m_mkv = make_transformer(attn_cls=MergedKVAttention)
    exp22["Merged K=V"]    = run(m_mkv, "Merged K=V")
    param_counts["Merged K=V"] = count_params(m_mkv)
    all_results['exp22'] = exp22
    plot_group(exp22, "Exp 2.2 – Q/K/V Necessity (K=V merged)",
               "exp22_merged_kv.png", out_dir)
    save_results(all_results, param_counts, out_dir)

    # ── Experiment 2.3: Residual Connections ─────────────────────────
    print("\n\n" + "━"*60)
    print("  EXPERIMENT 2.3  Residual (Skip) Connections")
    print("━"*60)
    exp23 = {}
    m_res   = make_transformer(use_residual=True)
    exp23["With Residual"]    = run(m_res,   "With Residual")
    param_counts["With Residual"] = count_params(m_res)

    m_nores = make_transformer(use_residual=False)
    exp23["Without Residual"] = run(m_nores, "Without Residual")
    param_counts["Without Residual"] = count_params(m_nores)
    all_results['exp23'] = exp23
    plot_group(exp23, "Exp 2.3 – Effect of Residual Connections",
               "exp23_residual.png", out_dir)
    save_results(all_results, param_counts, out_dir)

    # ── Experiment 2.4: CNN vs Transformer ───────────────────────────
    print("\n\n" + "━"*60)
    print("  EXPERIMENT 2.4  CNN vs Transformer")
    print("━"*60)
    exp24 = {}
    m_tf  = make_transformer(pe_type='sinusoidal')
    exp24["Transformer"]  = run(m_tf,  "Transformer")
    param_counts["Transformer"] = count_params(m_tf)

    m_cnn = CNNLM(V, CFG['d_model'], n_layers=6, dropout=CFG['dropout'])
    exp24["CNN (dilated)"] = run(m_cnn, "CNN (dilated)")
    param_counts["CNN (dilated)"] = count_params(m_cnn)
    all_results['exp24'] = exp24
    plot_group(exp24, "Exp 2.4 – CNN vs Transformer",
               "exp24_cnn_vs_transformer.png", out_dir)
    save_results(all_results, param_counts, out_dir)

    # ── Experiment 2.5: Gated Head Attention (proposed) ──────────────
    print("\n\n" + "━"*60)
    print("  EXPERIMENT 2.5  Proposed: Content-Dependent Head Gating")
    print("━"*60)
    exp25 = {}
    m_base = make_transformer(pe_type='sinusoidal', attn_cls=MultiHeadAttention)
    exp25["Baseline Transformer"]   = run(m_base, "Baseline Transformer")
    param_counts["Baseline Transformer"] = count_params(m_base)

    m_gate = make_transformer(pe_type='sinusoidal', attn_cls=GatedMultiHeadAttention)
    exp25["Gated-Head Transformer"] = run(m_gate, "Gated-Head Transformer")
    param_counts["Gated-Head Transformer"] = count_params(m_gate)
    all_results['exp25'] = exp25
    plot_group(exp25, "Exp 2.5 – Content-Dependent Head Gating",
               "exp25_gated_attention.png", out_dir)
    save_results(all_results, param_counts, out_dir)

    # ── Save raw results ──────────────────────────────────────────────
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'results': all_results, 'params': param_counts,
                   'cfg': CFG}, f, indent=2, ensure_ascii=False)
    print(f"\n\nResults saved → {json_path}")

    # ── Quick summary ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL LOSS SUMMARY")
    print("="*60)
    for exp_name, exp_data in all_results.items():
        print(f"\n  {exp_name}:")
        for label, hist in exp_data.items():
            final = hist[-1]['loss']
            print(f"    {label:<30s}  final_loss={final:.4f}"
                  f"  params={param_counts.get(label, '?'):,}")

    print("\n  Done. All plots and results saved to:", out_dir)


if __name__ == '__main__':
    main()
