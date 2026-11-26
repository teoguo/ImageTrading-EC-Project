"""Step 7b: pruning the final models.

Step 7a keeps the network and changes the number format (bfloat16, INT8). This step removes parts of the network:

  CNN, unstructured   Magnitude pruning: the smallest weights of each convolution / linear layer are set to zero
                      (same share in every layer, or one global ranking of scale-normalised weights). The file and the
                      dense arithmetic stay the same size; what changes is the number of non-zero weights.
  CNN, structured     Channel pruning: whole filters are removed (ranked by filter L2 norm, or by the first-order
                      Taylor score of Molchanov et al. 2019 on training windows) and a narrower network is rebuilt,
                      so parameters, MACs and latency all fall. It is an ordinary ChartCNN: Step 7a's INT8 code runs on it.
  ViT                 Head pruning (Michel et al. 2019: gradient of the loss w.r.t. a gate on each head, training
                      windows) and token pruning (after attention in chosen blocks, keep the tokens with the most attention).

Every pruned network is fine-tuned with the source run's rule and compared with the unpruned network on identical
validation rows (efficiency.compare: whole days resampled). Pruning decisions use weights or training windows only.

MACs are counted per sample for convolutions, linear layers and the two matrix products of attention
(QK^T and attention x V); normalisation, activations, pooling and unfolding are not counted.
"""
import copy
import io
import json
import math
import re
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss

import efficiency as ef
from chartcnn.vit import Block

__version__ = "2026-11-26.1"

BASE = "unpruned"


# ----------------------------------------------------------------------------- counting

def _attention_width(m):
    """heads x head size of a transformer block (the width of its attention products), else None."""
    if isinstance(m, PrunedBlock):
        return len(m.kept) * m.head_dim
    if isinstance(m, Block):
        return m.qkv.out_features // 3
    return None


def layer_table(model, x, t):
    """One row per counted layer for ONE sample: parameters, non-zero parameters, MACs, and the MACs an ideal
    sparse engine would need (dense MACs x share of non-zero weights). The last row holds the uncounted
    parameters (norms, embeddings), so `parameters` sums to the model total."""
    rows, hooks = [], []

    def hook(name):
        def f(m, inp, out):
            width = _attention_width(m)
            if width is not None:
                n = inp[0].shape[1]
                rows.append(dict(layer=f"{name} attention products", tokens=n, parameters=0, nonzero=0,
                                 macs=2 * n * n * width, macs_sparse=2 * n * n * width))
                return
            w = m.weight
            macs = out.numel() * w[0].numel()                # every output value is one dot product with a filter / row
            nz = int(torch.count_nonzero(w))
            params = sum(p.numel() for p in m.parameters())
            rows.append(dict(layer=f"{name} {m.__class__.__name__}{tuple(w.shape)}", tokens=np.nan, parameters=params,
                             nonzero=nz + (params - w.numel()), macs=macs, macs_sparse=macs * nz / w.numel()))
        return f

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) or _attention_width(m) is not None:
            hooks.append(m.register_forward_hook(hook(name)))
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            p = next(model.parameters())
            model(x[:1].to(p.device).float(), t[:1].to(p.device).float())
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    table = pd.DataFrame(rows)
    total = sum(p.numel() for p in model.parameters())
    rest = total - int(table.parameters.sum())
    if rest:
        other = total_nonzero(model) - int(table.nonzero.sum())
        table = pd.concat([table, pd.DataFrame([dict(layer="other (norms, embeddings)", tokens=np.nan, parameters=rest,
                                                     nonzero=other, macs=0, macs_sparse=0)])], ignore_index=True)
    return table


def total_nonzero(model):
    return int(sum(torch.count_nonzero(p) for p in model.parameters()))


def stats(model, x, t):
    """Size of one network: parameters, non-zero parameters, MACs per sample, bytes of the saved weights
    (as stored, and zlib-compressed: zeros compress, so this is what unstructured pruning saves on disk)."""
    table = layer_table(model, x, t)
    total = sum(p.numel() for p in model.parameters())
    nz = total_nonzero(model)
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return dict(parameters=total, nonzero_parameters=nz, sparsity=1 - nz / total, macs_g=float(table.macs.sum()) / 1e9,
                macs_sparse_g=float(table.macs_sparse.sum()) / 1e9, size_mb=buf.tell() / 2**20,
                size_zlib_mb=len(zlib.compress(buf.getvalue(), 6)) / 2**20)


# ----------------------------------------------------------------------------- data

def palette_of(dataset, n=32):
    """Pixel values of the charts (0, 180 = volume, 255 = price / indicators, plus any found in a sample)."""
    values = {0, 180, 255}
    for i in np.linspace(0, len(dataset) - 1, min(n, len(dataset))).astype(int):
        values.update(np.unique(ef.to_uint8(dataset[int(i)][0].numpy())).tolist())
    return np.array(sorted(values), np.uint8)


class Feed:
    """Batches (x, tab, y, index) on the device, either drawn by DataLoader workers or read from Step 7a's
    lossless bit-packed cache (drawn once; every pruned variant then reuses the same charts)."""

    def __init__(self, dataset, workers=0, cache=False, name="", log=print):
        self.source, self.workers, self.cache = dataset, workers, None
        if cache and len(dataset):
            try:
                self.cache = ef.PackedCharts(dataset, palette_of(dataset), workers=workers)
                log(f"  {name}: {len(dataset):,} charts cached in {self.cache.seconds_to_build:.0f} s, "
                    f"{len(dataset) * self.cache.bytes_per_image() / 2**20:,.0f} MB")
            except AssertionError as e:                      # a pixel value the palette does not have: draw instead
                log(f"  {name}: not cached ({e}); charts are drawn each time")

    def __len__(self):
        return len(self.source)

    def batches(self, batch_size, device, shuffle=False, seed=0):
        cached = self.cache is not None
        for x, t, y, i in ef.loader(self.cache if cached else self.source, batch_size, 0 if cached else self.workers,
                                    shuffle, seed):
            x = x.to(device, non_blocking=True)
            x = self.cache.decode(x) if cached else x.float()
            yield x, t.to(device, non_blocking=True).float(), y.to(device, non_blocking=True), i


@torch.inference_mode()
def predict(model, feed, device, batch_size, amp=False):
    """P(y = 1) for every item of `feed`, in order (float32 unless `amp`)."""
    model.eval()
    out = np.empty(len(feed))
    dtype = ef._amp_dtype(device) if amp else None
    for x, t, _, i in feed.batches(batch_size, device):
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            logits = model(x, t)
        out[i.numpy()] = logits.float().softmax(1)[:, 1].cpu().numpy()
    return out


# ----------------------------------------------------------------------------- unstructured magnitude pruning

def prunable_layers(model, first_conv=False):
    """name -> Conv2d / Linear of the image network (the small feature MLP `tab` is left alone). The first
    convolution reads the picture with very few weights (960 in the chart CNN) and is kept dense by default."""
    layers = {n: m for n, m in model.named_modules()
              if isinstance(m, (nn.Conv2d, nn.Linear)) and n != "tab" and not n.startswith("tab.")}
    convs = [n for n, m in layers.items() if isinstance(m, nn.Conv2d)]
    if convs and not first_conv:
        layers.pop(convs[0])
    return layers


def magnitude_masks(layers, sparsity, scope="layer"):
    """Boolean keep-masks that zero the smallest |weights|.

    scope "layer"   `sparsity` of every layer's weights (a float, or {layer name: sparsity});
    scope "global"  `sparsity` of all listed weights together, ranked by |w| / mean|w| of their own layer
                    (raw magnitudes are not comparable across layers: the head starts at zero and stays small)."""
    if scope == "layer":
        share = sparsity if isinstance(sparsity, dict) else {n: sparsity for n in layers}
        masks = {}
        for n, m in layers.items():
            w = m.weight.detach().abs().flatten()
            drop = int(round(float(share.get(n, 0.0)) * w.numel()))
            mask = torch.ones(w.numel(), dtype=torch.bool)
            if drop:
                mask[torch.topk(w, drop, largest=False).indices.cpu()] = False
            masks[n] = mask.view_as(m.weight)
        return masks
    assert scope == "global" and not isinstance(sparsity, dict)
    score = {n: (m.weight.detach().abs() / m.weight.detach().abs().mean().clamp_min(1e-12)).flatten().cpu()
             for n, m in layers.items()}
    flat = torch.cat(list(score.values()))
    drop = int(round(float(sparsity) * flat.numel()))
    mask = torch.ones(flat.numel(), dtype=torch.bool)
    if drop:
        mask[torch.topk(flat, drop, largest=False).indices] = False
    masks, at = {}, 0
    for n, m in layers.items():
        masks[n] = mask[at:at + m.weight.numel()].view_as(m.weight)
        at += m.weight.numel()
    return masks


@torch.no_grad()
def apply_masks(model, masks):
    """Zero the masked weights in place (called again after every optimizer step while fine-tuning)."""
    modules = dict(model.named_modules())
    for n, mask in masks.items():
        w = modules[n].weight
        w.mul_(mask.to(w.device, w.dtype))


def mask_table(model, masks):
    modules = dict(model.named_modules())
    return pd.DataFrame([dict(layer=n, weights=m.numel(), kept=int(m.sum()), sparsity=1 - float(m.float().mean()),
                              shape=tuple(modules[n].weight.shape)) for n, m in masks.items()])


# ----------------------------------------------------------------------------- structured channel pruning (CNN)

def conv_stack(net):
    """Positions of the convolutions in the plain ChartCNN feature stack (each is followed by its BatchNorm)."""
    f = net.features
    assert isinstance(f, nn.Sequential), "channel pruning needs the plain ChartCNN feature stack"
    idx = [i for i, m in enumerate(f) if isinstance(m, nn.Conv2d)]
    assert idx and all(isinstance(f[i + 1], nn.BatchNorm2d) and f[i].groups == 1 for i in idx), \
        "channel pruning needs Conv2d -> BatchNorm2d blocks (the plain ChartCNN, not the residual one)"
    return idx


def channel_importance(net, criterion="l2", batches=None, device=None):
    """One score per output channel of every convolution.

    "l2"      L2 norm of the filter (Li et al. 2017); needs no data.
    "taylor"  first-order Taylor estimate of the loss change when the channel is removed, taken at its BatchNorm
              (Molchanov et al. 2019): mean over `batches` of (gamma * dL/dgamma + beta * dL/dbeta)^2."""
    if criterion == "l2":
        return [net.features[i].weight.detach().flatten(1).norm(dim=1).cpu() for i in conv_stack(net)]
    assert criterion == "taylor" and batches is not None, "taylor needs training batches"
    m = copy.deepcopy(net).to(device).eval()                 # eval: running BatchNorm statistics, no dropout
    bns = [m.features[i + 1] for i in conv_stack(m)]
    total, n = [torch.zeros(b.num_features) for b in bns], 0
    for x, t, y, _ in batches:
        m.zero_grad(set_to_none=True)
        F.cross_entropy(m(x, t).float(), y).backward()
        for j, b in enumerate(bns):
            total[j] += ((b.weight * b.weight.grad + b.bias * b.bias.grad) ** 2).detach().cpu()
        n += 1
    return [v / max(1, n) for v in total]


@torch.no_grad()
def prune_channels(net, keep, importance):
    """A narrower copy of `net`: each convolution keeps its `keep` share (a float, or one per convolution) of
    output channels with the highest importance; the next convolution loses the matching inputs and the head
    the matching columns. Returns (network, channels kept per convolution)."""
    idx, f = conv_stack(net), net.features
    keep = [float(keep)] * len(idx) if np.isscalar(keep) else [float(k) for k in keep]
    assert len(keep) == len(idx), f"{len(idx)} convolutions, {len(keep)} keep ratios"
    kept = [torch.sort(torch.topk(imp, max(1, int(round(k * len(imp))))).indices).values
            for imp, k in zip(importance, keep)]
    layers, prev, j = [], None, 0
    for m in f:
        if isinstance(m, nn.Conv2d):
            w = m.weight[kept[j]]
            if prev is not None:
                w = w[:, prev]
            conv = nn.Conv2d(w.shape[1], w.shape[0], m.kernel_size, m.stride, m.padding, m.dilation, bias=m.bias is not None)
            conv.weight.copy_(w)
            if m.bias is not None:
                conv.bias.copy_(m.bias[kept[j]])
            layers.append(conv)
        elif isinstance(m, nn.BatchNorm2d):
            sel = kept[j]
            bn = nn.BatchNorm2d(len(sel), eps=m.eps, momentum=m.momentum)
            bn.weight.copy_(m.weight[sel]); bn.bias.copy_(m.bias[sel])
            bn.running_mean.copy_(m.running_mean[sel]); bn.running_var.copy_(m.running_var[sel])
            bn.num_batches_tracked.copy_(m.num_batches_tracked)
            layers.append(bn)
            prev, j = sel, j + 1
        else:
            layers.append(copy.deepcopy(m))
    lin = net.head[-1]
    c_old = len(importance[-1])
    w = lin.weight.view(lin.out_features, c_old, lin.in_features // c_old)[:, prev].reshape(lin.out_features, -1)
    new = nn.Linear(w.shape[1], lin.out_features)
    new.weight.copy_(w); new.bias.copy_(lin.bias)
    head = nn.Sequential(*[copy.deepcopy(m) for m in list(net.head)[:-1]], new)
    return ef.Net(nn.Sequential(*layers), head, copy.deepcopy(net.tab), net.fp32_head).eval(), [len(k) for k in kept]


def reinitialised(net, seed=0):
    """The same architecture with fresh weights, initialised as ChartCNN / IndicatorCNN do (Xavier convolutions,
    zero head and zero last layer of the feature MLP): the "train the small network from scratch" control."""
    m = copy.deepcopy(net)
    torch.manual_seed(seed)
    for mod in m.modules():
        if hasattr(mod, "reset_parameters"):
            mod.reset_parameters()
    for part in (m.features, m.head):
        for mod in part.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
    nn.init.zeros_(m.head[-1].weight)
    if m.tab is not None:
        nn.init.zeros_(m.tab[-1].weight); nn.init.zeros_(m.tab[-1].bias)
    return m


# ----------------------------------------------------------------------------- ViT: heads and tokens

class PrunedBlock(nn.Module):
    """chartcnn.vit.Block with a subset of its attention heads (`kept`) and, optionally, token pruning: after the
    attention update only the class token and the `keep_tokens` share of patch tokens with the highest score go on
    to the MLP and to later blocks. score "received": attention a token receives, averaged over heads and over all
    queries (the scratch ViT pools patch tokens, so its class token is not trained to find the informative ones);
    "cls": attention from the class token (EViT, Liang et al. 2022)."""

    def __init__(self, block, kept=None, keep_tokens=None, score="received"):
        super().__init__()
        d, heads = block.proj.out_features, block.heads
        hd = d // heads
        self.kept = list(range(heads)) if kept is None else sorted(int(h) for h in kept)
        assert self.kept and set(self.kept) <= set(range(heads)), f"heads {kept} of {heads}"
        cols = torch.cat([torch.arange(h * hd, (h + 1) * hd) for h in self.kept])
        rows = torch.cat([cols + part * d for part in range(3)])            # q, k, v rows of the kept heads
        self.n1, self.n2, self.mlp = copy.deepcopy(block.n1), copy.deepcopy(block.n2), copy.deepcopy(block.mlp)
        self.qkv, self.proj = nn.Linear(d, len(rows)), nn.Linear(len(cols), d)
        with torch.no_grad():
            self.qkv.weight.copy_(block.qkv.weight[rows]); self.qkv.bias.copy_(block.qkv.bias[rows])
            self.proj.weight.copy_(block.proj.weight[:, cols]); self.proj.bias.copy_(block.proj.bias)
        self.head_dim, self.drop_path = hd, block.drop_path
        self.keep_tokens, self.score = keep_tokens, score
        self.gate = None                                     # per-head gate, set only to measure head importance

    def _drop(self, x):
        if not self.training or self.drop_path == 0:
            return x
        keep = torch.rand(x.shape[0], 1, 1, device=x.device) >= self.drop_path
        return x * keep / (1 - self.drop_path)

    def forward(self, x):
        b, n, d = x.shape
        h = len(self.kept)
        q, k, v = self.qkv(self.n1(x)).view(b, n, 3, h, self.head_dim).permute(2, 0, 3, 1, 4)
        w = None
        if self.keep_tokens is None:
            a = F.scaled_dot_product_attention(q, k, v)
        else:                                                # the same attention, written out to read its weights
            w = ((q @ k.transpose(-2, -1)) * self.head_dim ** -0.5).softmax(-1)
            a = w @ v
        if self.gate is not None:
            a = a * self.gate.view(1, -1, 1, 1)
        x = x + self._drop(self.proj(a.transpose(1, 2).reshape(b, n, h * self.head_dim)))
        if w is not None and n > 2:
            s = w[:, :, 0, 1:].mean(1) if self.score == "cls" else w[:, :, :, 1:].mean((1, 2))
            top = s.float().topk(max(1, math.ceil(self.keep_tokens * (n - 1))), dim=1).indices.sort(1).values + 1
            idx = torch.cat([torch.zeros(b, 1, dtype=top.dtype, device=top.device), top], 1)
            x = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, d))
        return x + self._drop(self.mlp(self.n2(x)))


def vit_blocks(net):
    """Positions of the backbone blocks in `net.features` (the tokenizer's own block is inside the tokenizer)."""
    return [i for i, m in enumerate(net.features) if isinstance(m, (Block, PrunedBlock))]


def prune_vit(net, heads=None, tokens=None, score="received"):
    """A copy of a ChartViT network with `heads[l]` = heads kept in backbone block l (None: all) and
    `tokens[l]` = share of patch tokens kept after block l's attention (blocks not listed keep all)."""
    m = copy.deepcopy(net)
    for l, i in enumerate(vit_blocks(m)):
        assert isinstance(m.features[i], Block), "prune_vit starts from the unpruned network"
        m.features[i] = PrunedBlock(m.features[i], None if heads is None else heads[l],
                                    None if tokens is None else tokens.get(l), score)
    return m.eval()


def head_importance(net, batches, device):
    """[blocks, heads] importance of every backbone head: mean |dLoss / d gate| with a gate of 1 on each head's
    output, on training batches, L2-normalised within each block (Michel et al. 2019)."""
    m = prune_vit(net).to(device).eval()
    blocks = [m.features[i] for i in vit_blocks(m)]
    for p in m.parameters():
        p.requires_grad_(False)
    for b in blocks:
        b.gate = torch.ones(len(b.kept), device=device, requires_grad=True)
    total, n = torch.zeros(len(blocks), len(blocks[0].kept)), 0
    for x, t, y, _ in batches:
        grads = torch.autograd.grad(F.cross_entropy(m(x, t).float(), y), [b.gate for b in blocks])
        total += torch.stack([g.abs().cpu() for g in grads])
        n += 1
    imp = total / max(1, n)
    return imp / imp.norm(dim=1, keepdim=True).clamp_min(1e-12)


def select_heads(importance, n_keep):
    """{block: heads kept}: the `n_keep` most important heads overall, with at least one head in every block."""
    blocks, heads = importance.shape
    assert blocks <= n_keep <= blocks * heads, f"keep between {blocks} and {blocks * heads} heads"
    kept = {l: [int(importance[l].argmax())] for l in range(blocks)}
    left = n_keep - blocks
    for flat in torch.argsort(importance.flatten(), descending=True).tolist():
        l, h = divmod(flat, heads)
        if left and h not in kept[l]:
            kept[l].append(h); left -= 1
    return {l: sorted(v) for l, v in kept.items()}


# ----------------------------------------------------------------------------- fine-tuning

def finetune_settings(src, epochs, patience, eval_batch_size, min_delta=1e-5, lr_scale=1.0, warmup_steps=None,
                      batch_size=None, amp=True):
    """The source run's own training rule (optimizer, learning rates, weight decay, clipping), for a few epochs."""
    s = src.settings
    if src.kind == "grid":
        from chartcnn import grid
        a = grid.arch_settings(s, s["arch"])
        out = dict(optimizer=a.get("optimizer", "adam"), lr=a["learning_rate"], tab_lr=None, weight_decay=a["weight_decay"],
                   warmup_steps=a.get("warmup_steps", 0), grad_clip=a.get("grad_clip"))
    else:
        out = dict(optimizer="adam", lr=s["learning_rate"], tab_lr=s.get("tab_learning_rate"), weight_decay=s["weight_decay"],
                   warmup_steps=0, grad_clip=None)
    out["lr"] = out["lr"] * lr_scale
    if warmup_steps is not None:
        out["warmup_steps"] = warmup_steps
    out.update(epochs=epochs, patience=patience, min_delta=min_delta, batch_size=batch_size or s["batch_size"],
               eval_batch_size=eval_batch_size, seed=s["seed"], amp=amp)
    return out


def finetune(model, train, es, y_es, s, device, masks=None, log=print):
    """Train `model` on the feed `train`; keep the weights with the lowest log loss on the early-stopping feed `es`
    (the state before any training counts as epoch 0, so fine-tuning cannot end worse than it started on those
    days). With `masks`, pruned weights are zeroed again after every step. Returns (history, best epoch, minutes)."""
    model.to(device)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    body = [p for n, p in named if not n.startswith("tab.")]
    tab = [p for n, p in named if n.startswith("tab.")]
    groups = [dict(params=body, lr=s["lr"])] + ([dict(params=tab, lr=s["tab_lr"] or s["lr"])] if tab else [])
    base_lr = [g["lr"] for g in groups]
    opt = (torch.optim.AdamW if s["optimizer"] == "adamw" else torch.optim.Adam)(groups, weight_decay=s["weight_decay"])
    dtype = ef._amp_dtype(device) if s["amp"] else None
    params = body + tab

    def check():
        p = predict(model, es, device, s["eval_batch_size"], s["amp"])
        return log_loss(y_es, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]), ef._auc(y_es, p)

    def snapshot():
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    t_all = time.perf_counter()
    ll, auc = check()
    history = [dict(epoch=0, train_loss=np.nan, train_seconds=0.0, images_per_s=np.nan, es_log_loss=ll, es_auc=auc)]
    best, best_state, best_epoch, bad, step = ll, snapshot(), 0, 0, 0
    torch.manual_seed(s["seed"])
    for epoch in range(1, s["epochs"] + 1):
        model.train()
        seen, loss_sum, start = 0, 0.0, time.perf_counter()
        for x, t, y, _ in train.batches(s["batch_size"], device, True, s["seed"] + epoch):
            if s["warmup_steps"]:
                for g, lr in zip(opt.param_groups, base_lr):
                    g["lr"] = lr * min(1.0, (step + 1) / s["warmup_steps"])
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                logits = model(x, t)
            loss = F.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if s["grad_clip"]:
                nn.utils.clip_grad_norm_(params, s["grad_clip"])
            opt.step()
            if masks:
                apply_masks(model, masks)
            seen += len(y); loss_sum += loss.item() * len(y); step += 1
        seconds = time.perf_counter() - start
        ll, auc = check()
        history.append(dict(epoch=epoch, train_loss=loss_sum / max(1, seen), train_seconds=seconds,
                            images_per_s=seen / seconds, es_log_loss=ll, es_auc=auc))
        log(f"    epoch {epoch}: train loss {loss_sum / max(1, seen):.4f} | early-stop days AUC {auc:.4f}, log loss {ll:.5f} "
            f"(start {history[0]['es_log_loss']:.5f}) | {seen / seconds:,.0f} img/s")
        if ll < best:
            best_state, best_epoch = snapshot(), epoch
        bad = 0 if ll < best - s["min_delta"] else bad + 1
        best = min(best, ll)
        if bad >= s["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    return pd.DataFrame(history), best_epoch, (time.perf_counter() - t_all) / 60


# ----------------------------------------------------------------------------- one source model and its variants

def _slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


class Study:
    """One trained source network (efficiency.Source) on fixed rows.

    valid   the source run's validation rows on every `valid_every_kth_day`-th day (all rows by default)
    es      those of every `es_every_kth_day`-th of these days: early stopping while fine-tuning
    train   at most `max_train` of the source run's training rows: fine-tuning
    scan    `scan_rows` other training rows: sensitivity scans and importance scores (never validation rows)

    Each variant is scored on `valid` in float32 and saved (scores, sizes, the network itself), so a rerun with
    the same RUN_ID skips it."""

    def __init__(self, src, tag, cfg, device, run_dir, workers=0, log=print):
        self.src, self.tag, self.cfg, self.device, self.log = src, tag, cfg, device, log
        self.dir = Path(run_dir) / tag
        (self.dir / "variants").mkdir(parents=True, exist_ok=True)
        seed = cfg["seed"]
        self.valid_pos = ef.every_kth_day(src.valid, cfg["valid_every_kth_day"])
        self.valid = src.valid.iloc[self.valid_pos].reset_index(drop=True)
        days = np.sort(self.valid.decision_date.unique())[::max(1, cfg["es_every_kth_day"])]
        self.es = self.valid[self.valid.decision_date.isin(days)].reset_index(drop=True)
        self.train = src.train.sample(min(cfg["max_train"], len(src.train)), random_state=seed).reset_index(drop=True)
        self.scan = src.train.sample(min(cfg["scan_rows"], len(src.train)), random_state=seed + 1).reset_index(drop=True)
        log(f"{tag}: {src.label} | validation {len(self.valid):,} rows on {self.valid.decision_date.nunique()} days | "
            f"early stopping {len(self.es):,} rows on {len(days)} days | fine-tuning {len(self.train):,} training rows | "
            f"scans {len(self.scan):,} training rows")
        cache = cfg.get("cache", {})
        mk = lambda frame, split, name: Feed(src.dataset(frame, split), workers, cache.get(name, False), f"{tag} {name}", log)
        self.feeds = dict(valid=mk(self.valid, "valid", "valid"), es=mk(self.es, "valid", "es"),
                          train=mk(self.train, "train", "train"), scan=mk(self.scan, "train", "scan"))
        x, t, _, _ = self.feeds["valid"].source[0]
        self.x0, self.t0 = x[None].float(), t[None].float()
        self.scores, self.info = {}, {}

    def release(self):
        """Drop the cached charts (the feeds then draw again if they are used)."""
        for f in self.feeds.values():
            f.cache = None

    # ---- scoring and bookkeeping
    def auc(self, model, feed="scan"):
        frame = dict(scan=self.scan, es=self.es, valid=self.valid)[feed]
        p = predict(model.to(self.device), self.feeds[feed], self.device, self.cfg["eval_batch"])
        return ef._auc(frame.label.to_numpy(), p)

    def scan_batches(self, n_batches, batch_size):
        for k, b in enumerate(self.feeds["scan"].batches(batch_size, self.device)):
            if k >= n_batches:
                break
            yield b

    def model(self, name):
        """The saved network of a variant (CPU, eval)."""
        if name == BASE:
            return copy.deepcopy(self.src.model).cpu().eval()
        return torch.load(self.dir / "variants" / _slug(name) / "model.pt", map_location="cpu", weights_only=False).eval()

    def baseline(self):
        """Scores of the unpruned network on `valid`, checked against the source run's saved scores."""
        path = self.dir / "variants" / f"{BASE}.npy"
        if path.exists():
            p = np.load(path)
        else:
            p = predict(copy.deepcopy(self.src.model).to(self.device), self.feeds["valid"], self.device, self.cfg["eval_batch"])
            np.save(path, p)
        saved = self.src.saved[self.valid_pos]
        self.log(f"{self.tag}: reproduction of the source run's saved scores: corr {np.corrcoef(p, saved)[0, 1]:.6f}, "
                 f"max |diff| {np.abs(p - saved).max():.2e} | AUC {ef._auc(self.valid.label.to_numpy(), p):.4f}")
        self.scores[BASE] = p
        self.info[BASE] = dict(family="reference", note="the source run's network", finetune_best_epoch=np.nan,
                               finetune_minutes=0.0, **stats(self.src.model, self.x0, self.t0))
        return p

    def variant(self, name, family, build, tune=None, note=""):
        """Build (`build()` -> dict(model=..., masks=None, **extra)), optionally fine-tune (`tune` = settings of
        `finetune`), score on `valid`, record sizes. Skipped when already saved."""
        folder = self.dir / "variants" / _slug(name)
        if (folder / "variant.json").exists() and (folder / "valid_prob.npy").exists():
            self.scores[name] = np.load(folder / "valid_prob.npy")
            self.info[name] = json.loads((folder / "variant.json").read_text())
            self.log(f"  {name}: already done, loaded")
            return self.info[name]
        folder.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        made = build()
        model, masks = made.pop("model"), made.pop("masks", None)
        info = dict(family=family, note=note, finetune_best_epoch=np.nan, finetune_minutes=0.0, **made)
        if tune is not None:
            self.log(f"  {name}: fine-tuning (at most {tune['epochs']} epochs)")
            history, best_epoch, minutes = finetune(model, self.feeds["train"], self.feeds["es"], self.es.label.to_numpy(),
                                                    tune, self.device, masks, self.log)
            history.to_csv(folder / "history.csv", index=False)
            info.update(finetune_best_epoch=best_epoch, finetune_minutes=minutes, es_auc_start=float(history.es_auc.iloc[0]),
                        es_auc_best=float(history.es_auc[history.epoch == best_epoch].iloc[0]))
        p = predict(model.to(self.device), self.feeds["valid"], self.device, self.cfg["eval_batch"])
        model = model.cpu().eval()
        info.update(stats(model, self.x0, self.t0))
        info["minutes"] = (time.perf_counter() - t0) / 60
        torch.save(model, folder / "model.pt")
        np.save(folder / "valid_prob.npy", p)
        (folder / "variant.json").write_text(json.dumps(info, indent=2, default=float))
        self.scores[name], self.info[name] = p, info
        self.log(f"  {name}: AUC {ef._auc(self.valid.label.to_numpy(), p):.4f} (unpruned "
                 f"{ef._auc(self.valid.label.to_numpy(), self.scores[BASE]):.4f}) | {info['nonzero_parameters']:,} non-zero "
                 f"parameters | {info['macs_g']:.3f} GMACs | {info['minutes']:.1f} min")
        return info

    # ---- tables
    def table(self, n_boot=300, seed=0, margin=0.002):
        """efficiency.compare against the unpruned network (same rows, whole days resampled) joined with sizes.
        `no_loss`: the 95% interval of the AUC difference lies above -`margin`."""
        cmp = ef.compare(self.valid, self.scores, BASE, n_boot, seed).set_index("model")
        base = self.info[BASE]
        rows = []
        for name, info in self.info.items():
            r = cmp.loc[name]
            rows.append(dict(
                source=self.tag, model=self.src.label, variant=name, family=info["family"], auc=r.auc, auc_low=r.auc_low,
                auc_high=r.auc_high, diff_vs_fp32=r.diff_vs_ref, diff_low=r.diff_low, diff_high=r.diff_high, verdict=r.verdict,
                no_loss=bool(r.diff_low > -margin), score_corr_vs_ref=r.score_corr_vs_ref, auc_rows=int(r.rows), days=int(r.days),
                parameters=info["parameters"], nonzero_parameters=info["nonzero_parameters"], sparsity=info["sparsity"],
                compression_ratio=base["parameters"] / max(1, info["nonzero_parameters"]),
                macs_g=info["macs_g"], macs_reduction=base["macs_g"] / info["macs_g"],
                macs_sparse_g=info["macs_sparse_g"], size_mb=info["size_mb"], size_zlib_mb=info["size_zlib_mb"],
                finetune_best_epoch=info.get("finetune_best_epoch", np.nan), finetune_minutes=info.get("finetune_minutes", 0.0),
                note=info.get("note", "")))
        return pd.DataFrame(rows)

    def paired(self, pairs, n_boot=300, seed=0, extra=None):
        """AUC(a) - AUC(b) with a 95% paired day-bootstrap interval for each (a, b, what) of `pairs`;
        `extra`: more scores on the same rows (e.g. the source run's saved reference models)."""
        scores = dict(self.scores, **(extra or {}))
        boot = ef.day_boot(self.valid, scores, n_boot, seed)
        y = self.valid.label.to_numpy()
        rows = []
        for a, b, what in pairs:
            if a in scores and b in scores:
                d = boot[a] - boot[b]
                lo, hi = d.quantile(0.025), d.quantile(0.975)
                rows.append(dict(source=self.tag, check=what, a=a, b=b, diff=ef._auc(y, scores[a]) - ef._auc(y, scores[b]),
                                 low=lo, high=hi, verdict="better" if lo > 0 else "worse" if hi < 0 else "tied"))
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- sensitivity scans (training rows)

def weight_sensitivity(study, sparsities, first_conv=True):
    """AUC on the scan rows when ONE layer alone is magnitude-pruned, without fine-tuning."""
    layers = prunable_layers(study.src.model, first_conv=first_conv)
    base = study.auc(copy.deepcopy(study.src.model))
    rows = []
    for name in layers:
        for s in sparsities:
            m = copy.deepcopy(study.src.model)
            apply_masks(m, magnitude_masks({name: dict(m.named_modules())[name]}, s))
            a = study.auc(m)
            rows.append(dict(layer=name, weights=layers[name].weight.numel(), sparsity=s, auc=a, auc_change=a - base))
    return pd.DataFrame(rows), base


def channel_sensitivity(study, keeps, importance):
    """AUC on the scan rows when ONE convolution alone loses channels, without fine-tuning."""
    n = len(conv_stack(study.src.model))
    base = study.auc(copy.deepcopy(study.src.model))
    rows = []
    for j in range(n):
        for k in keeps:
            m, channels = prune_channels(study.src.model, [k if i == j else 1.0 for i in range(n)], importance)
            a = study.auc(m)
            rows.append(dict(conv=j + 1, keep=k, channels="-".join(map(str, channels)), auc=a, auc_change=a - base))
    return pd.DataFrame(rows), base
