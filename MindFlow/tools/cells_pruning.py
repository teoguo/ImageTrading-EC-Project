"""Cells of MindFlow Step 7b (pruning)."""
from cells import code, header, md, setup


def pruning_cells():
    c = [header("7b", "Pruning", "Step 7a keeps the network and changes the number format. How much of the network "
                "itself can be removed (weights, channels, attention heads, tokens) before validation AUC falls, and "
                "what does each kind of pruning save in parameters, MACs, file size and latency?", """
**Run this after Step 6b (final model), Step 4a and Step 3a**: their trained networks are loaded from their best
checkpoints and scored on their own validation rows (2023). The test years are not used.

| part | network | what is removed | what it saves |
|---|---|---|---|
| unstructured | final hybrid CNN; image-only CNN (`ewsd`) | the smallest weights of conv2, conv3 and the head (50–95%) | non-zero parameters, compressed file size; dense MACs and latency do not change |
| structured | the same two CNNs | whole filters (L2 norm; one Taylor-ranked run), a narrower CNN is rebuilt | parameters, MACs, latency; Step 7a's INT8 code still runs on it |
| ViT | scratch T2T-ViT (`vit`) | attention heads of the 8 backbone blocks; patch tokens after blocks 3, 5, 7 | MACs of the backbone |

**Why two CNNs.** In Step 4a the 38 numbers alone (a 6,786-parameter MLP) reached the hybrid CNN's AUC at 5 minutes,
so the final model can lose most of its CNN without losing AUC. The image-only CNN shows what pruning costs when the
picture is the only input; the features-only score is drawn as the floor of the final model.

**Rules.** Pruning decisions use weights or *training* windows only (sensitivity scans, Taylor and head-importance
scores). Every pruned network is fine-tuned with its source run's rule (same optimizer and learning rates, early
stopping on every 4th validation day) and compared with the unpruned network on identical rows; differences are
paired by resampling whole days (`efficiency.compare`). MACs are per sample: convolutions, linear layers and the two
attention products; normalisation, activations and pooling are not counted.
""")]
    c += setup("""
import copy
import chartcnn
import efficiency as ef
import pruning as pr
print("pruning", pr.__version__, "| efficiency", ef.__version__)
torch.backends.cudnn.benchmark = True
""")
    c += [md("""
## 2. Configuration
**`CNN_SOURCES` / `VIT_SOURCE`**: the networks to prune (`run_id=None` = newest run of that step; `model=None` for
Step 6b = the model written to `final_choice.json`). Set `VIT_SOURCE = None` to skip the ViT part.
**To resume after a disconnect**, set `RUN_ID` to the printed id and Run all: finished variants are loaded.

**`QUICK = True`** is a first pass: 40,000 fine-tuning windows, 1 epoch, every 5th validation day, fewer pruning levels.

**Memory.** Charts are drawn once and kept bit-packed in RAM (Step 7a's cache): about 19 KB per chart, so a full run
holds ~9 GB per CNN source (300,000 training + ~200,000 early-stopping charts); use High-RAM, or set
`ROWS["cache"]` entries to `False` to draw every time instead.

**Time (estimate from Step 4a's L4 figures, not measured):** QUICK about 1 hour; full run about 7–8 hours for
`vol_h1` (validation scoring of ~800,000 freshly drawn charts per variant is the largest part).
"""), code("""
SMOKE = LOCAL_TEST                                # tiny code check on synthetic data; never a result
assert DEVICE.type == "cuda" or SMOKE, "Select a GPU runtime (L4 or A100)"
RUN_ID = None                                     # None = new run; or an old RUN_ID to resume it
QUICK = False
CNN_SOURCES = {"final": dict(step="6b", run_id=None, cell="vol_h1", model=None),          # the final hybrid CNN
               "image_only": dict(step="4a", run_id=None, cell="vol_h1", model="ewsd")}   # the picture alone
VIT_SOURCE = dict(step="3a", run_id=None, cell="vol_h1", model="vit")                     # scratch T2T-ViT; None = skip
WORKERS = 0 if SMOKE else min(12, os.cpu_count() or 2)
ROWS = dict(valid_every_kth_day=1, es_every_kth_day=4, max_train=300_000, scan_rows=20_000, eval_batch=1024, seed=69,
            cache=dict(train=True, es=True, scan=True, valid=False))
FINETUNE = dict(epochs=3, patience=1, vit_lr_scale=0.1, vit_warmup_steps=100)
UNSTRUCTURED = dict(sparsities=[0.5, 0.7, 0.8, 0.9, 0.95], scope="layer")     # "layer": same share in every layer
STRUCTURED = dict(configs=[dict(keep=0.75), dict(keep=0.5), dict(keep=0.5, criterion="taylor"), dict(keep=0.375),
                           dict(keep=0.25), dict(keep=[0.75, 0.5, 0.375])],     # a list = one ratio per convolution
                  taylor_batches=32, scratch_keep=0.5)      # scratch_keep: also train that narrow CNN from scratch
SCAN = dict(sparsities=[0.5, 0.7, 0.9, 0.95], keeps=[0.75, 0.5, 0.25])
VIT = dict(heads=[24, 16, 8], tokens=[dict(after_blocks=[3, 5, 7], keep=0.7), dict(after_blocks=[3, 5, 7], keep=0.5)],
           combined=dict(heads=16, after_blocks=[3, 5, 7], keep=0.7), importance_batches=32, score="received")
BENCH = dict(gpu_batches=[1, 128, 1024], cpu_batches=[1, 64], reps=20, int8_keep=0.5, calibration_windows=2_048,
             calib_batch=128, cpu_every_kth_day=8, cpu_max_rows=30_000)
STATS = dict(n_boot=300, margin=0.002)            # "no loss": 95% interval of the AUC difference above -margin
if QUICK:
    ROWS.update(valid_every_kth_day=5, max_train=40_000, scan_rows=8_000, cache=dict(train=True, es=True, scan=True, valid=True))
    FINETUNE.update(epochs=1)
    UNSTRUCTURED.update(sparsities=[0.5, 0.8, 0.95])
    STRUCTURED.update(configs=[dict(keep=0.5), dict(keep=0.25)], scratch_keep=None)
    SCAN.update(sparsities=[0.7, 0.95], keeps=[0.5])
    VIT.update(heads=[16], tokens=[dict(after_blocks=[3, 5, 7], keep=0.7)], combined=None)
    BENCH.update(reps=10)
if SMOKE:
    ROWS.update(es_every_kth_day=1, max_train=96, scan_rows=48, eval_batch=32, cache=dict(train=True, es=True, scan=True, valid=True))
    FINETUNE.update(epochs=1, vit_warmup_steps=2)
    UNSTRUCTURED.update(sparsities=[0.5, 0.9])
    STRUCTURED.update(configs=[dict(keep=0.5), dict(keep=0.5, criterion="taylor"), dict(keep=[1.0, 0.5, 0.25])],
                      taylor_batches=2, scratch_keep=0.5)
    SCAN.update(sparsities=[0.9], keeps=[0.5])
    VIT.update(heads=[16], tokens=[dict(after_blocks=[3, 5, 7], keep=0.5)], importance_batches=2)
    BENCH.update(gpu_batches=[1, 8], cpu_batches=[1, 8], reps=3, calibration_windows=32, calib_batch=16, cpu_every_kth_day=1,
                 cpu_max_rows=200)
    STATS.update(n_boot=30)
RUN_ID, RUN_DIR, RESUME = mf.new_run(HERE, "7b", RUN_ID, "pruning", LOCAL_TEST)
mf.save_settings(RUN_DIR, RESUME, cnn_sources=CNN_SOURCES, vit_source=VIT_SOURCE, rows=ROWS, finetune=FINETUNE,
                 unstructured=UNSTRUCTURED, structured=STRUCTURED, scan=SCAN, vit=VIT, bench=BENCH, stats=STATS, quick=QUICK,
                 pruning=pr.__version__, efficiency=ef.__version__, torch=torch.__version__,
                 gpu=torch.cuda.get_device_name() if DEVICE.type == "cuda" else "cpu", cpu_threads=torch.get_num_threads())
print("RUN_ID:", RUN_ID, "->", RUN_DIR)
"""), md("""
## 3. The CNNs: rows, unpruned scores, where the parameters and MACs are
Each network is loaded from its best checkpoint with the source run's own rows (checked row by row). The first check
is that this notebook reproduces the source run's saved validation scores. The layer table shows what there is to prune.
"""), code("""
store = chartcnn.Store(PREPARED)
RVT = None


def load_source(spec):
    global RVT
    run = mf.latest_run(HERE, spec["step"], spec["run_id"])
    assert run is not None, f"no finished Step {spec['step']} run"
    if spec["step"] in ("2a", "3a"):
        return ef.load_grid_source(PREPARED, run, spec["cell"], spec["model"], smoke=SMOKE), run
    if spec["step"] == "6b":                          # the final model: 38 numbers + multi-day RV
        import final_model as fm, regime as rg
        fm.register()
        if RVT is None:
            RVT = rg.rv_tables(store, 10 if SMOKE else 60)
        name = spec["model"] or json.loads((run / "final_choice.json").read_text())["final"]
        return ef.load_indicator_source(PREPARED, store, run, spec["cell"], name, tab_fn=lambda st, w: fm.numbers(st, w, RVT)[1]), run
    return ef.load_indicator_source(PREPARED, store, run, spec["cell"], spec["model"]), run


def show_layers(study):
    t = pr.layer_table(study.src.model, study.x0, study.t0)
    t["parameter_share"], t["macs_share"] = t.parameters / t.parameters.sum(), t.macs / t.macs.sum()
    t.to_csv(study.dir / "layers_unpruned.csv", index=False)
    b = study.info[pr.BASE]
    print(f"{study.tag}: {b['parameters']:,} parameters, {b['macs_g']:.3f} GMACs per sample, {b['size_mb']:.2f} MB")
    display(t.drop(columns=["tokens"] if t.tokens.isna().all() else []).round(4))


CNN = {}
for tag, spec in CNN_SOURCES.items():
    src, run = load_source(spec)
    print(f"--- {tag}: Step {spec['step']} run {run.name}, cell {spec['cell']}, model {src.name}")
    CNN[tag] = pr.Study(src, tag, ROWS, DEVICE, RUN_DIR, WORKERS)
    CNN[tag].baseline()
    show_layers(CNN[tag])
"""), md("""
## 4. Sensitivity scan (training rows, no fine-tuning)
One layer at a time: magnitude-prune only that layer, or remove channels only from that convolution, and score a
sample of *training* windows. The layers that tolerate the most pruning are where the non-uniform channel
configuration in `STRUCTURED["configs"]` should cut deepest; edit that list after reading these tables.
"""), code("""
for tag, st in CNN.items():
    wp, cp = st.dir / "sensitivity_weights.csv", st.dir / "sensitivity_channels.csv"
    if wp.exists() and cp.exists():
        W, C = pd.read_csv(wp), pd.read_csv(cp)
    else:
        W, base = pr.weight_sensitivity(st, SCAN["sparsities"])
        C, _ = pr.channel_sensitivity(st, SCAN["keeps"], pr.channel_importance(st.src.model, "l2"))
        W["auc_unpruned"], C["auc_unpruned"] = base, base
        W.to_csv(wp, index=False); C.to_csv(cp, index=False)
    print(f"{tag}: AUC change on {len(st.scan):,} training rows (unpruned {W.auc_unpruned.iloc[0]:.4f})")
    display(W.pivot_table(index=["layer", "weights"], columns="sparsity", values="auc_change").round(4))
    display(C.pivot_table(index="conv", columns="keep", values="auc_change").round(4))
"""), md("""
## 5. Unstructured magnitude pruning
The smallest weights of conv2, conv3 and the head are set to zero, the same share in every layer
(`UNSTRUCTURED["scope"] = "global"` ranks scale-normalised weights across layers instead). Each level is scored
as pruned and after fine-tuning with the zeros held fixed. The first convolution (960 weights) and the feature MLP are
left dense.
"""), code("""
for tag, st in CNN.items():
    print(f"--- {tag}")
    tune = pr.finetune_settings(st.src, FINETUNE["epochs"], FINETUNE["patience"], ROWS["eval_batch"])
    for s in UNSTRUCTURED["sparsities"]:
        def build(s=s, st=st):
            m = copy.deepcopy(st.src.model)
            masks = pr.magnitude_masks(pr.prunable_layers(m), s, UNSTRUCTURED["scope"])
            pr.apply_masks(m, masks)
            return dict(model=m, masks=masks, target_sparsity=s)
        st.variant(f"magnitude {s:.0%}, no fine-tuning", "unstructured, one-shot", build)
        st.variant(f"magnitude {s:.0%}, fine-tuned", "unstructured, fine-tuned", build, tune)
"""), md("""
## 6. Structured channel pruning
Filters are ranked inside each convolution (L2 norm of the filter; `criterion="taylor"` uses the first-order Taylor
score on training windows) and the lowest are removed. The result is a narrower ChartCNN with the surviving weights,
then fine-tuned. `scratch_keep` also trains that narrow architecture from fresh weights with the source run's rule:
if it does as well, the pruned weights carried nothing the small network could not learn itself (Liu et al. 2019).
The validation AUC before fine-tuning is in each variant's `history.csv` (epoch 0, early-stopping days).
"""), code("""
for tag, st in CNN.items():
    print(f"--- {tag}")
    tune = pr.finetune_settings(st.src, FINETUNE["epochs"], FINETUNE["patience"], ROWS["eval_batch"])
    IMP = {"l2": pr.channel_importance(st.src.model, "l2")}
    for cfg in STRUCTURED["configs"]:
        crit = cfg.get("criterion", "l2")
        if crit not in IMP:
            IMP[crit] = pr.channel_importance(st.src.model, crit, st.scan_batches(STRUCTURED["taylor_batches"], tune["batch_size"]), DEVICE)
        narrow, ch = pr.prune_channels(st.src.model, cfg["keep"], IMP[crit])
        st.variant(f"channels {'-'.join(map(str, ch))} ({crit}), fine-tuned", "structured, fine-tuned",
                   lambda narrow=narrow, ch=ch, crit=crit, cfg=cfg: dict(model=copy.deepcopy(narrow), channels=ch, criterion=crit, keep=cfg["keep"]),
                   tune)
    if STRUCTURED["scratch_keep"]:
        narrow, ch = pr.prune_channels(st.src.model, STRUCTURED["scratch_keep"], IMP["l2"])
        s = st.src.settings
        rule = pr.finetune_settings(st.src, s["epochs"], s["patience"], ROWS["eval_batch"])
        st.variant(f"channels {'-'.join(map(str, ch))}, trained from scratch", "structured, from scratch",
                   lambda narrow=narrow, ch=ch, s=s: dict(model=pr.reinitialised(narrow, s["seed"]), channels=ch, keep=STRUCTURED["scratch_keep"]),
                   rule)
"""), md("""
### Latency, and INT8 after pruning
A fixed batch of real charts (drawing excluded), as in Step 7a. Only channel pruning changes the dense network, so
only those variants are timed; magnitude-pruned networks run exactly as fast as the unpruned one on this hardware.
One narrow CNN (`BENCH["int8_keep"]`) is also quantized with Step 7a's post-training static INT8 and scored on the
same CPU rows as the unpruned INT8 network: do the two savings add up?
"""), code("""
CPU = torch.device("cpu")
BITS16 = f"{'bf16' if ef._amp_dtype(DEVICE) == torch.bfloat16 else 'fp16'} GPU"


def fixed_batch(st):
    n = max(BENCH["gpu_batches"] + BENCH["cpu_batches"])
    X, T = [], []
    for x, t, _, _ in ef.loader(st.src.dataset(st.valid.iloc[:n].reset_index(drop=True), "valid"), n, 0):
        X.append(x); T.append(t)
    return torch.cat(X).float(), torch.cat(T).float()


def latency_rows(st, names):
    X, T = fixed_batch(st)
    rows = []
    for name in names:
        m = st.model(name)
        forms = {"fp32 CPU": (m, CPU, "fp32")}
        if DEVICE.type == "cuda":
            g = copy.deepcopy(m).to(DEVICE)
            forms = {"fp32 GPU": (g, DEVICE, "fp32"), BITS16: (g, DEVICE, "bf16"), **forms}
        for form, (mm, dev, prec) in forms.items():
            for r in ef.benchmark(form, mm, dev, prec, X, T, BENCH["gpu_batches"] if dev.type == "cuda" else BENCH["cpu_batches"], BENCH["reps"]):
                rows.append(dict(source=st.tag, variant=name, precision=r.pop("variant"), **r))
    return rows


LATENCY, INT8 = [], []
for tag, st in CNN.items():
    timed = [pr.BASE] + [n for n, i in st.info.items() if i["family"] == "structured, fine-tuned"]
    LATENCY += latency_rows(st, timed)
    pick = [n for n, i in st.info.items() if i["family"] == "structured, fine-tuned" and i.get("criterion") == "l2"
            and i.get("keep") == BENCH["int8_keep"]]
    if tag != "final" or not pick:
        continue
    try:                                              # Step 7a's static PTQ, on the unpruned and the narrow CNN
        calib = ef.calibration_batches(st.src, BENCH["calibration_windows"], BENCH["calib_batch"], ROWS["seed"])
        pos = ef.every_kth_day(st.valid, BENCH["cpu_every_kth_day"], BENCH["cpu_max_rows"], ROWS["seed"])
        sub = st.valid.iloc[pos].reset_index(drop=True)
        X, T = fixed_batch(st)
        scores = {"unpruned, fp32": st.scores[pr.BASE][pos], f"{pick[0]}, fp32": st.scores[pick[0]][pos]}
        for name, label in ((pr.BASE, "unpruned"), (pick[0], pick[0])):
            q = ef.quantize_static(st.model(name), calib)
            scores[f"{label}, INT8"] = ef.score(q, st.src.dataset(sub, "valid"), CPU, "int8", ROWS["eval_batch"], WORKERS)
            for r in ef.benchmark("INT8 CPU", q, CPU, "int8", X, T, BENCH["cpu_batches"], BENCH["reps"]):
                LATENCY.append(dict(source=tag, variant=name, precision=r.pop("variant"), size_mb=ef.model_bytes(q) / 2**20, **r))
        INT8 = ef.compare(sub, scores, "unpruned, fp32", STATS["n_boot"], ROWS["seed"])
        INT8.to_csv(RUN_DIR / "int8_after_pruning.csv", index=False)
        print(f"{tag}: INT8 on {len(sub):,} CPU rows"); display(INT8.round(4))
    except (RuntimeError, NotImplementedError) as e:  # no x86 quantized backend on this machine
        print("INT8 after pruning skipped:", str(e).splitlines()[0])
LATENCY = pd.DataFrame(LATENCY)
LATENCY.to_csv(RUN_DIR / "latency.csv", index=False)
display(LATENCY.round(3))
"""), md("""
## 7. The scratch ViT: attention heads and tokens
The layer table comes first: it shows where this ViT spends its MACs. Head importance is the mean absolute gradient
of the loss with respect to a gate on each head's output, on training windows (Michel et al. 2019); the least
important heads over all 8 backbone blocks are removed (at least one stays in every block). Token pruning keeps,
after the attention of the listed blocks, the patch tokens that receive the most attention; later blocks then work
on fewer tokens. Fine-tuning uses the ViT's own rule (AdamW, clipping) at `vit_lr_scale` of its learning rate.
"""), code("""
VS, VIT_LATENCY = None, pd.DataFrame()
if VIT_SOURCE is not None:
    for st in CNN.values():
        st.release()                                  # free the CNN chart caches
    src, run = load_source(VIT_SOURCE)
    print(f"--- vit: Step {VIT_SOURCE['step']} run {run.name}, cell {VIT_SOURCE['cell']}, model {src.name}")
    VS = pr.Study(src, "vit", ROWS, DEVICE, RUN_DIR, WORKERS)
    VS.baseline()
    show_layers(VS)
    tune = pr.finetune_settings(src, FINETUNE["epochs"], FINETUNE["patience"], ROWS["eval_batch"],
                                lr_scale=FINETUNE["vit_lr_scale"], warmup_steps=FINETUNE["vit_warmup_steps"])
    ip = VS.dir / "head_importance.csv"
    if ip.exists():
        IMPORTANCE = torch.tensor(pd.read_csv(ip, index_col=0).to_numpy(), dtype=torch.float32)
    else:
        IMPORTANCE = pr.head_importance(src.model, VS.scan_batches(VIT["importance_batches"], tune["batch_size"]), DEVICE)
        pd.DataFrame(IMPORTANCE.numpy(), index=[f"block {l + 1}" for l in range(IMPORTANCE.shape[0])],
                     columns=[f"head {h + 1}" for h in range(IMPORTANCE.shape[1])]).to_csv(ip)
    print("head importance (L2-normalised within each block):"); display(pd.read_csv(ip, index_col=0).round(3))
    ALL_HEADS = IMPORTANCE.numel()
    blocks0 = lambda cfg: {b - 1: cfg["keep"] for b in cfg["after_blocks"]}
    said = lambda cfg: f"tokens {cfg['keep']:.0%} after blocks {', '.join(map(str, cfg['after_blocks']))}"
    for n in VIT["heads"]:
        kept = pr.select_heads(IMPORTANCE, n)
        VS.variant(f"{n} of {ALL_HEADS} heads, fine-tuned", "ViT heads",
                   lambda kept=kept, n=n: dict(model=pr.prune_vit(src.model, heads=kept), heads_kept=n,
                                               heads={str(l + 1): [h + 1 for h in v] for l, v in kept.items()}), tune)
    for cfg in VIT["tokens"]:
        make = lambda cfg=cfg: dict(model=pr.prune_vit(src.model, tokens=blocks0(cfg), score=VIT["score"]),
                                    token_keep=cfg["keep"], after_blocks=cfg["after_blocks"])
        VS.variant(f"{said(cfg)}, no fine-tuning", "ViT tokens, one-shot", make)
        VS.variant(f"{said(cfg)}, fine-tuned", "ViT tokens", make, tune)
    if VIT["combined"]:
        cfg = VIT["combined"]
        kept = pr.select_heads(IMPORTANCE, cfg["heads"])
        VS.variant(f"{cfg['heads']} of {ALL_HEADS} heads + {said(cfg)}, fine-tuned", "ViT heads + tokens",
                   lambda: dict(model=pr.prune_vit(src.model, heads=kept, tokens=blocks0(cfg), score=VIT["score"]),
                                heads_kept=cfg["heads"], token_keep=cfg["keep"], after_blocks=cfg["after_blocks"]), tune)
    VIT_LATENCY = pd.DataFrame(latency_rows(VS, [n for n, i in VS.info.items() if "one-shot" not in i["family"]]))
    VIT_LATENCY.to_csv(RUN_DIR / "latency_vit.csv", index=False)
    display(VIT_LATENCY.round(3))
"""), md("""
## 8. Summary
One row per network: validation AUC with the paired difference to the unpruned float32 network (same rows, whole
days resampled), then what was saved. `compression_ratio` = unpruned parameters ÷ non-zero parameters;
`macs_reduction` is for dense arithmetic (what a GPU runs), `macs_sparse_g` what an ideal sparse engine would need.
`no_loss`: the 95% interval of the difference lies above −`STATS["margin"]` AUC (a "tied" verdict alone does not
show that nothing was lost).
"""), code("""
STUDIES = list(CNN.values()) + ([VS] if VS is not None else [])
SUMMARY = pd.concat([s.table(STATS["n_boot"], ROWS["seed"], STATS["margin"]) for s in STUDIES], ignore_index=True)
lat = pd.concat([LATENCY, VIT_LATENCY], ignore_index=True)
REF = "fp32 GPU" if DEVICE.type == "cuda" else "fp32 CPU"
if len(lat):
    key = ["source", "variant"]
    ref = lat[lat.precision == REF]
    b1 = ref[ref.batch == 1].set_index(key)
    big = ref.sort_values("batch").groupby(key).tail(1).set_index(key)
    cpu1 = lat[(lat.precision == "fp32 CPU") & (lat.batch == 1)].set_index(key)
    SUMMARY = SUMMARY.set_index(key)
    SUMMARY["ms_batch1"], SUMMARY["images_per_s_large_batch"] = b1.ms_per_batch, big.images_per_s
    SUMMARY["gpu_memory_mb"], SUMMARY["cpu_ms_batch1"] = big.peak_memory_mb, cpu1.ms_per_batch
    SUMMARY = SUMMARY.reset_index()
SUMMARY.to_csv(RUN_DIR / "summary.csv", index=False)
display(SUMMARY.drop(columns=["model", "note"]).round(4))
"""), code("""
CHECKS = []
for st in STUDIES:
    names, fam = list(st.info), {n: i["family"] for n, i in st.info.items()}
    pairs = [(n, n.replace("fine-tuned", "no fine-tuning"), "fine-tuning after pruning") for n in names
             if n.endswith("fine-tuned") and n.replace("fine-tuned", "no fine-tuning") in st.info]
    by_channels = {}
    for n in names:
        if fam[n].startswith("structured"):
            how = "scratch" if fam[n] == "structured, from scratch" else st.info[n]["criterion"]
            by_channels.setdefault(tuple(st.info[n]["channels"]), {})[how] = n
    for d in by_channels.values():
        if "taylor" in d and "l2" in d:
            pairs.append((d["taylor"], d["l2"], "Taylor ranking vs L2 ranking, same widths"))
        if "scratch" in d and "l2" in d:
            pairs.append((d["l2"], d["scratch"], "pruned + fine-tuned vs the same narrow CNN trained from scratch"))
    extra = {}
    floor = next((k for k in ("ind_tab_only_rv", "ind_tab_only") if k in st.src.references), None)
    if floor and st.src.n_tab:
        extra["saved: features only (no image)"] = st.src.references[floor][st.valid_pos]
        pairs += [(n, "saved: features only (no image)", "CNN + numbers vs the numbers alone") for n in names
                  if fam[n] in ("reference", "structured, fine-tuned")]
    CHECKS.append(st.paired(pairs, STATS["n_boot"], ROWS["seed"], extra))
CHECKS = pd.concat(CHECKS, ignore_index=True)
CHECKS.to_csv(RUN_DIR / "checks.csv", index=False); display(CHECKS.round(4))
"""), code("""
import matplotlib.pyplot as plt
n_panels = 3 if VS is not None else 2
fig, ax = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 4.8))
err = lambda d: [d.auc - d.auc_low, d.auc_high - d.auc]
has = lambda d, text: d.variant.str.contains(text, regex=False)
for st, colour in zip(CNN.values(), ("tab:blue", "tab:orange", "tab:purple", "tab:brown")):
    d = SUMMARY[SUMMARY.source == st.tag]
    ref, tuned = d[d.family == "reference"], d[d.family == "structured, fine-tuned"]
    l2 = pd.concat([ref, tuned[has(tuned, "(l2)")]]).sort_values("macs_g")
    ax[0].errorbar(l2.macs_g, l2.auc, yerr=err(l2), fmt="o-", capsize=3, color=colour, label=f"{st.tag}: L2 ranking, fine-tuned")
    for _, r in l2[l2.family != "reference"].iterrows():                 # channel widths next to each point
        ax[0].annotate(r.variant.split(" ")[1], (r.macs_g, r.auc), fontsize=7, xytext=(4, 4), textcoords="offset points")
    for marker, what, other in (("^", "Taylor ranking, fine-tuned", tuned[has(tuned, "(taylor)")]),
                                ("x", "trained from scratch", d[d.family == "structured, from scratch"])):
        if len(other):
            ax[0].errorbar(other.macs_g, other.auc, yerr=err(other), fmt=marker, capsize=3, color=colour, alpha=0.6, label=f"{st.tag}: {what}")
    for fam, ls in (("unstructured, one-shot", "--"), ("unstructured, fine-tuned", "-")):
        u = pd.concat([ref, d[d.family == fam]]).sort_values("sparsity")
        if len(u) > 1:
            ax[1].errorbar(u.sparsity, u.auc, yerr=err(u), fmt="o", ls=ls, capsize=3, color=colour,
                           label=f"{st.tag}: {'as pruned' if 'one-shot' in fam else 'fine-tuned'}")
    floor = next((k for k in ("ind_tab_only_rv", "ind_tab_only") if k in st.src.references), None)
    if floor and st.src.n_tab:                                             # the numbers alone: the floor of a hybrid CNN
        y = ef._auc(st.valid.label.to_numpy(), st.src.references[floor][st.valid_pos])
        for a in ax[:2]:
            a.axhline(y, color=colour, ls=":", lw=1.2, label=f"{st.tag}: features only (no image)")
ax[0].set_xscale("log")
ax[0].set(xlabel="GMACs per sample (log scale)", ylabel="validation AUC", title="Channel pruning: AUC vs MACs (labels: channels kept)")
ax[1].set(xlabel="share of zero parameters", ylabel="validation AUC", title="Magnitude pruning: AUC vs sparsity")
if VS is not None:
    d = SUMMARY[SUMMARY.source == "vit"]
    for (fam, g), marker in zip(d.groupby("family", sort=False), "osD^vP"):
        ax[2].errorbar(g.macs_g, g.auc, yerr=err(g), fmt=marker, capsize=3, label=fam)
    for k, (_, r) in enumerate(d[d.family != "reference"].iterrows()):
        short = r.variant.replace(", fine-tuned", "").replace(", no fine-tuning", " (as pruned)")
        ax[2].annotate(short, (r.macs_g, r.auc), fontsize=6, xytext=(4, 4 + 8 * (k % 3)), textcoords="offset points")
    ax[2].set(xlabel="GMACs per sample", ylabel="validation AUC", title="Scratch ViT: heads and tokens")
for a in ax:
    a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(RUN_DIR / "pruning.png", dpi=130); plt.show()
"""), code("""
lines = [f"# Step 7b: pruning (run {RUN_ID}{', QUICK' if QUICK else ''})", ""]
for source, d in SUMMARY.groupby("source", sort=False):
    b = d[d.family == "reference"].iloc[0]
    lines.append(f"- **{source}** ({b.model}): AUC {b.auc:.4f} on {b.auc_rows:,} rows / {b.days} days; {b.parameters:,} parameters, "
                 f"{b.macs_g:.3f} GMACs, {b.size_mb:.2f} MB")
    for _, r in d[d.family != "reference"].iterrows():
        saved = (f"{r.sparsity:.0%} zeros, {r.compression_ratio:.1f}x fewer non-zero parameters, {r.size_zlib_mb:.2f} MB compressed"
                 if r.family.startswith("unstructured") else
                 f"{r.parameters:,} parameters ({r.compression_ratio:.1f}x), {r.macs_g:.3f} GMACs ({r.macs_reduction:.2f}x)")
        lines.append(f"  - {r.variant}: AUC {r.auc:.4f}, vs unpruned {r.diff_vs_fp32:+.4f} [{r.diff_low:+.4f}, {r.diff_high:+.4f}] "
                     f"{'no loss' if r.no_loss else r.verdict}; {saved}")
for _, r in CHECKS.iterrows():
    lines.append(f"- {r.source}, {r.check}: {r.a} − {r.b} = {r['diff']:+.4f} [{r.low:+.4f}, {r.high:+.4f}] -> {r.verdict}")
TAKEAWAYS = "\\n".join(lines)
(RUN_DIR / "takeaways.md").write_text(TAKEAWAYS + "\\n"); display(Markdown(TAKEAWAYS))
"""), md("""
### How to read it
- **Magnitude pruning**: the gap between the dashed (as pruned) and solid (fine-tuned) lines is what fine-tuning
  recovers. The saving is in non-zero parameters and compressed size only; latency needs a sparse runtime.
- **Channel pruning**: the point furthest left that is still `no_loss` is the smallest CNN that keeps the AUC. For
  the final model compare it with the dotted features-only line: a narrow CNN that sits on that line adds nothing
  to the 43 numbers, so its picture input is no longer earning its MACs.
- **Pruned vs trained from scratch**: tied means the narrow architecture is what matters, not the inherited weights.
- **ViT**: read the layer table first. Head and token pruning act on the 8 backbone blocks; the tokenizer's single
  attention block over 1,521 tokens is outside their reach, which bounds the MACs they can save.
""")]
    return c
