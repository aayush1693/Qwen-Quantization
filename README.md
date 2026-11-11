# Quantizing Qwen2.5-3B-Instruct: Memory, Speed, and Quality Trade-offs

A reproducible benchmark of post-training quantization for [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) on a single free **Colab Tesla T4 (14.6 GB)**.

The notebook loads the same model in four precisions and compares them on weight memory, peak decode VRAM, prefill/decode throughput, WikiText-2 perplexity, MMLU accuracy, and the longest context that fits on the GPU.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aayush1693/Qwen-Quantization/blob/master/01_LLM_quantization_Qwen_3B.ipynb)

---

## TL;DR

| What | FP16 (baseline) | Best quantized result |
|---|---|---|
| Weight memory | 5.75 GB | **1.87 GB** with NF4 (**3.07×** smaller) |
| Peak decode VRAM | 5.84 GB | **2.02 GB** with NF4 + double quant (**−65%**) |
| Decode throughput | **18.7 tok/s** | 13.9 tok/s with NF4 (**0.75×**, i.e. slower) |
| WikiText-2 perplexity | **9.51** | 9.62 with INT8 (**+1.1%**); NF4 is 10.11 (**+6.3%**) |
| MMLU subset (n=300) | **68.3%** | 68.0% with NF4 (**−0.3 pts**) |
| Measured max context | 4,096 tokens | **8,192 tokens** for all quantized configs |

**Bottom line:** 4-bit NF4 cuts memory by roughly 3× and more than doubles the usable context on a 14.6 GB T4, at the cost of ~25% slower decoding and ~6% higher perplexity. bitsandbytes INT8 saves less memory and is ~4× slower to decode, so NF4 is the better trade-off here. Double quantization saves another ~0.12 GB of peak VRAM with no measurable quality cost in this run.

---

## Precision configurations

| Name | Kind | Quantization | Compute dtype |
|---|---|---|---|
| FP16 (baseline) | `fp16` | None | FP16 |
| INT8 (bnb LLM.int8) | `int8` | `load_in_8bit=True` | FP16 |
| NF4 (bnb 4-bit) | `nf4` | 4-bit NormalFloat, no double quant | FP16 |
| NF4 + double-quant | `nf4dq` | 4-bit NF4 + `bnb_4bit_use_double_quant=True` | FP16 |

A pre-quantized **AWQ INT4** slot is defined in the notebook but disabled (`RUN_PREQUANT = False`) because it requires extra packages. It is not part of the results below.

All models are loaded with `device_map={"": 0}` (single GPU), `model.eval()`, and no sampling during generation.

---

## Environment

The committed outputs were produced with:

| Component | Version |
|---|---|
| GPU | Tesla T4, 14.6 GB |
| Python | 3.13.16 |
| PyTorch | 2.11.0+cu130 |
| Transformers | 5.22.1 |
| Quantization backend | bitsandbytes (installed latest at runtime) |
| Seed | 42 |

> Library versions drift quickly. Re-running later may install newer `transformers`/`bitsandbytes` versions and produce different numbers.

---

## Results

Measured on the T4 with batch size 1, FP16 baseline, and seed 42.

### Memory and throughput

| Config | Weights (GB) | Compression | Peak decode VRAM (GB) | VRAM reduction | Decode (tok/s) | Decode vs FP16 | Prefill @2k (tok/s) | Prefill vs FP16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 (baseline) | 5.748 | 1.00× | 5.836 | 0.0% | 18.65 | 1.00× | 1438.2 | 1.00× |
| INT8 (bnb LLM.int8) | 3.164 | 1.82× | 3.351 | 42.6% | 4.82 | 0.26× | 1314.3 | 0.91× |
| NF4 (bnb 4-bit) | 1.872 | 3.07× | 2.145 | 63.2% | 13.93 | 0.75× | 1195.4 | 0.83× |
| NF4 + double-quant | 1.872 | 3.07× | 2.024 | 65.3% | 13.72 | 0.74× | 1200.8 | 0.84× |

- **Weights** are `model.get_memory_footprint()` after loading.
- **Peak decode VRAM** is `torch.cuda.max_memory_allocated()` while generating 128 tokens from a 512-token prompt.
- **Decode throughput** excludes prefill time by subtracting a 1-token run from a 128-token run.
- **Prefill throughput** includes one generated token per timed call.

### Quality

| Config | WikiText-2 PPL | ΔPPL | ΔPPL % | MMLU acc | ΔMMLU (pts) |
|---|---:|---:|---:|---:|---:|
| FP16 (baseline) | 9.509 | 0.000 | 0.0% | 68.3% | 0.0 |
| INT8 (bnb LLM.int8) | 9.617 | +0.108 | +1.1% | 64.7% | −3.7 |
| NF4 (bnb 4-bit) | 10.108 | +0.599 | +6.3% | 68.0% | −0.3 |
| NF4 + double-quant | 10.113 | +0.604 | +6.3% | 67.0% | −1.3 |

Perplexity uses the first 100 non-overlapping 1,024-token windows of the WikiText-2 **test** split. MMLU is a fixed random 300-question subset (56 subjects), zero-shot, scored by the logits of the ` A/B/C/D` tokens.

### Context length: measured OOM probe

Largest context that completed a forward pass + one generated token on this T4; probing stops at the first OOM.

| Config | 1k | 2k | 4k | 8k | 16k | Max OK |
|---|---:|---:|---:|---:|---:|---:|
| FP16 (baseline) | ✅ 5.99 GB | ✅ 6.50 GB | ✅ 8.41 GB | ❌ OOM | — | **4,096** |
| INT8 (bnb LLM.int8) | ✅ 3.41 GB | ✅ 3.93 GB | ✅ 5.83 GB | ✅ 13.10 GB | ❌ OOM | **8,192** |
| NF4 (bnb 4-bit) | ✅ 2.28 GB | ✅ 2.80 GB | ✅ 4.71 GB | ✅ 11.98 GB | ❌ OOM | **8,192** |
| NF4 + double-quant | ✅ 2.16 GB | ✅ 2.68 GB | ✅ 4.58 GB | ✅ 11.86 GB | ❌ OOM | **8,192** |

Values are peak allocated VRAM. The 8k runs are extremely close to the 14.6 GB limit, which is why the next step at 16k fails.

### Analytic fit for other GPUs

The notebook also estimates the maximum context for 4–24 GB cards using:

```text
free        = 0.92 × GPU_memory − weights_gb − 0.8 GB overhead
max_context = free / KV_bytes_per_token

KV_bytes_per_token = 2 × layers × kv_heads × head_dim × 2 bytes
```

For Qwen2.5-3B: 36 layers, 2 KV heads, head dimension 128 → **36 KiB/token**, i.e. **1.12 GB at 32k tokens**.

| Config | 4 GB | 6 GB | 8 GB | 12 GB | 16 GB | 24 GB |
|---|---:|---:|---:|---:|---:|---:|
| FP16 (baseline) | 0 | 0 | 23,650 | 32,768 | 32,768 | 32,768 |
| INT8 (bnb LLM.int8) | 0 | 32,768 | 32,768 | 32,768 | 32,768 | 32,768 |
| NF4 (bnb 4-bit) | 29,359 | 32,768 | 32,768 | 32,768 | 32,768 | 32,768 |
| NF4 + double-quant | 29,359 | 32,768 | 32,768 | 32,768 | 32,768 | 32,768 |

`0` means the weights alone do not fit. Values are capped at the model's `max_position_embeddings` (32,768). **This is an upper bound**: it ignores attention activation peaks, so real usable context is lower. Compare it with the measured probe above, where 16k OOMs despite the table predicting more room.

---

## Key findings

1. **Memory.** NF4 reduces weights from 5.75 GB to 1.87 GB (3.07×) and peak decode VRAM from 5.84 GB to 2.14 GB (−63%). INT8 reaches 3.16 GB weights and 3.35 GB peak (−43%). Double quantization removes another ~0.12 GB of peak VRAM on top of NF4.
2. **Quality.** WikiText-2 perplexity rises from 9.51 (FP16) to 9.62 (INT8, +1.1%) and 10.11 (NF4, +6.3%). On the MMLU subset, FP16 scores 68.3%, INT8 64.7%, NF4 68.0%, and NF4 + double-quant 67.0%.
3. **Speed.** Quantization **slows decoding** rather than speeding it up: 18.7 tok/s for FP16 versus 13.9 for NF4 and 4.8 for INT8. Prefill at 2,048 tokens is 9–17% slower than FP16. The bitsandbytes INT8 path is clearly the slowest.
4. **Context.** FP16 fits 4,096 tokens before OOM at 8,192. All three quantized configs reach 8,192 but OOM at 16,384. The extra headroom comes mostly from the smaller weights; the KV cache itself is unchanged and dominates at long context.
5. **Practical choice.** For this model on a T4, **NF4 (optionally with double quantization)** is the best memory/quality trade-off. Double quantization is essentially free in quality and saves a small amount of VRAM. Avoid bitsandbytes INT8 if decode latency matters.

---

## Repository layout

```text
Qwen-Quantization/
├── 01_LLM_quantization_Qwen_3B.ipynb   # the full benchmark notebook (executed outputs included)
└── README.md                            # this file
```

The notebook is organized into numbered sections:

| Section | Purpose |
|---|---|
| 1. Setup | Install dependencies, authenticate with Hugging Face, select GPU |
| 2. Configuration | Model ID, precision list, evaluation sizes, seed |
| 3. Data | Load WikiText-2 and a fixed MMLU subset |
| 4. Helper functions | Model loading, PPL, MMLU, timed generation, context probing |
| 5. Run the benchmark | Loop over configs, save JSON incrementally |
| 6. Results table | Build a dataframe with deltas versus the baseline and export CSV |
| 7. Plots | Bar charts, prefill curves, and a memory-vs-quality Pareto plot |
| 8. Fit estimate | Analytic max-context table for 4–24 GB GPUs |
| 9. Findings | Summary and caveats |
| Final cell | Zip and download all artifacts (Colab) |

---

## Running the notebook

### Option A: Google Colab (recommended)

1. Open the notebook with the Colab badge above, or upload `01_LLM_quantization_Qwen_3B.ipynb` to [colab.research.google.com](https://colab.research.google.com/).
2. Select **Runtime → Change runtime type → T4 GPU** (a free-tier T4 is sufficient for the full run).
3. Add a Hugging Face token:
   - In Colab, open the **🔑 Secrets** panel and add a secret named `HF_TOKEN`, or
   - set the `HF_TOKEN` environment variable before launching Jupyter.
4. Run all cells (**Runtime → Run all**).

The notebook reads the token from `google.colab.userdata` first, then falls back to `os.environ["HF_TOKEN"]`. A token is not strictly required for this public model, but it avoids anonymous rate limits and works for gated mirrors.

> **Expected runtime:** tens of minutes on a T4. Each configuration performs perplexity over ~100k tokens, 300 MMLU questions, throughput benchmarks, and up to five context probes. The FP16 load alone took ~166 s in the recorded run.

### Option B: Local GPU

Requirements:

- Linux (tested), CUDA-capable NVIDIA GPU
- Python 3.10+ and Jupyter
- A GPU with **at least ~8 GB** for the FP16 baseline at short context; **16 GB** is recommended to reproduce the full 8k context probes
- `bitsandbytes` must be able to build/load its CUDA kernels for your GPU

```bash
git clone https://github.com/aayush1693/Qwen-Quantization.git
cd Qwen-Quantization

python -m venv .venv
source .venv/bin/activate

pip install -U transformers accelerate bitsandbytes datasets matplotlib pandas jupyter
export HF_TOKEN="hf_..."     # optional but recommended

jupyter notebook 01_LLM_quantization_Qwen_3B.ipynb
```

Two Colab-specific details to change for a local run:

1. **Output directory.** In the setup cell, replace
   ```python
   OUT_DIR = "/content/qwen_quant_out"
   ```
   with a local path such as `OUT_DIR = "./qwen_quant_out"`.
2. **Download cell.** The final cell calls `google.colab.files.download`. It catches the exception outside Colab, so the run will not crash, but you may want to delete it or replace it with a local archive path.

---

## Configuration knobs

All main settings live in the **Configuration** cell (Section 2):

| Variable | Default | Meaning |
|---|---|---|
| `SEED` | `42` | Seed for Python, NumPy, PyTorch, CUDA, dataset shuffling |
| `MODEL_ID` | `Qwen/Qwen2.5-3B-Instruct` | Model to benchmark |
| `CONFIGS` | 4 configs | Precisions to compare; the first is the baseline for deltas |
| `RUN_PREQUANT` | `False` | Enable the AWQ INT4 pre-quantized slot |
| `PPL_MAX_LEN` | `1024` | Window length for perplexity |
| `PPL_MAX_WINDOWS` | `100` | Number of WikiText-2 windows (`None` = full test set, slower) |
| `MMLU_N` | `300` | Number of random MMLU test questions |
| `PREFILL_CTXS` | `[512, 1024, 2048, 4096]` | Prompt lengths for prefill throughput |
| `DECODE_PROMPT_LEN` | `512` | Prompt length for decode throughput |
| `DECODE_NEW_TOKENS` | `128` | Tokens generated per decode measurement |
| `REPEATS` | `3` | Median over repeated timings |
| `PROBE_CTXS` | `[1024, 2048, 4096, 8192, 16384]` | Context lengths tested for OOM |

To benchmark a different model, change `MODEL_ID` and adjust `PROBE_CTXS` for its context window. The analytic fit section reads `num_hidden_layers`, `num_key_value_heads`, `head_dim`, and `max_position_embeddings` directly from the model config.

### Adding AWQ / GPTQ checkpoints

Set `RUN_PREQUANT = True` after installing the relevant kernels (for AWQ: `autoawq`; for GPTQ: `auto-gptq` or `optimum`). Add your repository to `PREQUANT_CONFIGS`. The loading helper passes `torch_dtype=torch.float16` and no `BitsAndBytesConfig` for `kind="prequant"`, so the checkpoint's own quantization config is used.

---

## Methodology details

### Model loading

`load_model()` frees the previous model, resets CUDA peak-memory statistics, loads with `AutoModelForCausalLM.from_pretrained`, calls `model.eval()` and `torch.cuda.synchronize()`, and records load time. Each config is loaded, evaluated, and deleted in a `try/finally` block with `gc.collect()`, `torch.cuda.empty_cache()`, and `torch.cuda.ipc_collect()`.

### Perplexity

- WikiText-2 test split is tokenized as one long stream (299,078 tokens in this run).
- The stream is split into non-overlapping windows of `PPL_MAX_LEN` (1,024) tokens.
- Only the first `PPL_MAX_WINDOWS` (100) windows are used, i.e. ~102k tokens.
- Loss is weighted by the number of predicted tokens per window, summed, and exponentiated.
- This is a sliding-free, non-overlapping-window PPL, so absolute values may differ slightly from other harnesses.

### MMLU

- 300 questions are sampled once with `shuffle(seed=42).select(range(300))` from `cais/mmlu`, config `all`, split `test`.
- Each question is formatted as a subject-primed multiple-choice prompt ending in `Answer:`.
- The model is run once per question, and the prediction is the argmax over the last-token logits at the token IDs for `" A"`, `" B"`, `" C"`, `" D"`.
- This is **zero-shot** and uses answer-letter logits rather than generated text. Absolute scores are therefore not comparable to published 5-shot MMLU numbers. The useful signal is the **relative** FP16-vs-quantized delta.

### Throughput

- **Prefill:** for each context length, one warm-up call, then `REPEATS` timed calls generating exactly one token. Throughput is `context_length / median(seconds)`.
- **Decode:** warm up with 8 tokens. For each repeat, measure `t1` for one new token and `tn` for `new` tokens from the identical prompt. Decode throughput is `(new − 1) / (tn − t1)`, which removes prefill. The median across repeats is reported.
- Generation is greedy (`do_sample=False`, `use_cache=True`) and synchronized before/after timing.

### Context probe

`probe_contexts()` tries each context in `PROBE_CTXS` in ascending order. It runs one forward pass plus one generated token, records peak allocated VRAM on success, and **stops at the first OOM** (marking all larger contexts as failed). This measures what actually runs on the current GPU, not a theoretical limit.

### Analytic fit

The Section 8 estimate assumes 8% of GPU memory is unavailable, subtracts measured weights and a fixed 0.8 GB runtime overhead, then divides the remainder by the per-token KV cache size. It does **not** model attention activation peaks, CUDA context overhead, fragmentation, or allocator behavior, so it is an upper bound.

---

## Generated artifacts

All outputs are written to `OUT_DIR` (`/content/qwen_quant_out` on Colab):

| File | Contents |
|---|---|
| `results.json` | Raw per-config records, written after every config so a disconnect does not lose progress |
| `results_table.csv` | Results table with compression, VRAM reduction, speedups, and quality deltas |
| `summary_plots.png` | 2×3 panel: weight memory, peak VRAM, decode throughput, prefill vs context, PPL, MMLU |
| `pareto_vram_vs_ppl.png` | Peak VRAM vs perplexity scatter |
| `fit_table_max_context.csv` | Estimated max context for 4, 6, 8, 12, 16, and 24 GB GPUs |

The final cell additionally creates `/content/qwen_quant_results.zip` and triggers a browser download in Colab.

---

## Caveats and limitations

- **One model, one GPU, one seed, batch size 1.** Throughput and OOM thresholds will differ on other hardware, drivers, CUDA versions, and batch sizes. Do not treat these numbers as universal.
- **Small quality sample.** With n=300, the approximate 95% confidence interval on MMLU accuracy is about **±2.8 points**. The NF4 vs. NF4 + double-quant difference is within noise, and the INT8 drop is only borderline. A single PPL run of ~100k tokens is also noisy at the 0.1-point level.
- **MMLU is not the official protocol.** Zero-shot letter-logit scoring is a degradation proxy, not a substitute for 5-shot generation-based evaluation.
- **Peak VRAM is allocator-observed.** `max_memory_allocated` excludes some driver/CUDA context overhead. The analytic fit adds a fixed 0.8 GB allowance, but real free memory can be lower.
- **Context probes are coarse.** They only test 1k, 2k, 4k, 8k, and 16k, and they stop at the first failure. The true limit lies somewhere between the last success and the first OOM.
- **Quantization is inference-only bitsandbytes.** No AWQ, GPTQ, GGUF, or kernel-fused serving stack (vLLM, TensorRT-LLM, llama.cpp) is benchmarked. Those can be substantially faster than bitsandbytes at the same bit width.
- **Transformers deprecation warning.** The recorded run used `torch_dtype=...`, which newer Transformers versions deprecate in favor of `dtype=...`; it still works but may warn.
- **Package drift.** The notebook installs the latest `transformers`, `accelerate`, `bitsandbytes`, and `pandas` at runtime. Pinning versions is recommended for exact reproduction.

---

## Ideas for extending the benchmark

- Enable the provided AWQ INT4 slot and add GPTQ / GGUF baselines.
- Sweep multiple seeds and report confidence intervals for PPL and MMLU.
- Evaluate with the full WikiText-2 test set (`PPL_MAX_WINDOWS = None`).
- Add batched decode throughput (batch 2/4/8) and TTFT (time to first token).
- Compare against a fused serving runtime such as vLLM or TensorRT-LLM.
- Measure long-context quality (e.g. needle-in-a-haystack or RULER) rather than only whether a forward pass fits.
- Test larger Qwen2.5 sizes (7B, 14B) to see how the memory/quality trade-off scales.

---

## References

- Model: [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) · [Qwen2.5 technical report](https://arxiv.org/abs/2412.15115)
- bitsandbytes: [documentation](https://huggingface.co/docs/bitsandbytes) · [LLM.int8() paper](https://arxiv.org/abs/2208.07339) (Dettmers et al., 2022)
- NF4 / double quantization: [QLoRA paper](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023)
- WikiText-2: [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) · [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.05473) (Merity et al., 2016)
- MMLU: [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) · [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300) (Hendrycks et al., 2020)

---

## Notes on reproducibility

The notebook includes the executed outputs from the run summarized above. To reproduce, use the same model revision, GPU class, and pinned library versions. If you only need approximate trends, newer versions are fine, but expect the throughput and peak-VRAM figures to move.

No license file is included in this repository. Add one before redistributing or reusing the code.
