# charruaLLM

Modelo base (next-token prediction) entrenado sobre el dataset UY22: noticias
uruguayas en español de El Observador, El País y Montevideo Portal (corpus en `data/`).

Inspirado en nanoGPT / nanochat pero recortado al hueso para que entrene local
en una Mac (M-series via MPS) y se pueda lanzar igual en una GPU NVIDIA sin tocar código.

No incluye SFT, RLHF ni evaluación contra benchmarks. Es un text completer puro.

## Estructura

```
charruaLLM/
├── data/                   # corpus crudo (.txt, una oración por línea)
├── nanochat/               # repo de referencia, no se usa en runtime
├── model.py                # GPT decoder-only minimal
├── train_tokenizer.py      # entrena BPE byte-level → tokenizer.json
├── prepare_data.py         # tokeniza corpus → train.bin / val.bin / meta.json
├── train.py                # training loop
└── sample.py               # generación desde checkpoint
```

## Setup

Con `uv` (recomendado, ya lo usás en nanochat):

```bash
uv venv
source .venv/bin/activate
uv pip install -r <(uv pip compile pyproject.toml)
```

O con `pip`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install "torch>=2.4" "numpy>=1.26" "tokenizers>=0.20" "tqdm>=4.66"
```

## Pipeline

### 1) Entrenar el tokenizer (1–3 min)

```bash
python train_tokenizer.py
```

Lee una muestra de cada `.txt` en `data/` y escribe `tokenizer.json` (BPE
byte-level, vocab=16384 por defecto).

### 2) Tokenizar el corpus (3–10 min)

```bash
python prepare_data.py
```

Recorre todos los `.txt` y escribe `train.bin` y `val.bin` (numpy `uint16`),
más un `meta.json` con `vocab_size` y conteos.

### 3) Entrenar (default M4 Max friendly, ~30–60 min)

```bash
python train.py
```

Defaults: `n_layer=6, n_embd=384, n_head=6, block_size=512, batch_size=32,
max_steps=5000`. Modelo de ~14M parámetros. Auto-detecta dispositivo
(`cuda > mps > cpu`).

Ejemplos:

```bash
# corrida corta para validar que todo funciona
python train.py --max-steps 200 --eval-every 50

# corrida larga en NVIDIA con compile + bf16
python train.py --device cuda --compile --max-steps 20000 --batch-size 64

# modelo más grande
python train.py --n-layer 8 --n-embd 512 --n-head 8 --max-steps 20000
```

### 4) Generar

```bash
python sample.py --ckpt runs/run1/best.pt --prompt "El gobierno uruguayo"
```

## Notas

- En NVIDIA: bf16 autocast automático + opción `--compile` para `torch.compile`.
- En MPS: fp32 (es lo más estable; bf16 en MPS aún tiene casos raros).
- Los `.txt` están "splitted" por oración, así que el modelo aprende patrones
  cortos. Para mejorarlo en el futuro, conviene reagrupar por noticia entera
  antes de tokenizar.
- Para sumar más datos: dropea más `.txt` en `data/` y volvé a correr pasos 1 y 2.
- Con ~600M tokens de corpus y ~14M params estás cerca del óptimo Chinchilla.
  Si subís el modelo (más capas/ancho) sumá más datos o aceptás sub-entrenarlo.
