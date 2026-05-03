# DL3DV-10K Download Bundle

This bundle downloads DL3DV-10K from the official gated Hugging Face repositories.

Before running, accept the dataset terms on the relevant Hugging Face dataset page, then provide a token:

```bash
export HF_TOKEN=hf_xxx
```

Safe preview for the first 480P shard:

```bash
python scripts/download/dl3dv_10k/download_dl3dv.py \
  --resolution 480P \
  --file-type images+poses \
  --subset 1K \
  --out-dir data/DL3DV-10K \
  --dry-run
```

Download the first 480P shard:

```bash
python scripts/download/dl3dv_10k/download_dl3dv.py \
  --resolution 480P \
  --file-type images+poses \
  --subset 1K \
  --out-dir data/DL3DV-10K \
  --verify-zip
```

Download all 480P images+poses after checking disk space:

```bash
python scripts/download/dl3dv_10k/download_dl3dv.py \
  --resolution 480P \
  --file-type images+poses \
  --subset all \
  --out-dir data/DL3DV-10K \
  --verify-zip
```

Defaults use `https://hf-mirror.com`. Add `--use-official-hf` to use upstream Hugging Face.
