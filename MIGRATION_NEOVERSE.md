# NeoVerse DiffSynth Removal Plan

Goal: move all runtime code used by NeoVerse out of `diffsynth/`, keep the
NeoVerse behavior unchanged, and use a root-level `wan/` package for the
official Wan model code.

## Target Packages

- `wan/`: official Wan code used as the base model implementation.
- `neoverse/`: NeoVerse-specific runtime, preprocessing, control branch,
  reconstructor/VGGT integration, loaders, LoRA, data utilities, training
  helpers, and migration-only weight converters.
- `neoverse/adapters/`: bridge code between NeoVerse control logic and the
  official Wan module interfaces.
- `neoverse/utils/`: shared utility code such as device helpers and camera/video
  helpers.
- `diffsynth-old/`: backup only. Do not scan or migrate from it.
- `diffsynth/`: current behavior reference. Remove after all new paths pass
  training, inference, data, cache, and evaluation checks.

## Migration Checks

1. Run normal data loading, training, inference, cache building, and evaluation
   from the new `neoverse`/`wan` path.
2. Use a temporary legacy converter to load original NeoVerse weights and any
   modified/new control weights into the new package.
3. Run inference and evaluation with converted weights.
4. Confirm no production entrypoint imports `diffsynth`.
5. Remove the temporary converter only after new-format checkpoints are the
   default save/load path.

## Current First Phase

- Create root-level `wan/` from the official Wan2.1 implementation, because
  NeoVerse checkpoints are Wan2.1-based.
- Create root-level `neoverse/` and lift the current NeoVerse pipeline,
  control branch, Wan-compatible modules, loader pieces, LoRA loader, VRAM
  wrappers, scheduler, and auxiliary/data helpers.
- Keep old `diffsynth/` in place until the new package imports and behavior are
  validated.

## Progress

- Production imports have been switched from `diffsynth` to `neoverse`/`wan`.
- `neoverse` no longer imports `diffsynth`.
- Wan DiT loading now points at `wan.modules.model.WanModel`.
- FlowMatch inference sigmas are now derived from `wan.utils.fm_solvers`.
- The NeoVerse tiled/enhanced Wan VAE lives under `wan.modules.vae_neoverse`.
- Device helpers have been moved to `neoverse.utils.device`; the unused
  gradient helper was removed.
- Legacy state-dict converters were moved under `neoverse.loaders.converters`.
- Wan-specific migration bridge code was moved to `neoverse.adapters.wan`.
- NeoVerse control branch no longer inherits the migrated DiffSynth DiT block;
  it uses the official Wan block submodule layout with a NeoVerse-compatible
  forward path.
- The migrated `neoverse/models/wan_video_dit.py` compatibility copy has been
  removed.
- A migration-only legacy converter exists at
  `tools/convert/convert_neoverse_weights.py`.
- Production code no longer references `diffsynth`; only this migration note
  mentions it.

## Remaining Before Deleting `diffsynth/`

- Run a real `NeoVersePipeline.from_pretrained(...)` load with the project
  checkpoint set.
- Convert and evaluate original NeoVerse diffusion weights through the temporary
  converter.
- Run fixed-seed inference, frozen-cache build, training, and eval.
- Fix the local environment dependency issues observed during smoke tests
  (`omegaconf` missing and `transformers`/`huggingface_hub` version mismatch).
