# 06 – Anchor-fusion EF model and imbalanced-regression experiments

Developed on Google Colab (GPU). Paths default to `/content/drive/MyDrive/EchoNet-Dynamic`; set `ECHONET_ROOT` to override.
All scripts in this folder import their helpers by bare module name, so run them from inside this folder.

| Group | Files |
|---|---|
| Shared modules | `anchor_frame_sampler.py`, `anchor_fusion_dataset.py` (v2), `anchor_fusion_dataset_v3.py`, `anchor_fusion_model.py` (v2), `anchor_fusion_model_v3.py`, `fds_module_v3.py`, `lds_loss.py`, `weighted_sampler_v3.py` |
| Preparation | `build_anchor_lookup.py` (ED/ES anchor tables) |
| v2 baseline | `train_anchor_fusion.py`, `test_anchor_fusion.py` (DenseNet-BiLSTM, ED/ES-anchored sampling, LDS-weighted Huber) |
| v3 (FDS + weighted sampler) | `train_anchor_fusion_v3.py`, `test_anchor_fusion_v3.py` |
| Feature caches | `extract_cached_features_full.py`, `extract_cached_fused_sequences.py`, `extract_cached_spatial_maps.py`, `extract_3dcnn_fused_cache.py`, `extract_r3d_spatial_cache_convlstm.py` |
| FDS / LDS experiments | `train_v4_decoupled_fds.py`, `train_v4_fds_partial_unfreeze.py`, `train_v5_protected_fds.py` |
| Temporal-head experiments | `train_rank1_tam.py`, `train_rank2_convlstm.py`, `cached_feature_experiments/` (TCN, Transformer, dual-scale TCN on DenseNet caches; attention/TCN/Transformer/ConvLSTM on R3D-18 caches) |
| End-to-end TCN | `train_densenet_tcn_full_video.py`, `test_densenet_tcn_full_video.py` |

Typical order: `build_anchor_lookup.py` → `train_anchor_fusion.py` → `extract_cached_fused_sequences.py` → temporal-head experiments, or `train_densenet_tcn_full_video.py` → `test_densenet_tcn_full_video.py`.
Each script documents its required inputs and outputs in its header; test sets are only used by the `test_*` scripts.
