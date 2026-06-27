NVIDIA Nemotron Model Reasoning Challenge 2025
Competition submission for the NVIDIA Nemotron Model Reasoning Challenge Kaggle competition.

Competition Link: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge

Overview
This project fine-tunes a LoRA adapter for the Nemotron-3-Nano-30B model to solve logical reasoning puzzles. The competition required building a rank-32 LoRA adapter that could solve novel bit manipulation and algebraic equation reasoning tasks, with model outputs evaluated on exact string match or numerical tolerance.

Results
Placement: 1582nd out of 4182 participants (62.1 percentile)

Public Leaderboard Score: Version 49 (0.86 wMAE on public 50% test split)

Private Test Performance: Version 49 (0.840 full test wMAE)

Post-Competition Analysis: Version 48 (0.848 full test wMAE) outperformed Version 49 on the full private test data, revealing critical overfitting to the limited public leaderboard. This experience mirrored lessons from the NeurIPS Polymer Challenge — tuning specifically to optimize the partial public leaderboard led to worse generalization on unseen data. Version 49 achieved better public scores through aggressive parameter tuning, but this did not transfer to the private 50% of test data.

Not a winning entry, but demonstrates systematic optimization of adapter compression techniques and ensemble inference strategies on a real Kaggle competition — with reinforced lessons about the dangers of leaderboard overfitting and the importance of generalizable approaches over public metric chasing.

Methodology
The approach combines GlyphMatics v10 compression with conservative format enforcement and systematic parameter tuning of a pre-built huikang/nemotron-adapter (v20).

Core Strategy:
Aggressive LoRA Compression — tuned SVD_ENERGY_GAIN_CAP from 1.22 → 1.10 to preserve adapter detail while maintaining rank-32 constraints. Lower gain caps reduced information loss during rank reduction, prioritizing reasoning capacity over compression aggressiveness.

GlyphMatics v10 Configuration — used stacked gain + dual-phase rebuild with:
SVD_ENERGY_GAIN_CAP: 1.10 (less aggressive compression = more detail preserved)
ROW_NORM_GAIN_CAP: 1.05 (guard against per-row overshoot)
PAIRFOLD_MIN_SIM: 0.75 (stricter vector direction matching during transport)
DETERMINISTIC_REBUILD_ACCEPT_EPS: 0.004 (phase-2 rebuild fallback)

Conservative Format Enforcement — wrapped only short incomplete responses (< 15 words) with \boxed{} format, avoiding over-wrapping full reasoning chains that already contained the required format.

Inference Optimization — temperature=0.0 (deterministic), top_p=1.0, max_tokens=512 for focused reasoning output. Early testing showed temperature tuning, rank reduction, system prompt examples, and extended token limits did not improve public leaderboard performance.

Scoring Journey — iterative optimization from baseline 0.51 → 0.85 → 0.86 on public data through GlyphMatics parameter tuning alone. All subsequent experiments (temperature, rank 16 compression, prompt engineering, token limits) yielded no improvement, indicating a local optimum had been reached.

The Overfitting Lesson — Version 49 was submitted because it achieved the highest public leaderboard score (0.86). However, Version 48 with slightly different compression parameters (SVD_ENERGY_GAIN_CAP=1.10, ROW_NORM_GAIN_CAP=1.05, PAIRFOLD_MIN_SIM=0.75) actually scored 0.848 on full test data vs Version 49's 0.840. This revealed that aggressive tuning to the 50% public split created overfitting — the parameters that looked best on partial data generalized worse to the full evaluation set. The gap to the top score (0.92) was 0.08 points (8 additional problems correct out of 100), but this gap likely required fundamentally different approaches (custom-trained adapters, RL/GRPO fine-tuning, synthetic data) rather than parameter optimization of a pre-built adapter.

Code Structure
Main notebook: .ipynb file containing full adapter building and submission pipeline

Key stages:
Setup and package installation (tinker-cookbook, mamba wheels, transformers)
GlyphMatics v10 patch application to tinker-cookbook.weights._adapter
Adapter loading and building (base model + huikang/nemotron-adapter v20)
Inference configuration (temperature, top_p, max_tokens, format enforcement)
Submission packaging (.zip creation with adapter_config.json + adapter_model.safetensors)

Environment & Dependencies
This code was developed in a Kaggle notebook environment with restricted internet access, requiring pre-packaged wheel (.whl) files for Mamba dependencies:

tinker-cookbook (adapter building and compression) — installed via pip
causal_conv1d (Mamba-2 kernel) — installed via .whl from GitHub releases
mamba_ssm (Mamba state space model) — installed via .whl from GitHub releases
transformers, torch, peft (model loading and LoRA)
The notebook includes custom wheel installation logic to handle the offline Kaggle environment. Wheel URLs are hardcoded but can be updated for local use with standard internet access.

Key Learnings
The Overfitting Trap Revisited (Version 49 vs Version 48): This competition reinforced a critical lesson from the NeurIPS Polymer Challenge. Version 49 achieved the highest public leaderboard score (0.86) and was submitted. However, Version 48 actually generalized better to the full private test data (0.848 vs 0.840). This revealed that tuning specifically to optimize the partial public leaderboard led to worse generalization on unseen data. The insight forced a reckoning with optimization practices — relying solely on public leaderboard MAE can mask poor generalization. In real ML work, monitoring multiple evaluation windows, cross-validation patterns, and performance stability across data splits is essential to distinguish between true improvement and leaderboard overfitting. This competition taught me to always ask: "Does this improvement generalize beyond the public split, or am I just fitting the evaluation set?"

Systematic Iteration Pays Off — the jump from 0.51 → 0.86 (68% improvement) came from methodical GlyphMatics parameter tuning, not wild experimentation. However, this same methodology created the risk of overfitting to the public leaderboard — the parameters that looked best on 50% of data didn't transfer perfectly to 100%.

When to Stop Optimizing — after testing temperature, rank compression, prompts, token limits, and different adapters with zero improvement on public data, continuing to optimize represented diminishing returns. The local optimum had been reached on the available evaluation window. This suggests that further gains would require fundamentally different approaches (RL, custom training, data synthesis) rather than parameter tweaking.

Key Features
Aggressive GlyphMatics v10 compression — systematically tuned SVD gain caps and PAIRFOLD parameters for optimal adapter detail preservation
Pre-built adapter optimization — leveraged huikang/nemotron-adapter v20 with focused compression tuning rather than training from scratch
Conservative format enforcement — multi-strategy \boxed{} detection and wrapping to avoid over-correcting model outputs
Inference stability — temperature=0.0 deterministic decoding with proven parameter settings
Systematic ablation — tested temperature, rank, prompts, token limits; reverted changes that didn't improve public scores

Notes for Users
This is a Kaggle competition submission, not a production system. The code prioritizes the competition metric (reasoning accuracy on logical puzzles) and uses Kaggle-specific file paths (/kaggle/input/, /kaggle/working/). To adapt for local use:

Replace Kaggle input/output paths with your own directories
Install dependencies: pip install tinker-cookbook transformers torch peft
Download mamba wheels locally or install via pip if internet is available
Update base model and adapter paths to point to your local Nemotron-3-Nano-30B and huikang/nemotron-adapter checkpoints
Run cells sequentially; adapter building is I/O intensive
The submission generates submission.zip containing adapter_config.json and adapter_model.safetensors for Kaggle competition submission

Key Takeaway
Leaderboard scores on partial test data are a poor guide to generalization. Version 49 looked like the winner on public data but underperformed Version 48 on full data. This mirrors real ML practice: always validate across multiple evaluation windows and question whether improvements are genuine or leaderboard-specific. Chasing public scores without monitoring generalization patterns is a trap.

License
Apache License 2.0 — See LICENSE file for details.
