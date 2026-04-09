<<<<<<< HEAD
# Unsupervised Neural Network for Multi-Genre Music Generation

This repository implements all four assignment tasks for CSE425/EEE474:

1. Task 1: LSTM Autoencoder
2. Task 2: Variational Autoencoder (VAE)
3. Task 3: Transformer Decoder (autoregressive)
4. Task 4: RLHF-style preference tuning (policy gradient with human or simulated rewards)

## Project Structure

- `data/raw_midi/`: raw MIDI files grouped by genre subfolders
- `data/processed/`: tokenized sequences and vocabulary
- `data/train_test_split/`: optional split metadata
- `src/preprocessing/`: parsing, tokenization, piano-roll conversion
- `src/models/`: AE, VAE, Transformer, optional diffusion placeholder
- `src/training/`: training entrypoints for each task
- `src/evaluation/`: metrics and comparisons
- `src/generation/`: sequence sampling and MIDI export
- `outputs/generated_midis/`: generated samples
- `outputs/plots/`: loss/perplexity plots
- `outputs/survey_results/`: human feedback CSV and summaries

## Setup

```bash
pip install -r requirements.txt
```

## Data Format

Place MIDI files inside genre folders:

```text
data/raw_midi/
  classical/*.mid
  jazz/*.mid
  rock/*.mid
  pop/*.mid
  electronic/*.mid
```

## End-to-End Workflow

1. Preprocess and tokenize MIDI files
```bash
python -m src.preprocessing.midi_parser --input data/raw_midi --output data/processed
```

2. Train Task 1 (LSTM Autoencoder)
```bash
python -m src.training.train_ae --data data/processed/sequences.npz --out outputs
```

3. Train Task 2 (VAE)
```bash
python -m src.training.train_vae --data data/processed/sequences.npz --out outputs
```

4. Train Task 3 (Transformer)
```bash
python -m src.training.train_transformer --data data/processed/sequences.npz --out outputs
```

5. Run Task 4 (RLHF fine-tuning)
```bash
python -m src.training.train_rlhf --model-checkpoint outputs/checkpoints/transformer.pt --data data/processed/sequences.npz --out outputs
```

6. Evaluate
```bash
python -m src.evaluation.metrics --real data/processed/sequences.npz --generated outputs/generated_tokens.npz --out outputs
```

7. Export generated tokens to MIDI
```bash
python -m src.generation.generate_music --checkpoint outputs/checkpoints/transformer.pt --vocab data/processed/vocab.json --out outputs/generated_midis
```

## Baselines

- Random note generator: `src/evaluation/metrics.py` (`random_baseline`)
- Markov chain baseline: `notebooks/baseline_markov.ipynb` and utility in `src/generation/sample_latent.py`

## Notes

- Task 4 supports both real human scores (CSV) and a simulated reward function for debugging.
- Replace simulated rewards with survey results to complete final deliverables.
=======
# cse425
>>>>>>> b207f2421144af4e5effb0d93a442703bf18aed7
