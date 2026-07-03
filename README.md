# MambaLRP-IG: Explaining Mamba using Layer-Wise Relevance Propagation and Integrated Gradients

This repository contains the code for my bachelor's thesis at the University of Twente, which extends **MambaLRP** — a Layer-wise Relevance Propagation (LRP) method for explaining Mamba (State Space Model) predictions — in two directions:

1. **Cross-domain validation**: testing MambaLRP on a long-sequence legal NLP dataset (ECtHR-A, from LexGLUE) in addition to the five short-sequence datasets used in the original paper.
2. **MambaLRP-IG**: a novel combination of MambaLRP with Integrated Gradients (IG), designed to reduce gradient saturation and improve explanation sensitivity.

The base MambaLRP implementation this repository builds on is from Rezaei Jafari et al., *"MambaLRP: Explaining Selective State Space Sequence Models"* (NeurIPS 2024). See [Citation](#citation) below.

## Research Questions

- **RQ1**: How well does MambaLRP generalize as an explanation method across different data domains, including long-sequence data?
- **RQ2**: Does incorporating Integrated Gradients as a weighting function in MambaLRP improve the sensitivity/faithfulness of relevance redistribution, compared to standard MambaLRP?

## Summary of Findings

- MambaLRP generalizes well to long-sequence data, reaching a faithfulness score (ΔA_F) of 2.365 on ECtHR-A with the 1.4B parameter model.
- MambaLRP-IG slightly outperforms MambaLRP on the long-sequence dataset (2.486 vs. 2.365), but across all datasets the average difference is minimal (−0.027), suggesting standard MambaLRP is already a stable and sufficient explanation method.

Full results, discussion, and limitations are available in the accompanying thesis document.

## Repository Structure

```
.
├── mamba_lrp/                              # Core MambaLRP / MambaLRP-IG implementation
├── assets/                                 # Images used in this README
├── minimal_demo.ipynb                      # Minimal working example (SST-2, 130M model)
├── Emotion_implementation_mambalrp.ipynb   # MambaLRP / MambaLRP-IG on the Emotion dataset
├── Vision_implementation.ipynb             # MambaLRP / MambaLRP-IG on Vision Mamba (ImageNet)
├── requirements.txt                        # General dependencies
├── requirements_hf_mamba_env.txt           # Environment for Hugging Face Mamba models
├── requirements_vim_env.txt                # Environment for Vision Mamba models
├── setup.py
└── LICENSE
```

> Note: the ECtHR-A long-sequence experiments and the MambaLRP-IG evaluation pipeline follow the same structure as the notebooks above, using the relevance function and evaluation code in `mamba_lrp/`.

## Installation

Two separate environments are used, since the Hugging Face Mamba models and Vision Mamba (Vim) models have different dependencies.

```bash
git clone https://github.com/AdamBosch/MambaLRP-IG.git
cd MambaLRP-IG

# For text/NLP experiments (Hugging Face Mamba models)
pip install -r requirements_hf_mamba_env.txt

# For vision experiments (Vision Mamba)
pip install -r requirements_vim_env.txt

pip install -e .
```

## Usage

- **Quick start**: run `minimal_demo.ipynb` for a minimal example using the pretrained 130M parameter Mamba model on SST-2.
- **Text/NLP experiments**: `Emotion_implementation_mambalrp.ipynb` shows the full MambaLRP and MambaLRP-IG pipeline on a text classification dataset; the same structure applies to the other NLP datasets (SST-2, Medical-Bios, SNLI, ECtHR-A).
- **Vision experiments**: `Vision_implementation.ipynb` runs the same comparison using a Vision Mamba (Vim-S) model on ImageNet.

The MambaLRP-IG relevance function combines the standard MambaLRP propagation rule with a path integral over `m` interpolation steps between a baseline input and the actual input (see the thesis for the full derivation, Eq. 13).

## Datasets

| Dataset | Domain | Rows | Notes |
|---|---|---|---|
| SST-2 | Sentiment (movie reviews) | 70K | From the original MambaLRP paper |
| Medical-Bios | Occupation classification | 10K | From the original MambaLRP paper |
| SNLI | Natural language inference | 570K | From the original MambaLRP paper |
| Emotion | Emotion classification (tweets) | 20K | From the original MambaLRP paper |
| ImageNet | Image classification | 1.3M | Vision Mamba (Vim-S) |
| ECtHR-A (LexGLUE) | Legal NLP, long-sequence | 11K | Added in this thesis |

## Citation

If you use this code, please cite the original MambaLRP paper this work is built on:

```bibtex
@inproceedings{jafari2024mambalrp,
    title={MambaLRP: Explaining Selective State Space Sequence Models},
    author={Farnoush Rezaei Jafari and Grégoire Montavon and Klaus-Robert Müller and Oliver Eberle},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024}
}
```

If you use the MambaLRP-IG extension or the ECtHR-A long-sequence evaluation, please also reference this thesis:

```
Bosch, A. R. (2026). MambaLRP-IG: Explaining Mamba using Layer-Wise Relevance
Propagation and Integrated Gradients. Bachelor's thesis, University of Twente.
```

## Acknowledgements

- This repository is built on top of the original [MambaLRP](https://github.com/FarnoushRJ/MambaLRP) implementation by Rezaei Jafari et al.
- It uses components from [Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/mamba), [Mamba](https://github.com/state-spaces/mamba), and [Vision Mamba](https://github.com/hustvl/Vim).

## License

This project is released under the MIT License — see [LICENSE](LICENSE) for details.
