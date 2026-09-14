# BRepDGT

Source release for BRepDGT, a single-stage diffusion model for generating
boundary-representation (B-rep) geometry from masked UV grids.

## Method

BRepDGT redesigns the main self-attention DiT block with two components:

- **SCEA (Shared Cross-head Efficient Attention):** combines grouped/shared
  key-value heads with learned post-softmax cross-head interaction.
- **DyT (Dynamic Tanh):** replaces LayerNorm inside the main DiT blocks with a
  learned bounded feature transformation.

The proposed components are isolated in
`src/brepdgt/models/layers/scea.py` and
`src/brepdgt/models/layers/dyt.py`, and are integrated by
`BRepDGTBlock`.

## Release scope

This is a **code-only research release**. It contains the model implementation,
geometry pipeline, and component-level tests. It intentionally does not
include:

- pretrained checkpoints;
- processed datasets or generated samples;
- paper-specific training configuration files;
- exact end-to-end training and evaluation recipes.

The omitted artifacts are not required to inspect the proposed SCEA and DyT
implementation. The repository does not claim one-command reproduction of the
paper's numerical results.

Basic environment setup is described in
[docs/installation.md](docs/installation.md). The core tests can be run with:

```bash
PYTHONPATH=src python -m unittest -v tests/test_brepdgt_layers.py
```

## Attribution

This repository is a research derivative of
[BRepDiff](https://github.com/brepdiff/brepdiff) by Mingi Lee, Dongsu Zhang,
Clément Jambon, and Young Min Kim. The inherited single-stage generation,
masked-UV-grid representation, training, evaluation, viewer, and
postprocessing code remains under its MIT license. Please cite the original
paper when using those inherited components:

```bibtex
@inproceedings{lee2025brepdiff,
  title={BRepDiff: Single-Stage B-rep Diffusion Model},
  author={Lee, Mingi and Zhang, Dongsu and Jambon, Cl{\'e}ment and Kim, Young Min},
  booktitle={SIGGRAPH Conference Papers},
  year={2025},
  doi={10.1145/3721238.3730698}
}
```

Please also cite the BRepDGT paper when its final bibliographic record is
available.

## License

Released under the MIT License. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
