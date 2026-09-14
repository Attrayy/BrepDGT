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

## Installation

Environment setup is described in
[docs/installation.md](docs/installation.md).

## Tests

Run the component tests with:

```bash
PYTHONPATH=src python -m unittest -v tests/test_brepdgt_layers.py
```

## License

Released under the MIT License. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
