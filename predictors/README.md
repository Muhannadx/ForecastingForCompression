# predictors

Scripts to run each predictor and the shared error-bounded back-end on ERA5.

## Planned contents

```
predictors/
├── sz3/run_sz3.py       SZ3.1 baseline, single spatial pass
├── cra5/run_cra5.py     CRA5 reconstruction as prediction, single pass
├── graphcast/run_graphcast.py   GraphCast in the autoregressive loop
└── aurora/run_aurora.py         Aurora in the autoregressive loop
```

## How the methods differ

- **SZ3.1** and **CRA5** are single pass. They form a prediction for each field
  independently and hand the residual to the back-end. CRA5 supplies its decoded
  reconstruction as the prediction; the back-end stores the residual CRA5 itself
  discards, which turns an unbounded codec into an error-bounded one.
- **GraphCast** and **Aurora** are temporal. They predict the next state from the two
  preceding states, so at decode time the inputs are the corrected states, not the
  truth. Both seed with two lossless states, then at each step `p_k = F(p'_{k-2}, p'_{k-1})`, correct the residual under `epsilon`, and feed the corrected state forward. 
  Correcting at every step re-anchors the rollout and
  keeps each model input within `epsilon * r_X` of the truth.

## Upstream dependencies

The learned predictors are run in inference mode with released weights. We do not
vendor them; install each from its upstream project:

- SZ3.1: https://github.com/szcompressor/SZ3
- CRA5 (VAEformer): https://github.com/taohan10200/CRA5
- GraphCast: https://github.com/google-deepmind/graphcast
- Aurora: https://github.com/microsoft/aurora

GraphCast and Aurora need a deterministic configuration for the autoregressive loop:
the decoder re-runs the model to reproduce each prediction, so encoder and decoder
must compute the same values bit for bit.
