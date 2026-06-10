# Can Deep Neural Networks Improve Compression of Very Large Scientific Data?

Companion artifact for the Can Deep Neural Networks Improve Compression of Very Large Scientific Data? paper. 
The paper asks one question: does a strong machine learning predictor compress error-bounded scientific data better than the
spatial predictor of SZ3.1? 
We answer it on ERA5 climate reanalysis with three
pretrained learned predictors, CRA5, GraphCast, and Aurora, against the SZ3.1
baseline.

This repository holds the code and data recipes and the appendix.

## Framework in one paragraph

All four methods share the same prediction-residual pipeline. A predictor forms an
estimate of each field, the back-end stores the residual against the ground truth
under a relative error bound `epsilon`, and an entropy coder compresses the quantized
residual. Only the predictor changes, so any difference in compression ratio comes
from the predictor alone. CRA5 reconstructs each field on its own (a learned spatial
predictor, single pass). GraphCast and Aurora forecast the next state from past states
(temporal predictors), so we run them in an autoregressive loop that feeds back the
error-corrected state and keeps the bound at every step.

## Repository structure

```
FC-compression/
├── predictors/    scripts to run prediction + error-bounded compression on ERA5
│                  (shared back-end, SZ3.1 baseline, CRA5, GraphCast, Aurora)
├── datasets/      recipes to download ERA5 and the compressed CRA5 binaries
└── results/       extended results appendix
```

See the `README.md` in each directory for what it contains.

## General Info

- Data: ERA5 reanalysis (ECMWF), 0.25 degree, 721 x 1440 grid, sampled every 6 hours.
- Variables: the method share 9 common variables, 4 surface and 5 atmospheric, reported per
  variable and per pressure level. However, some methods cover more variables\levels than others.
- Error bounds: relative `epsilon` in {1e-2, 1e-3, 1e-4}.
- Predictors: CRA5, GraphCast, Aurora, with SZ3.1 as the classical baseline.
