# TODO

Idea: Convert Spleeter to CoreML (e.g. and integrate it into Mixxx or an app)

- Issue: Original TF model non-trivial to convert
- This PyTorch port looks more promising, in particular because it's simple
- However: `RuntimeError: PyTorch convert function for op 'stft' not implemented.`
  - See https://github.com/apple/coremltools/issues/1311
