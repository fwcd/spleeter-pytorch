# spleeter-pytorch

Spleeter implementation in PyTorch.

## Example

Install the package using `pip3 install .`, then run

```sh
spleeter-pytorch audio_example.mp3
```

to separate the example file. The output will be located in `output/stems`.

## Conversion to Core ML

To convert the model to Core ML, run

```sh
./convert-to-coreml
```

The `.mlpackage` will be located under `output/coreml`.

> Note: The converted model corresponds to the [`Separator`](spleeter_pytorch/separator.py) module and still requires the consumer of the model to manually perform the STFT conversion as performed in the [`Estimator`](spleeter_pytorch/estimator.py). This is due to Core ML [not supporting FFT operations yet](https://github.com/apple/coremltools/issues/1311).

## Note

* I only tested with 2stems model, not sure if it works for other models.
* There might be some bugs, the quality of output isn't as good as the original. See [output](./output) for some results. If someone found the reason, please open a pull request. Thanks.

## Reference

* [Original Spleeter](https://github.com/deezer/spleeter)
* [Original `spleeter-pytorch`](https://github.com/tuan3w/spleeter-pytorch)

## License

**MIT**.
