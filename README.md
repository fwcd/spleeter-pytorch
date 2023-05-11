# spleeter-pytorch

A small implementation of the [Spleeter](https://github.com/deezer/spleeter) stem separation model in PyTorch. Using this model, audio files can be demixed into vocals, instrumentation etc.

## Example

Install the package using `pip3 install .`, then run

```sh
spleeter-pytorch audio_example.mp3
```

to separate the example file. The output will be located in `output/stems`.

## Conversion to Core ML

The non-FFT parts of the Spleeter model can be converted to Core ML, for efficient inference on macOS/iOS devices. To perform the conversion, run

```sh
./convert-to-coreml
```

The `.mlpackage` will be located under `output/coreml`.

> Note: The converted model corresponds to the [`Separator`](spleeter_pytorch/separator.py) module and still requires the consumer of the model to manually perform the STFT conversion as performed in the [`Estimator`](spleeter_pytorch/estimator.py). This is due to Core ML [not supporting FFT operations yet](https://github.com/apple/coremltools/issues/1311).

## Note

* Currently this is only tested with the 2stems model. Feel free to [get one of the other models](https://github.com/deezer/spleeter/releases/tag/v1.4.0) and test it on them.
* There might be some bugs, the quality of output isn't as good as the original. If someone found the reason, please open a pull request. Thanks.

## Reference

* [Original Spleeter](https://github.com/deezer/spleeter) by [`deezer`](https://github.com/deezer)
* [Original `spleeter-pytorch`](https://github.com/tuan3w/spleeter-pytorch) by [`tuan3w`](https://github.com/tuan3w)

## License

**MIT**.
