import argparse

from pathlib import Path

from spleeter_pytorch.audio import read_audio, write_wav
from spleeter_pytorch.estimator import Estimator

ROOT = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser(description='Separate stems from an audio file.')
    parser.add_argument('-n', '--num-instruments', type=int, default=2, help='The number of stems.')
    parser.add_argument('-m', '--model', type=Path, default=ROOT / 'checkpoints' / '2stems' / 'model', help='The path to the model to use.')
    parser.add_argument('-o', '--output', type=Path, default=ROOT / 'output' / 'stems', help='The path to the output directory.')
    parser.add_argument('--torch-stft', default=True, action=argparse.BooleanOptionalAction, help="Whether to use PyTorch's native STFT.")
    parser.add_argument('input', type=Path, help='The path to the input file to process')

    args = parser.parse_args()
    estimator = Estimator(
        num_instruments=args.num_instruments,
        checkpoint_path=args.model,
        use_torch_stft=args.torch_stft,
    )
    estimator.eval()

    # Load wav audio
    input: Path = args.input
    wav, samplerate = read_audio(input)

    # Normalize audio
    # wav_torch = wav / (wav.max() + 1e-8)

    wavs = estimator.separate(wav)

    output_dir: Path = args.output / input.with_suffix('').name
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(wavs)):
        output = output_dir / f'stem-{i:02d}.wav'
        print(f'==> Writing {output}')
        write_wav(output, wav=wavs[i], samplerate=samplerate)
