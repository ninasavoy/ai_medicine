from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import wave
from pathlib import Path


# ------------------------------
# Editable configuration
# ------------------------------
PROCESS_ALL_FILES = True
INPUT_WAV_PATH = "data/101_1b1_Al_sc_Meditron.wav"
INPUT_DIR = "data"
RECURSIVE_SEARCH = True
MAX_WORKERS = None  
OUTPUT_DIR = "processed_audio"


def _require_module(name: str):
	try:
		return importlib.import_module(name)
	except ModuleNotFoundError as exc:
		raise ModuleNotFoundError(
			f"Missing dependency '{name}'. Install with: pip install {name}"
		) from exc


def load_wav_mono(file_path: Path):
	"""Load a WAV file and return mono samples in float32 [-1, 1] and sample rate."""
	with wave.open(str(file_path), "rb") as wav_file:
		sample_rate = wav_file.getframerate()
		num_channels = wav_file.getnchannels()
		sample_width = wav_file.getsampwidth()
		num_frames = wav_file.getnframes()
		raw_audio = wav_file.readframes(num_frames)

	np = _require_module("numpy")

	if sample_width == 1:
		audio = np.frombuffer(raw_audio, dtype=np.uint8).astype(np.float32)
		audio = (audio - 128.0) / 128.0
	elif sample_width == 2:
		audio = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
		audio = audio / 32768.0
	elif sample_width == 3:
		bytes_array = np.frombuffer(raw_audio, dtype=np.uint8).reshape(-1, 3)
		audio_int = (
			bytes_array[:, 0].astype(np.int32)
			| (bytes_array[:, 1].astype(np.int32) << 8)
			| (bytes_array[:, 2].astype(np.int32) << 16)
		)
		sign_bit = 1 << 23
		audio_int = (audio_int ^ sign_bit) - sign_bit
		audio = audio_int.astype(np.float32) / float(1 << 23)
	elif sample_width == 4:
		audio = np.frombuffer(raw_audio, dtype=np.int32).astype(np.float32)
		audio = audio / float(1 << 31)
	else:
		raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

	if num_channels > 1:
		audio = audio.reshape(-1, num_channels).mean(axis=1)

	return audio, sample_rate


def wav_to_spectrogram(input_wav: Path, output_png: Path) -> None:
	"""Generate and save a spectrogram PNG from a WAV file."""
	plt = _require_module("matplotlib.pyplot")
	samples, sample_rate = load_wav_mono(input_wav)

	output_png.parent.mkdir(parents=True, exist_ok=True)

	plt.figure(figsize=(12, 5))
	plt.specgram(samples, Fs=sample_rate, NFFT=1024, noverlap=512, cmap="magma")
	plt.title(f"Spectrogram - {input_wav.name}")
	plt.xlabel("Time (s)")
	plt.ylabel("Frequency (Hz)")
	plt.colorbar(label="Intensity (dB)")
	plt.tight_layout()
	plt.savefig(output_png, dpi=200)
	plt.close()


def _build_output_path(input_wav: Path, input_base_dir: Path, output_dir: Path) -> Path:
	"""Build output path while preserving the relative folder structure."""
	try:
		rel = input_wav.relative_to(input_base_dir)
		return output_dir / rel.parent / f"{input_wav.stem}_spectrogram.png"
	except ValueError:
		return output_dir / f"{input_wav.stem}_spectrogram.png"


def _process_one(input_wav: Path, input_base_dir: Path, output_dir: Path):
	"""Worker function for single-file spectrogram generation."""
	output_file = _build_output_path(input_wav, input_base_dir, output_dir)
	wav_to_spectrogram(input_wav, output_file)
	return input_wav, output_file


def process_many_wavs(
	input_dir: Path,
	output_dir: Path,
	recursive: bool = True,
	max_workers: int | None = None,
) -> None:
	"""Process all WAV files in a directory, optionally in parallel."""
	if not input_dir.exists() or not input_dir.is_dir():
		raise FileNotFoundError("Please set INPUT_DIR to a valid existing folder.")

	pattern = "**/*.wav" if recursive else "*.wav"
	wav_files = sorted(input_dir.glob(pattern))

	if not wav_files:
		raise FileNotFoundError(f"No .wav files found in: {input_dir.resolve()}")

	print(f"Found {len(wav_files)} WAV files.")

	if max_workers == 1 or len(wav_files) == 1:
		for wav_file in wav_files:
			_, output_file = _process_one(wav_file, input_dir, output_dir)
			print(f"Saved: {output_file.resolve()}")
		return

	workers_to_use = max_workers
	with ProcessPoolExecutor(max_workers=workers_to_use) as executor:
		futures = [
			executor.submit(_process_one, wav_file, input_dir, output_dir)
			for wav_file in wav_files
		]

		for future in as_completed(futures):
			input_file, output_file = future.result()
			print(f"Saved: {output_file.resolve()} (from {input_file.name})")


def main() -> None:
	output_dir = Path(OUTPUT_DIR)

	if PROCESS_ALL_FILES:
		process_many_wavs(
			input_dir=Path(INPUT_DIR),
			output_dir=output_dir,
			recursive=RECURSIVE_SEARCH,
			max_workers=MAX_WORKERS,
		)
		return

	input_wav = Path(INPUT_WAV_PATH)
	if not input_wav.exists() or input_wav.suffix.lower() != ".wav":
		raise FileNotFoundError(
			"Please set INPUT_WAV_PATH to a valid existing .wav file."
		)

	output_file = _build_output_path(input_wav, input_wav.parent, output_dir)
	wav_to_spectrogram(input_wav, output_file)
	print(f"Saved spectrogram to: {output_file.resolve()}")


if __name__ == "__main__":
	main()
