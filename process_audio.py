from __future__ import annotations

import csv
from dataclasses import dataclass
import importlib
from pathlib import Path
import re
from typing import Iterable


INPUT_DIR = "data"
OUTPUT_DIR = "processed_audio"
METADATA_FILE = "processed_audio/metadata.csv"
RECURSIVE_SEARCH = True


@dataclass(frozen=True)
class MelPipelineConfig:
	"""Configuration for Mel-spectrogram dataset generation."""

	sampling_rate: int = 22050
	target_duration_seconds: float = 5.0
	n_fft: int = 1024
	hop_length: int = 512
	n_mels: int = 128
	fmin: int = 50
	fmax: int = 4000
	target_shape: tuple[int, int] = (128, 128)

	@property
	def target_num_samples(self) -> int:
		return int(self.sampling_rate * self.target_duration_seconds)


CONFIG = MelPipelineConfig()


def _require_module(name: str):
	try:
		return importlib.import_module(name)
	except ModuleNotFoundError as exc:
		raise ModuleNotFoundError(
			f"Missing dependency '{name}'. Install with: pip install {name}"
		) from exc


def _load_audio_mono_resampled(file_path: Path, sr: int):
	"""Load audio as mono float32 in range [-1, 1] at a fixed sample rate."""
	librosa = _require_module("librosa")
	np = _require_module("numpy")
	audio, _ = librosa.load(str(file_path), sr=sr, mono=True)
	return np.asarray(audio, dtype=np.float32)


def _read_cycles(annotation_path: Path) -> list[tuple[float, float, int, int]]:
	"""Read cycle annotations: start, end, crackles, wheezes."""
	cycles: list[tuple[float, float, int, int]] = []

	for line_number, raw_line in enumerate(annotation_path.read_text().splitlines(), start=1):
		line = raw_line.strip()
		if not line:
			continue

		parts = re.split(r"[\s,]+", line)
		if len(parts) < 4:
			print(
				f"Warning: malformed annotation at {annotation_path.name}:{line_number}. "
				"Expected at least 4 columns."
			)
			continue

		try:
			start = float(parts[0])
			end = float(parts[1])
			crackles = 1 if int(float(parts[2])) > 0 else 0
			wheezes = 1 if int(float(parts[3])) > 0 else 0
		except ValueError:
			print(
				f"Warning: invalid numeric values at {annotation_path.name}:{line_number}."
			)
			continue

		if end <= start:
			print(
				f"Warning: invalid cycle interval at {annotation_path.name}:{line_number} "
				f"(start={start}, end={end})."
			)
			continue

		cycles.append((start, end, crackles, wheezes))

	return cycles


def _normalize_audio_amplitude(audio):
	"""Peak-normalize audio to [-1, 1]."""
	np = _require_module("numpy")
	peak = float(np.max(np.abs(audio))) if audio.size else 0.0
	if peak > 0:
		audio = audio / peak
	return np.clip(audio, -1.0, 1.0)


def _pad_or_truncate(audio, target_samples: int):
	"""Force fixed-length waveform using trim or zero-padding."""
	np = _require_module("numpy")
	if audio.shape[0] >= target_samples:
		return audio[:target_samples]

	padding = target_samples - audio.shape[0]
	return np.pad(audio, (0, padding), mode="constant")


def _to_mel_spectrogram(audio, config: MelPipelineConfig):
	"""Generate normalized Mel-spectrogram with fixed shape [128, 128]."""
	librosa = _require_module("librosa")
	np = _require_module("numpy")

	mel = librosa.feature.melspectrogram(
		y=audio,
		sr=config.sampling_rate,
		n_fft=config.n_fft,
		hop_length=config.hop_length,
		n_mels=config.n_mels,
		fmin=config.fmin,
		fmax=config.fmax,
	)
	mel_db = librosa.power_to_db(mel, ref=np.max)

	mel_min = float(mel_db.min())
	mel_max = float(mel_db.max())
	if mel_max > mel_min:
		mel_norm = (mel_db - mel_min) / (mel_max - mel_min)
	else:
		mel_norm = np.zeros_like(mel_db, dtype=np.float32)

	target_h, target_w = config.target_shape
	mel_norm = librosa.util.fix_length(mel_norm, size=target_h, axis=0)
	mel_norm = librosa.util.fix_length(mel_norm, size=target_w, axis=1)

	return mel_norm.astype(np.float32)


def _get_patient_id(file_stem: str) -> str:
	"""Extract patient ID from filename stem (e.g., '101_1b1_...')."""
	return file_stem.split("_", maxsplit=1)[0]


def _find_audio_annotation_pairs(input_dir: Path, recursive: bool) -> list[tuple[Path, Path]]:
	"""Match .wav and .txt files by identical stem and directory."""
	if not input_dir.exists() or not input_dir.is_dir():
		raise FileNotFoundError("Please set INPUT_DIR to a valid existing folder.")

	pattern = "**/*.wav" if recursive else "*.wav"
	wav_files = sorted(input_dir.glob(pattern))
	if not wav_files:
		raise FileNotFoundError(f"No .wav files found in: {input_dir.resolve()}")

	pairs: list[tuple[Path, Path]] = []
	missing_annotations: list[Path] = []

	for wav_path in wav_files:
		annotation_path = wav_path.with_suffix(".txt")
		if annotation_path.exists():
			pairs.append((wav_path, annotation_path))
		else:
			missing_annotations.append(wav_path)

	if missing_annotations:
		print(f"Warning: {len(missing_annotations)} .wav files without matching .txt annotations.")

	if not pairs:
		raise FileNotFoundError("No valid .wav/.txt pairs found.")

	return pairs


def _extract_cycle(audio, sr: int, start_time: float, end_time: float):
	"""Extract cycle waveform from start/end times (seconds)."""
	start_idx = max(0, int(round(start_time * sr)))
	end_idx = min(audio.shape[0], int(round(end_time * sr)))
	if end_idx <= start_idx:
		return audio[:0]
	return audio[start_idx:end_idx]


def _relative_output_path(path: Path) -> str:
	"""Return a clean relative-like path for metadata CSV."""
	return path.as_posix()


def _process_single_pair(
	wav_path: Path,
	annotation_path: Path,
	output_dir: Path,
	config: MelPipelineConfig,
) -> list[dict[str, str | int | float]]:
	"""Process one audio/annotation pair into cycle-level .npy Mel-spectrograms."""
	np = _require_module("numpy")
	audio = _load_audio_mono_resampled(wav_path, sr=config.sampling_rate)
	cycles = _read_cycles(annotation_path)

	rows: list[dict[str, str | int | float]] = []
	original_file = wav_path.stem
	patient_id = _get_patient_id(original_file)
	patient_output_dir = output_dir / patient_id
	patient_output_dir.mkdir(parents=True, exist_ok=True)

	for cycle_index, (start_time, end_time, crackles, wheezes) in enumerate(cycles):
		cycle_audio = _extract_cycle(audio, config.sampling_rate, start_time, end_time)
		if cycle_audio.size == 0:
			print(
				f"Warning: empty cycle skipped for {wav_path.name} "
				f"(cycle_index={cycle_index})."
			)
			continue

		cycle_audio = _normalize_audio_amplitude(cycle_audio)
		cycle_audio = _pad_or_truncate(cycle_audio, config.target_num_samples)

		mel = _to_mel_spectrogram(cycle_audio, config=config)
		output_file = patient_output_dir / f"{original_file}_cycle_{cycle_index}.npy"
		np.save(output_file, mel)

		rows.append(
			{
				"patient_id": patient_id,
				"original_file": original_file,
				"cycle_index": cycle_index,
				"spectrogram_path": _relative_output_path(output_file),
				"start_time": start_time,
				"end_time": end_time,
				"duration": end_time - start_time,
				"crackles": crackles,
				"wheezes": wheezes,
			}
		)

	return rows


def _write_metadata_csv(rows: Iterable[dict[str, str | int | float]], metadata_file: Path) -> None:
	"""Write cycle-level metadata CSV for training."""
	metadata_file.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = [
		"patient_id",
		"original_file",
		"cycle_index",
		"spectrogram_path",
		"start_time",
		"end_time",
		"duration",
		"crackles",
		"wheezes",
	]

	with metadata_file.open("w", newline="", encoding="utf-8") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)


def process_dataset(
	input_dir: Path,
	output_dir: Path,
	metadata_file: Path,
	recursive: bool = True,
	config: MelPipelineConfig = CONFIG,
) -> None:
	"""
	Generate fixed-size Mel spectrograms per respiratory cycle and metadata CSV.

	Consistency guarantees:
	- Single sample rate for all files
	- Fixed cycle duration
	- Fixed Mel configuration
	- Fixed output shape [128, 128]
	"""
	pairs = _find_audio_annotation_pairs(input_dir=input_dir, recursive=recursive)
	print(f"Found {len(pairs)} valid .wav/.txt pairs.")

	all_rows: list[dict[str, str | int | float]] = []
	for index, (wav_path, annotation_path) in enumerate(pairs, start=1):
		rows = _process_single_pair(
			wav_path=wav_path,
			annotation_path=annotation_path,
			output_dir=output_dir,
			config=config,
		)
		all_rows.extend(rows)
		print(
			f"[{index}/{len(pairs)}] Processed {wav_path.name} -> "
			f"{len(rows)} cycles"
		)

	if not all_rows:
		raise RuntimeError("No cycles were generated. Please verify annotation files.")

	_write_metadata_csv(all_rows, metadata_file)
	unique_patients = len({str(row["patient_id"]) for row in all_rows})
	print(
		f"Done. Generated {len(all_rows)} spectrograms for {unique_patients} patients. "
		f"Metadata saved to: {metadata_file.resolve()}"
	)


def main() -> None:
	process_dataset(
		input_dir=Path(INPUT_DIR),
		output_dir=Path(OUTPUT_DIR),
		metadata_file=Path(METADATA_FILE),
		recursive=RECURSIVE_SEARCH,
	)


if __name__ == "__main__":
	main()
