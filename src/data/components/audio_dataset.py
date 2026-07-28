import os
import io
import subprocess
import shutil
import tempfile
import wave
from typing import Dict, List, Optional, Any, Tuple, Union

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

import torchaudio
import torchaudio.transforms as T
from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__)


def convert_raw_audio_to_pcm(raw_bytes: bytes, target_sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """Convert raw 3GP / M4A binary bytes to float32 PCM numpy array.

    Uses PyAV (FFmpeg bindings) for in-memory cross-platform decoding,
    falling back to system ffmpeg or macOS afconvert if necessary.

    Args:
        raw_bytes: Raw binary bytes from database bytea column.
        target_sample_rate: Desired sample rate in Hz.

    Returns:
        Tuple of (pcm_array float32 normalized in [-1.0, 1.0], target_sample_rate)
    """
    # 1. Try PyAV (FFmpeg C-bindings) - Cross-platform in-memory decoding
    try:
        import av
        container = av.open(io.BytesIO(raw_bytes))
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=target_sample_rate)
        pcm_frames = []
        for frame in container.decode(stream):
            resampled_frames = resampler.resample(frame)
            for rf in resampled_frames:
                pcm_frames.append(rf.to_ndarray())
        if pcm_frames:
            pcm_array = np.concatenate(pcm_frames, axis=1).squeeze(0)
            return pcm_array, target_sample_rate
    except Exception as e:
        logger.debug(f"PyAV in-memory decoding skipped/failed: {e}")

    # 2. Fallback to system ffmpeg or afconvert via tempfile
    with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as src_f, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as dst_f:
        src_path = src_f.name
        dst_path = dst_f.name
        src_f.write(raw_bytes)

    try:
        converted = False
        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-y", "-i", src_path, "-ar", str(target_sample_rate), "-ac", "1", dst_path]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                converted = True

        if not converted and shutil.which("afconvert"):
            cmd = ["afconvert", "-f", "WAVE", "-d", f"LEI16@{target_sample_rate}", src_path, dst_path]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                converted = True

        if not converted:
            raise RuntimeError("Failed to decode audio bytes: PyAV, ffmpeg, and afconvert all failed.")

        with wave.open(dst_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)
            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if n_channels > 1:
                audio_data = audio_data.reshape(-1, n_channels).mean(axis=1)
        return audio_data, sr

    finally:
        for p in [src_path, dst_path]:
            if os.path.exists(p):
                os.remove(p)


def compute_torchaudio_features(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_mfcc: int = 13,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> Dict[str, torch.Tensor]:
    """Compute standard PyTorch / torchaudio feature representations.

    Args:
        waveform: Float tensor of shape [1, num_samples] or [num_samples].
        sample_rate: Audio sampling rate in Hz.
        n_mels: Number of Mel filterbank bins.
        n_mfcc: Number of MFCC coefficients.
        n_fft: FFT window size.
        hop_length: Hop length between frames.

    Returns:
        Dict containing:
            - 'waveform': Tensor [1, num_samples]
            - 'mel_spectrogram': Log-mel spectrogram tensor [1, n_mels, time_frames]
            - 'mfcc': MFCC tensor [1, n_mfcc, time_frames]
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # 1. Mel Spectrogram (Log-scale)
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel_spec = mel_transform(waveform)
    log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-9))

    # 2. MFCC
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": n_fft, "hop_length": hop_length, "n_mels": n_mels},
    )
    mfcc = mfcc_transform(waveform)

    return {
        "waveform": waveform,
        "mel_spectrogram": log_mel_spec,
        "mfcc": mfcc,
    }


class AudioDataset(Dataset):
    """PyTorch Dataset for audio survey responses.

    Supports loading audio features (raw waveform, Log-Mel Spectrogram, and MFCCs)
    computed via ``torchaudio``.

    Attributes:
        metadata (pd.DataFrame): Metadata dataframe containing sample records.
        sample_rate (int): Target sampling rate (default 16000 Hz).
        n_mels (int): Number of Mel frequency bins.
        n_mfcc (int): Number of MFCC features.
        target_duration_sec (Optional[float]): Optional target duration in seconds for fixed-length padding/cropping.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_mfcc: int = 13,
        n_fft: int = 1024,
        hop_length: int = 512,
        target_duration_sec: Optional[float] = None,
    ) -> None:
        """Initializes the AudioDataset.

        Args:
            metadata: DataFrame containing columns 'sample_id', 'audio_data' (bytes) or 'file_path'.
            sample_rate: Target sample rate in Hz.
            n_mels: Mel spectrogram resolution.
            n_mfcc: Number of MFCCs.
            n_fft: FFT window size.
            hop_length: Hop length for STFT.
            target_duration_sec: If set, pads or crops waveforms to fixed duration (sec).
        """
        self.metadata = metadata.reset_index(drop=True)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_duration_sec = target_duration_sec
        self.target_num_samples = int(sample_rate * target_duration_sec) if target_duration_sec else None

    def __len__(self) -> int:
        return len(self.metadata)

    def _load_waveform(self, row: pd.Series) -> Tuple[torch.Tensor, int]:
        if "audio_data" in row and isinstance(row["audio_data"], (bytes, bytearray, memoryview)):
            pcm, sr = convert_raw_audio_to_pcm(bytes(row["audio_data"]), target_sample_rate=self.sample_rate)
            waveform = torch.from_numpy(pcm).unsqueeze(0)
            return waveform, sr
        elif "output_path" in row and os.path.exists(row["output_path"]):
            with wave.open(row["output_path"], "rb") as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                frames = wf.readframes(n_frames)
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                waveform = torch.from_numpy(audio_data).unsqueeze(0)
                if sr != self.sample_rate:
                    resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
                    waveform = resampler(waveform)
                    sr = self.sample_rate
                return waveform, sr
        else:
            raise ValueError(f"Record {row.get('sample_id')} has no valid 'audio_data' bytes or 'output_path' file.")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.metadata.iloc[idx]
        waveform, sr = self._load_waveform(row)

        # Optional fixed-length padding or cropping
        if self.target_num_samples is not None:
            curr_len = waveform.shape[-1]
            if curr_len < self.target_num_samples:
                pad_len = self.target_num_samples - curr_len
                waveform = torch.nn.functional.pad(waveform, (0, pad_len))
            elif curr_len > self.target_num_samples:
                waveform = waveform[:, :self.target_num_samples]

        features = compute_torchaudio_features(
            waveform=waveform,
            sample_rate=sr,
            n_mels=self.n_mels,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        sample_id = int(row.get("sample_id", idx))
        question_id = int(row.get("audio_question_id", -1))
        survey_resp_id = int(row.get("survey_response_id", -1))
        phq9_score = float(row.get("phq9_score", 0.0)) if pd.notnull(row.get("phq9_score")) else 0.0

        return {
            "sample_id": sample_id,
            "survey_response_id": survey_resp_id,
            "audio_question_id": question_id,
            "question_text": question_text,
            "phq9_score": phq9_score,
            "target": torch.tensor(phq9_score, dtype=torch.float32),
            "waveform": features["waveform"],                 # Tensor [1, T]
            "mel_spectrogram": features["mel_spectrogram"],  # Tensor [1, 80, F]
            "mfcc": features["mfcc"],                        # Tensor [1, 13, F]
            "num_samples": waveform.shape[-1],
            "duration_sec": round(waveform.shape[-1] / float(sr), 2),
            "sample_rate": sr,
        }
