#!/usr/bin/env python3
"""
Script to extract audio data from PostgreSQL database and encode it into PyTorch / torchaudio
tensors (Waveform, Log-Mel Spectrogram, and MFCC) for machine learning modeling.

Usage:
    python scripts/prepare_audio_dataset.py --limit 10 --output-dir data/audio_features
"""

import argparse
import logging
import os
import pandas as pd
import torch
from dotenv import load_dotenv

from src.utils.database_service import DatabaseService
from src.data.components.audio_dataset import AudioDataset, compute_torchaudio_features, convert_raw_audio_to_pcm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Encode database audio into PyTorch ML tensors.")
    parser.add_argument("--output-dir", "-o", type=str, default="data/audio_features", help="Directory to save ML tensors and metadata.")
    parser.add_argument("--limit", "-n", type=int, default=10, help="Number of samples to process (0 for all 155).")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sampling rate in Hz.")
    parser.add_argument("--n-mels", type=int, default=80, help="Number of Mel spectrogram channels.")
    parser.add_argument("--n-mfcc", type=int, default=13, help="Number of MFCC coefficients.")
    parser.add_argument("--target-duration-sec", type=float, default=None, help="Optional fixed duration in sec for padding/cropping.")
    parser.add_argument("--env-file", type=str, default=".env", help="Path to .env file.")
    return parser.parse_args()


def main():
    args = parse_args()
    if os.path.exists(args.env_file):
        load_dotenv(args.env_file)

    tensor_dir = os.path.join(args.output_dir, "tensors")
    os.makedirs(tensor_dir, exist_ok=True)

    db = DatabaseService()
    logger.info(f"Connecting to database '{db.dbname}' at {db.host}:{db.port}...")
    if not db.connect():
        logger.error("Failed to connect to database.")
        return

    cursor = db.connection.cursor()
    query = """
        SELECT ar.id, ar.survey_response_id, ar.audio_question_id, q.question, 
               COALESCE(SUM(CAST(a.answer AS FLOAT)), 0.0) as phq9_score,
               ar.timestamp, ar.created_at, LENGTH(ar.audio_data), ar.audio_data
        FROM audio_response ar
        JOIN survey_response sr ON ar.survey_response_id = sr.id
        LEFT JOIN question q ON ar.audio_question_id = q.id
        LEFT JOIN answer a ON a.survey_response_id = sr.id
        WHERE LENGTH(ar.audio_data) > 100
        GROUP BY ar.id, ar.survey_response_id, ar.audio_question_id, q.question, ar.timestamp, ar.created_at, ar.audio_data
        ORDER BY ar.id
    """
    if args.limit > 0:
        query += f" LIMIT {args.limit}"

    cursor.execute(query)
    rows = cursor.fetchall()
    logger.info(f"Scraped {len(rows)} audio records with PHQ-9 scores from database.")

    metadata_rows = []
    manifest_dict = {}

    for idx, r in enumerate(rows, 1):
        row_id, s_id, q_id, q_text, phq9_score, ts, created_at, size_bytes, blob = r
        raw_bytes = bytes(blob)

        try:
            pcm, sr = convert_raw_audio_to_pcm(raw_bytes, target_sample_rate=args.sample_rate)
            waveform = torch.from_numpy(pcm).unsqueeze(0)

            # Optional fixed-length padding/cropping
            if args.target_duration_sec is not None:
                target_samples = int(args.sample_rate * args.target_duration_sec)
                curr_len = waveform.shape[-1]
                if curr_len < target_samples:
                    waveform = torch.nn.functional.pad(waveform, (0, target_samples - curr_len))
                elif curr_len > target_samples:
                    waveform = waveform[:, :target_samples]

            features = compute_torchaudio_features(
                waveform=waveform,
                sample_rate=sr,
                n_mels=args.n_mels,
                n_mfcc=args.n_mfcc,
            )

            duration_sec = round(waveform.shape[-1] / float(sr), 2)
            tensor_filename = f"sample_{row_id}_q{q_id}.pt"
            tensor_path = os.path.join(tensor_dir, tensor_filename)

            sample_dict = {
                "sample_id": row_id,
                "survey_response_id": s_id,
                "audio_question_id": q_id,
                "question_text": q_text,
                "timestamp": str(ts),
                "created_at": str(created_at),
                "waveform": features["waveform"],                 # [1, num_samples]
                "mel_spectrogram": features["mel_spectrogram"],  # [1, 80, time_frames]
                "mfcc": features["mfcc"],                        # [1, 13, time_frames]
                "sample_rate": sr,
                "duration_sec": duration_sec,
                "raw_size_bytes": size_bytes,
            }

            # Save individual .pt tensor file
            torch.save(sample_dict, tensor_path)
            manifest_dict[row_id] = sample_dict

            logger.info(
                f"[{idx}/{len(rows)}] Encoded Sample {row_id} (Question {q_id}): "
                f"Waveform={tuple(features['waveform'].shape)}, "
                f"LogMel={tuple(features['mel_spectrogram'].shape)}, "
                f"MFCC={tuple(features['mfcc'].shape)} -> {tensor_filename}"
            )

            metadata_rows.append({
                "sample_id": row_id,
                "survey_response_id": s_id,
                "audio_question_id": q_id,
                "question_text": q_text,
                "timestamp": ts,
                "created_at": created_at,
                "raw_size_bytes": size_bytes,
                "duration_sec": duration_sec,
                "waveform_shape": str(list(features["waveform"].shape)),
                "mel_spectrogram_shape": str(list(features["mel_spectrogram"].shape)),
                "mfcc_shape": str(list(features["mfcc"].shape)),
                "tensor_file": tensor_filename,
                "tensor_path": os.path.abspath(tensor_path),
            })

        except Exception as e:
            logger.error(f"Failed to encode sample {row_id}: {e}")

    db.disconnect()

    if metadata_rows:
        index_csv_path = os.path.join(args.output_dir, "audio_features_index.csv")
        manifest_pt_path = os.path.join(args.output_dir, "audio_dataset_manifest.pt")

        df = pd.DataFrame(metadata_rows)
        df.to_csv(index_csv_path, index=False)
        torch.save(manifest_dict, manifest_pt_path)

        logger.info(f"Saved index CSV metadata: {index_csv_path}")
        logger.info(f"Saved consolidated PyTorch dataset manifest: {manifest_pt_path}")

    logger.info(f"Successfully processed {len(metadata_rows)} audio samples into PyTorch tensors in {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
