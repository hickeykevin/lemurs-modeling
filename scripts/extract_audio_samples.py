#!/usr/bin/env python3
"""
Script to extract and encode audio samples from PostgreSQL database (`audio_response` table).
Saves playable audio files (.wav, .m4a, .3gp) along with a metadata CSV index.

Usage:
    python scripts/extract_audio_samples.py --limit 10 --output-dir data/audio_samples --format wav
"""

import argparse
import logging
import os
import shutil
import subprocess
import wave
import pandas as pd
from dotenv import load_dotenv

from src.utils.database_service import DatabaseService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and encode audio samples from lemurs database.")
    parser.add_argument("--output-dir", "-o", type=str, default="data/audio_samples", help="Directory to save audio files.")
    parser.add_argument("--limit", "-n", type=int, default=10, help="Number of audio samples to extract (0 for all).")
    parser.add_argument("--format", "-f", type=str, choices=["wav", "mp3", "m4a", "original"], default="wav", help="Output audio format.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate for WAV conversion (default: 16000 Hz).")
    parser.add_argument("--env-file", type=str, default=".env", help="Path to .env file for DB credentials.")
    return parser.parse_args()


def detect_audio_container(raw_bytes: bytes) -> str:
    """Detect whether raw bytes are M4A, 3GP, or unknown based on FTYP brand."""
    if len(raw_bytes) >= 12:
        brand = raw_bytes[4:12]
        if brand == b"ftypM4A ":
            return "m4a"
        elif brand == b"ftyp3gp4":
            return "3gp"
    return "bin"


def get_wav_duration(wav_path: str) -> float:
    """Read duration in seconds from WAV file header."""
    try:
        with wave.open(wav_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return round(frames / float(rate), 2)
    except Exception:
        return 0.0


def convert_audio(src_path: str, dst_path: str, target_format: str, sample_rate: int) -> bool:
    """Convert audio file using macOS afconvert or ffmpeg."""
    # Check if ffmpeg is available
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-i", src_path, "-ar", str(sample_rate), "-ac", "1", dst_path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            return True

    # Fallback to macOS native afconvert
    if shutil.which("afconvert"):
        if target_format == "wav":
            fmt_arg = "WAVE"
            desc_arg = f"LEI16@{sample_rate}"
        elif target_format == "m4a":
            fmt_arg = "m4af"
            desc_arg = "aac"
        else:
            fmt_arg = "WAVE"
            desc_arg = f"LEI16@{sample_rate}"

        cmd = ["afconvert", "-f", fmt_arg, "-d", desc_arg, src_path, dst_path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            return True
        else:
            logger.warning(f"afconvert failed for {src_path}: {res.stderr.strip()}")

    return False


def main():
    args = parse_args()
    if os.path.exists(args.env_file):
        load_dotenv(args.env_file)

    os.makedirs(args.output_dir, exist_ok=True)

    db = DatabaseService()
    logger.info(f"Connecting to database '{db.dbname}' at {db.host}:{db.port}...")
    if not db.connect():
        logger.error("Failed to connect to database.")
        return

    cursor = db.connection.cursor()

    query = """
        SELECT ar.id, ar.survey_response_id, ar.audio_question_id, q.question, 
               ar.timestamp, ar.created_at, LENGTH(ar.audio_data), ar.audio_data
        FROM audio_response ar
        LEFT JOIN question q ON ar.audio_question_id = q.id
        WHERE LENGTH(ar.audio_data) > 100
        ORDER BY ar.id
    """
    if args.limit > 0:
        query += f" LIMIT {args.limit}"

    cursor.execute(query)
    rows = cursor.fetchall()
    logger.info(f"Retrieved {len(rows)} audio records from `audio_response` table.")

    metadata_list = []

    for idx, r in enumerate(rows, 1):
        row_id, s_id, q_id, q_text, ts, created_at, size_bytes, blob = r
        raw_data = bytes(blob)
        container_fmt = detect_audio_container(raw_data)

        raw_filename = f"sample_{row_id}_q{q_id}.{container_fmt}"
        raw_filepath = os.path.join(args.output_dir, raw_filename)

        with open(raw_filepath, "wb") as f:
            f.write(raw_data)

        final_filepath = raw_filepath
        duration_sec = 0.0

        if args.format != "original" and container_fmt in ["3gp", "m4a"]:
            converted_filename = f"sample_{row_id}_q{q_id}.{args.format}"
            converted_filepath = os.path.join(args.output_dir, converted_filename)
            success = convert_audio(raw_filepath, converted_filepath, args.format, args.sample_rate)
            if success:
                final_filepath = converted_filepath
                if args.format == "wav":
                    duration_sec = get_wav_duration(final_filepath)

        logger.info(f"[{idx}/{len(rows)}] Sample ID {row_id} (Question {q_id}): format={container_fmt} -> {os.path.basename(final_filepath)} ({size_bytes / 1024:.1f} KB, duration={duration_sec}s)")

        metadata_list.append({
            "sample_id": row_id,
            "survey_response_id": s_id,
            "audio_question_id": q_id,
            "question_text": q_text,
            "timestamp": ts,
            "created_at": created_at,
            "original_format": container_fmt,
            "output_file": os.path.basename(final_filepath),
            "output_path": os.path.abspath(final_filepath),
            "raw_size_bytes": size_bytes,
            "duration_sec": duration_sec
        })

    db.disconnect()

    if metadata_list:
        csv_path = os.path.join(args.output_dir, "audio_samples_index.csv")
        df = pd.DataFrame(metadata_list)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved index metadata CSV to {csv_path}")

    logger.info(f"Done! Audio files are ready in: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
