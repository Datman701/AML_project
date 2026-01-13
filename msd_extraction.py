#!/usr/bin/env python3
"""
MSD 10k Subset Data Extraction Pipeline

Extracts audio features, metadata, and sequence data from Million Song Dataset
.h5 files into structured CSV files for downstream analysis.

Outputs:
    1. songs_10k_features.csv - Song-level features (one row per song)
    2. artists_1500_features.csv - Artist-level aggregated features (one row per artist)
    3. segments_long.csv - Long-format segment data (one row per segment)
    4. beats_long.csv - Long-format beat data (one row per beat)
"""

import os
import sys
import csv
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import h5py
import numpy as np
import pandas as pd


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Configuration
DEFAULT_INPUT_DIR = './millionsongsubset/MillionSongSubset'
DEFAULT_OUTPUT_DIR = './output'
PROGRESS_INTERVAL = 100  # Log progress every N files


def safe_get_scalar(data_array, index=0, default=np.nan):
    """Safely extract a scalar value from an HDF5 array."""
    try:
        if len(data_array) > index:
            val = data_array[index]
            # Handle bytes (convert to string)
            if isinstance(val, bytes):
                return val.decode('utf-8', errors='ignore')
            # Handle numpy types
            if isinstance(val, (np.integer, np.floating)):
                if np.isnan(val) or np.isinf(val):
                    return default
                return float(val) if isinstance(val, np.floating) else int(val)
            return val
        return default
    except (IndexError, TypeError, ValueError):
        return default


def safe_get_array(data_array, indices, default=None):
    """Safely extract array data using indices."""
    try:
        if indices is None or len(indices) == 0:
            return default if default is not None else np.array([])

        start_idx = int(indices[0])
        end_idx = int(indices[1]) if len(indices) > 1 else len(data_array)

        return np.array(data_array[start_idx:end_idx])
    except (IndexError, TypeError, ValueError):
        return default if default is not None else np.array([])


def extract_song_features(h5_file_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract song-level features from a single .h5 file.

    Args:
        h5_file_path: Path to the .h5 file

    Returns:
        Dictionary containing all song-level features, or None if extraction fails
    """
    try:
        with h5py.File(h5_file_path, 'r') as h5:
            features = {}

            # Access data groups
            metadata_songs = h5['metadata']['songs']
            analysis_songs = h5['analysis']['songs']

            # Get indices for array data
            song_idx = 0  # Typically only one song per file

            # === IDENTIFIERS & METADATA ===
            features['track_id'] = safe_get_scalar(analysis_songs['track_id'], song_idx, '')
            features['song_id'] = safe_get_scalar(metadata_songs['song_id'], song_idx, '')
            features['artist_id'] = safe_get_scalar(metadata_songs['artist_id'], song_idx, '')
            features['artist_name'] = safe_get_scalar(metadata_songs['artist_name'], song_idx, '')
            features['title'] = safe_get_scalar(metadata_songs['title'], song_idx, '')

            # Year - handle missing/invalid years
            year = safe_get_scalar(metadata_songs['year'], song_idx, 0) if 'year' in metadata_songs.dtype.names else 0
            features['year'] = int(year) if year and not np.isnan(float(year)) else 0

            features['release'] = safe_get_scalar(metadata_songs['release'], song_idx, '')
            features['duration'] = safe_get_scalar(analysis_songs['duration'], song_idx, np.nan)
            features['analysis_sample_rate'] = safe_get_scalar(analysis_songs['analysis_sample_rate'], song_idx, np.nan)

            # === POPULARITY & EVALUATION FIELDS ===
            features['artist_familiarity'] = safe_get_scalar(metadata_songs['artist_familiarity'], song_idx, np.nan)
            features['artist_hotttnesss'] = safe_get_scalar(metadata_songs['artist_hotttnesss'], song_idx, np.nan)

            # === SCALAR AUDIO ANALYSIS FIELDS ===
            features['tempo'] = safe_get_scalar(analysis_songs['tempo'], song_idx, np.nan)
            features['time_signature'] = int(safe_get_scalar(analysis_songs['time_signature'], song_idx, 0))
            features['time_signature_confidence'] = safe_get_scalar(analysis_songs['time_signature_confidence'], song_idx, np.nan)
            features['key'] = int(safe_get_scalar(analysis_songs['key'], song_idx, -1))
            features['key_confidence'] = safe_get_scalar(analysis_songs['key_confidence'], song_idx, np.nan)
            features['mode'] = int(safe_get_scalar(analysis_songs['mode'], song_idx, -1))
            features['mode_confidence'] = safe_get_scalar(analysis_songs['mode_confidence'], song_idx, np.nan)
            features['loudness'] = safe_get_scalar(analysis_songs['loudness'], song_idx, np.nan)
            features['end_of_fade_in'] = safe_get_scalar(analysis_songs['end_of_fade_in'], song_idx, np.nan)
            features['start_of_fade_out'] = safe_get_scalar(analysis_songs['start_of_fade_out'], song_idx, np.nan)
            features['danceability'] = safe_get_scalar(analysis_songs['danceability'], song_idx, np.nan)
            features['energy'] = safe_get_scalar(analysis_songs['energy'], song_idx, np.nan)

            # Get array lengths directly from the HDF5 arrays
            # Each .h5 file contains one song, so we use the full array length
            features['num_bars'] = len(h5['analysis']['bars_start']) if 'bars_start' in h5['analysis'] else 0
            features['num_beats'] = len(h5['analysis']['beats_start']) if 'beats_start' in h5['analysis'] else 0
            features['num_sections'] = len(h5['analysis']['sections_start']) if 'sections_start' in h5['analysis'] else 0
            features['num_tatums'] = len(h5['analysis']['tatums_start']) if 'tatums_start' in h5['analysis'] else 0
            features['num_segments'] = len(h5['analysis']['segments_start']) if 'segments_start' in h5['analysis'] else 0

            # === AGGREGATED TIMBRE FEATURES ===
            # Get timbre data - each .h5 file has one song, so use all segments
            if 'segments_timbre' in h5['analysis']:
                segments_timbre = h5['analysis']['segments_timbre'][:]

                if len(segments_timbre) > 0 and segments_timbre.shape[1] == 12:
                    for i in range(1, 13):
                        col_data = segments_timbre[:, i-1]
                        features[f'timbre_mean_{i}'] = float(np.nanmean(col_data)) if len(col_data) > 0 else np.nan
                        features[f'timbre_std_{i}'] = float(np.nanstd(col_data)) if len(col_data) > 0 else np.nan
                else:
                    for i in range(1, 13):
                        features[f'timbre_mean_{i}'] = np.nan
                        features[f'timbre_std_{i}'] = np.nan
            else:
                for i in range(1, 13):
                    features[f'timbre_mean_{i}'] = np.nan
                    features[f'timbre_std_{i}'] = np.nan

            # === AGGREGATED PITCH FEATURES ===
            if 'segments_pitches' in h5['analysis']:
                segments_pitches = h5['analysis']['segments_pitches'][:]

                if len(segments_pitches) > 0 and segments_pitches.shape[1] == 12:
                    for i in range(1, 13):
                        col_data = segments_pitches[:, i-1]
                        features[f'pitch_mean_{i}'] = float(np.nanmean(col_data)) if len(col_data) > 0 else np.nan
                        features[f'pitch_std_{i}'] = float(np.nanstd(col_data)) if len(col_data) > 0 else np.nan
                else:
                    for i in range(1, 13):
                        features[f'pitch_mean_{i}'] = np.nan
                        features[f'pitch_std_{i}'] = np.nan
            else:
                for i in range(1, 13):
                    features[f'pitch_mean_{i}'] = np.nan
                    features[f'pitch_std_{i}'] = np.nan

            # === AGGREGATED LOUDNESS FEATURES ===
            if 'segments_loudness_max' in h5['analysis']:
                loudness_max = h5['analysis']['segments_loudness_max'][:]
                features['segments_loudness_max_mean'] = float(np.nanmean(loudness_max)) if len(loudness_max) > 0 else np.nan
                features['segments_loudness_max_std'] = float(np.nanstd(loudness_max)) if len(loudness_max) > 0 else np.nan
            else:
                features['segments_loudness_max_mean'] = np.nan
                features['segments_loudness_max_std'] = np.nan

            if 'segments_loudness_max_time' in h5['analysis']:
                loudness_max_time = h5['analysis']['segments_loudness_max_time'][:]
                features['segments_loudness_max_time_mean'] = float(np.nanmean(loudness_max_time)) if len(loudness_max_time) > 0 else np.nan
                features['segments_loudness_max_time_std'] = float(np.nanstd(loudness_max_time)) if len(loudness_max_time) > 0 else np.nan
            else:
                features['segments_loudness_max_time_mean'] = np.nan
                features['segments_loudness_max_time_std'] = np.nan

            if 'segments_loudness_start' in h5['analysis']:
                loudness_start = h5['analysis']['segments_loudness_start'][:]
                features['segments_loudness_max_start_mean'] = float(np.nanmean(loudness_start)) if len(loudness_start) > 0 else np.nan
                features['segments_loudness_max_start_std'] = float(np.nanstd(loudness_start)) if len(loudness_start) > 0 else np.nan
            else:
                features['segments_loudness_max_start_mean'] = np.nan
                features['segments_loudness_max_start_std'] = np.nan

            return features

    except Exception as e:
        logger.error(f"Error extracting features from {h5_file_path}: {e}")
        return None


def extract_segments(h5_file_path: str, track_id: str, artist_id: str, title: str) -> List[Dict[str, Any]]:
    """
    Extract segment-level data from a single .h5 file.

    Args:
        h5_file_path: Path to the .h5 file
        track_id: Track ID for this song
        artist_id: Artist ID for reference
        title: Song title for reference

    Returns:
        List of dictionaries, one per segment
    """
    segments = []

    try:
        with h5py.File(h5_file_path, 'r') as h5:
            # Each .h5 file contains one song, so we use all the data in the arrays
            if 'segments_start' not in h5['analysis']:
                return segments

            segments_start = h5['analysis']['segments_start'][:]
            num_segments = len(segments_start)

            if num_segments == 0:
                return segments

            # Get confidence if available
            segments_confidence = h5['analysis']['segments_confidence'][:] if 'segments_confidence' in h5['analysis'] else np.full(num_segments, np.nan)

            # Get timbre
            segments_timbre = h5['analysis']['segments_timbre'][:] if 'segments_timbre' in h5['analysis'] else np.full((num_segments, 12), np.nan)

            # Get pitches
            segments_pitches = h5['analysis']['segments_pitches'][:] if 'segments_pitches' in h5['analysis'] else np.full((num_segments, 12), np.nan)

            # Get loudness data
            segments_loudness_max = h5['analysis']['segments_loudness_max'][:] if 'segments_loudness_max' in h5['analysis'] else np.full(num_segments, np.nan)
            segments_loudness_max_time = h5['analysis']['segments_loudness_max_time'][:] if 'segments_loudness_max_time' in h5['analysis'] else np.full(num_segments, np.nan)
            segments_loudness_start = h5['analysis']['segments_loudness_start'][:] if 'segments_loudness_start' in h5['analysis'] else np.full(num_segments, np.nan)

            # Build segment records
            for i in range(num_segments):
                segment = {
                    'track_id': track_id,
                    'segment_index': i,
                    'song_artist_id': artist_id,
                    'song_title': title,
                    'segment_start': float(segments_start[i]),
                    'segment_duration': float(segments_start[i+1] - segments_start[i]) if i < num_segments - 1 else np.nan,
                    'segment_confidence': float(segments_confidence[i]) if i < len(segments_confidence) else np.nan,
                }

                # Add timbre coefficients
                if i < len(segments_timbre) and segments_timbre.shape[1] == 12:
                    for j in range(12):
                        segment[f'timbre_{j+1}'] = float(segments_timbre[i, j])
                else:
                    for j in range(12):
                        segment[f'timbre_{j+1}'] = np.nan

                # Add pitch coefficients
                if i < len(segments_pitches) and segments_pitches.shape[1] == 12:
                    for j in range(12):
                        segment[f'pitch_{j+1}'] = float(segments_pitches[i, j])
                else:
                    for j in range(12):
                        segment[f'pitch_{j+1}'] = np.nan

                # Add loudness envelope
                segment['loudness_max'] = float(segments_loudness_max[i]) if i < len(segments_loudness_max) else np.nan
                segment['loudness_max_time'] = float(segments_loudness_max_time[i]) if i < len(segments_loudness_max_time) else np.nan
                segment['loudness_max_start'] = float(segments_loudness_start[i]) if i < len(segments_loudness_start) else np.nan

                segments.append(segment)

    except Exception as e:
        logger.error(f"Error extracting segments from {h5_file_path}: {e}")

    return segments


def extract_beats(h5_file_path: str, track_id: str, artist_id: str, title: str) -> List[Dict[str, Any]]:
    """
    Extract beat-level data from a single .h5 file.

    Args:
        h5_file_path: Path to the .h5 file
        track_id: Track ID for this song
        artist_id: Artist ID for reference
        title: Song title for reference

    Returns:
        List of dictionaries, one per beat
    """
    beats = []

    try:
        with h5py.File(h5_file_path, 'r') as h5:
            # Each .h5 file contains one song, so we use all the data in the arrays
            if 'beats_start' not in h5['analysis']:
                return beats

            beats_start = h5['analysis']['beats_start'][:]
            num_beats = len(beats_start)

            if num_beats == 0:
                return beats

            # Get confidence if available
            beats_confidence = h5['analysis']['beats_confidence'][:] if 'beats_confidence' in h5['analysis'] else np.full(num_beats, np.nan)

            # Build beat records
            for i in range(num_beats):
                beat = {
                    'track_id': track_id,
                    'beat_index': i,
                    'song_artist_id': artist_id,
                    'song_title': title,
                    'beat_start': float(beats_start[i]),
                    'beat_duration': float(beats_start[i+1] - beats_start[i]) if i < num_beats - 1 else np.nan,
                    'beat_confidence': float(beats_confidence[i]) if i < len(beats_confidence) else np.nan,
                }
                beats.append(beat)

    except Exception as e:
        logger.error(f"Error extracting beats from {h5_file_path}: {e}")

    return beats


def process_dataset(input_dir: str, output_dir: str):
    """
    Main processing loop: walk directory, extract features, write CSVs.

    Args:
        input_dir: Root directory of MSD subset
        output_dir: Directory to save output CSVs
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize CSV writers
    songs_csv_path = output_path / 'songs_10k_features.csv'
    segments_csv_path = output_path / 'segments_long.csv'
    beats_csv_path = output_path / 'beats_long.csv'

    # Collect all .h5 files
    logger.info(f"Scanning for .h5 files in {input_dir}...")
    h5_files = list(Path(input_dir).rglob('*.h5'))
    logger.info(f"Found {len(h5_files)} .h5 files")

    if len(h5_files) == 0:
        logger.error("No .h5 files found. Check the input directory path.")
        return

    # Statistics
    songs_processed = 0
    songs_failed = 0
    total_segments = 0
    total_beats = 0

    # Process first file to get column names
    first_features = None
    for h5_file in h5_files:
        first_features = extract_song_features(str(h5_file))
        if first_features:
            break

    if not first_features:
        logger.error("Could not extract features from any file to determine column structure")
        return

    songs_columns = list(first_features.keys())

    # Open CSV files for writing
    with open(songs_csv_path, 'w', newline='', encoding='utf-8') as songs_file, \
         open(segments_csv_path, 'w', newline='', encoding='utf-8') as segments_file, \
         open(beats_csv_path, 'w', newline='', encoding='utf-8') as beats_file:

        songs_writer = csv.DictWriter(songs_file, fieldnames=songs_columns)
        songs_writer.writeheader()

        # We'll write segment/beat headers after processing first file
        segments_writer = None
        beats_writer = None

        # Process each .h5 file
        for idx, h5_file in enumerate(h5_files, 1):
            try:
                # Extract song features
                features = extract_song_features(str(h5_file))

                if features is None:
                    songs_failed += 1
                    continue

                # Write song features
                songs_writer.writerow(features)
                songs_processed += 1

                track_id = features.get('track_id', '')
                artist_id = features.get('artist_id', '')
                title = features.get('title', '')

                # Extract and write segments
                segments = extract_segments(str(h5_file), track_id, artist_id, title)
                if segments:
                    if segments_writer is None:
                        # Initialize segments CSV with column names from first segment
                        segments_columns = list(segments[0].keys())
                        segments_writer = csv.DictWriter(segments_file, fieldnames=segments_columns)
                        segments_writer.writeheader()

                    for segment in segments:
                        segments_writer.writerow(segment)
                    total_segments += len(segments)

                # Extract and write beats
                beats = extract_beats(str(h5_file), track_id, artist_id, title)
                if beats:
                    if beats_writer is None:
                        # Initialize beats CSV with column names from first beat
                        beats_columns = list(beats[0].keys())
                        beats_writer = csv.DictWriter(beats_file, fieldnames=beats_columns)
                        beats_writer.writeheader()

                    for beat in beats:
                        beats_writer.writerow(beat)
                    total_beats += len(beats)

                # Log progress
                if idx % PROGRESS_INTERVAL == 0:
                    logger.info(f"Processed {idx}/{len(h5_files)} files ({songs_processed} songs, "
                               f"{total_segments} segments, {total_beats} beats)")

            except Exception as e:
                logger.error(f"Unexpected error processing {h5_file}: {e}")
                songs_failed += 1

    logger.info(f"\n=== Song-level extraction complete ===")
    logger.info(f"Songs processed: {songs_processed}")
    logger.info(f"Songs failed: {songs_failed}")
    logger.info(f"Total segments: {total_segments}")
    logger.info(f"Total beats: {total_beats}")
    logger.info(f"Songs CSV saved to: {songs_csv_path}")
    logger.info(f"Segments CSV saved to: {segments_csv_path}")
    logger.info(f"Beats CSV saved to: {beats_csv_path}")

    # Now build artist-level aggregation
    build_artist_features(songs_csv_path, output_path / 'artists_1500_features.csv')


def build_artist_features(songs_csv_path: Path, output_csv_path: Path):
    """
    Build artist-level aggregated features from songs CSV.

    Args:
        songs_csv_path: Path to songs_10k_features.csv
        output_csv_path: Path to save artists_1500_features.csv
    """
    logger.info("\n=== Building artist-level features ===")

    try:
        # Load songs data
        songs_df = pd.read_csv(songs_csv_path)
        logger.info(f"Loaded {len(songs_df)} songs")

        # Group by artist
        artist_groups = songs_df.groupby('artist_id')

        artist_features = []

        for artist_id, group in artist_groups:
            artist_feat = {
                'artist_id': artist_id,
                'artist_name': group['artist_name'].iloc[0],
                'song_count': len(group)
            }

            # Popularity fields (mean across songs)
            artist_feat['artist_familiarity_mean'] = group['artist_familiarity'].mean()
            artist_feat['artist_hotttnesss_mean'] = group['artist_hotttnesss'].mean()

            # Aggregate all audio features
            # Identify numeric columns (exclude IDs, names, etc.)
            numeric_cols = songs_df.select_dtypes(include=[np.number]).columns
            exclude_cols = ['year']  # Year doesn't make sense to aggregate

            for col in numeric_cols:
                if col in exclude_cols or 'familiarity' in col or 'hotttnesss' in col:
                    continue

                # Calculate mean and std for each feature
                artist_feat[f'artist_{col}_mean'] = group[col].mean()
                artist_feat[f'artist_{col}_std'] = group[col].std()

            artist_features.append(artist_feat)

        # Create DataFrame and save
        artists_df = pd.DataFrame(artist_features)
        artists_df.to_csv(output_csv_path, index=False)

        logger.info(f"Artist-level features complete")
        logger.info(f"Unique artists: {len(artists_df)}")
        logger.info(f"Artists CSV saved to: {output_csv_path}")
        logger.info(f"Artists CSV shape: {artists_df.shape}")

    except Exception as e:
        logger.error(f"Error building artist features: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Extract features from Million Song Dataset .h5 files'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default=DEFAULT_INPUT_DIR,
        help=f'Root directory of MSD subset (default: {DEFAULT_INPUT_DIR})'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory for CSV files (default: {DEFAULT_OUTPUT_DIR})'
    )

    args = parser.parse_args()

    # Validate input directory
    if not Path(args.input_dir).exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    logger.info(f"Starting MSD extraction pipeline")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    # Run processing
    process_dataset(args.input_dir, args.output_dir)

    logger.info("\n=== Extraction pipeline complete ===")


if __name__ == '__main__':
    main()
