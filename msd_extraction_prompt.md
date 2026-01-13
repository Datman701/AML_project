# MSD 10k Subset Data Extraction Prompt

You are tasked with creating a Python extraction pipeline that reads all .h5 files from the Million Song Dataset (MSD) 10k subset and produces multiple CSV files for downstream analysis.

## Objective

Extract all relevant audio features, metadata, and sequence data from MSD .h5 files into structured CSV files, optimized for artist-level clustering and detailed audio analysis.

## Output Requirements

Generate a single Python file (`msd_extraction.py`) that produces **4 CSV files**:

1. **songs_10k_features.csv** – Song-level features (one row per song)
2. **artists_1500_features.csv** – Artist-level aggregated features (one row per artist)
3. **segments_long.csv** – Long-format segment data (one row per segment)
4. **beats_long.csv** – Long-format beat data (one row per beat)

---

## CSV 1: songs_10k_features.csv

**Granularity:** One row per song (per .h5 file)

### Columns to include:

#### Identifiers & Metadata (9 columns)
- `track_id` (str)
- `song_id` (str)
- `artist_id` (str)
- `artist_name` (str)
- `title` (str)
- `year` (int, use 0 if missing)
- `release` (str, album name)
- `duration` (float, in seconds)
- `analysis_sample_rate` (float)

#### Popularity & Evaluation Fields (2 columns) – **NOT used as clustering input, only for later analysis**
- `artist_familiarity` (float, 0-1, NaN if missing)
- `artist_hotttnesss` (float, 0-1, NaN if missing)

#### Scalar Audio Analysis Fields (16 columns)
- `tempo` (float, BPM)
- `time_signature` (int)
- `time_signature_confidence` (float)
- `key` (int, 0-11 or -1 if unknown)
- `key_confidence` (float)
- `mode` (int, 0=minor, 1=major, -1 if unknown)
- `mode_confidence` (float)
- `loudness` (float, in dB)
- `end_of_fade_in` (float, seconds)
- `start_of_fade_out` (float, seconds)
- `danceability` (float, 0-1, NaN if missing)
- `energy` (float, 0-1, NaN if missing)
- `num_bars` (int, length of bars_start array)
- `num_beats` (int, length of beats_start array)
- `num_sections` (int, length of sections_start array)
- `num_tatums` (int, length of tatums_start array)
- `num_segments` (int, length of segments_start array)

#### Aggregated Timbre Features (24 columns)
From `segments_timbre` (shape: num_segments × 12):
- For each coefficient i (1-12):
  - `timbre_mean_{i}` (float, mean across all segments)
  - `timbre_std_{i}` (float, std across all segments)
  - If all values are NaN/missing, use 0 or NaN consistently.

#### Aggregated Pitch Features (24 columns)
From `segments_pitches` (shape: num_segments × 12):
- For each coefficient i (1-12):
  - `pitch_mean_{i}` (float, mean across all segments)
  - `pitch_std_{i}` (float, std across all segments)
  - If all values are NaN/missing, use 0 or NaN consistently.

#### Aggregated Loudness Features (6 columns)
From `segments_loudness_max`, `segments_loudness_max_time`, `segments_loudness_max_start`:
- `segments_loudness_max_mean` (float)
- `segments_loudness_max_std` (float)
- `segments_loudness_max_time_mean` (float)
- `segments_loudness_max_time_std` (float)
- `segments_loudness_max_start_mean` (float)
- `segments_loudness_max_start_std` (float)

**Total: ~55 columns per song**

---

## CSV 2: artists_1500_features.csv

**Granularity:** One row per artist

Built by grouping `songs_10k_features.csv` by `artist_id`.

### Columns to include:

#### Artist Identifiers (3 columns)
- `artist_id` (str)
- `artist_name` (str)
- `song_count` (int, number of songs by this artist in the 10k subset)

#### Popularity Fields (2 columns) – **For evaluation/analysis only, NOT clustering input**
- `artist_familiarity_mean` (float, mean across artist's songs)
- `artist_hotttnesss_mean` (float, mean across artist's songs)

#### Aggregated Audio Features (varies, ~45-50 columns)
For **each** audio feature from song-level CSV (tempo, loudness, danceability, energy, timbre_mean_*, pitch_mean_*, etc.):
- `artist_{feature}_mean` (float, mean across songs)
- `artist_{feature}_std` (float, std across songs)

**Rationale:**  
- `_mean` captures the artist's average sound.
- `_std` captures variability across the artist's songs (consistency).

**Example columns:**
- `artist_tempo_mean`, `artist_tempo_std`
- `artist_loudness_mean`, `artist_loudness_std`
- `artist_timbre_mean_1_mean`, `artist_timbre_mean_1_std`
- `artist_timbre_std_1_mean`, `artist_timbre_std_1_std`
- ... (similarly for all 24 timbre, 24 pitch, 6 loudness aggregates)

**Total: ~100-110 columns per artist** (can be pruned later in analysis)

---

## CSV 3: segments_long.csv

**Granularity:** One row per segment per song

### Columns to include:

#### Identifiers (4 columns)
- `track_id` (str)
- `segment_index` (int, 0-based index of segment within the track)
- `song_artist_id` (str, for optional grouping)
- `song_title` (str, for optional reference)

#### Segment Timing (3 columns)
- `segment_start` (float, seconds, when this segment begins)
- `segment_duration` (float, seconds, optional: `segments_start[i+1] - segments_start[i]`)
- `segment_confidence` (float, if available from segments_confidence array)

#### Timbre Coefficients (12 columns)
From `segments_timbre[i]` (each row is a 12-D vector):
- `timbre_1` through `timbre_12` (float, values from the timbre vector)

#### Pitch Coefficients (12 columns)
From `segments_pitches[i]` (each row is a 12-D vector):
- `pitch_1` through `pitch_12` (float, chroma vector)

#### Loudness Envelope (3 columns)
- `loudness_max` (float, peak dB)
- `loudness_max_time` (float, seconds into segment, when peak occurs)
- `loudness_max_start` (float, dB at segment onset)

**Total: ~34 columns per segment**

**Handling variable-length sequences:**
- Each song can have a different number of segments.
- Use long-format (one row per segment, not one row per song).
- If a song has 100 segments, it contributes 100 rows to this CSV.

---

## CSV 4: beats_long.csv

**Granularity:** One row per beat per song

### Columns to include:

#### Identifiers (4 columns)
- `track_id` (str)
- `beat_index` (int, 0-based index of beat within track)
- `song_artist_id` (str, for optional grouping)
- `song_title` (str, for optional reference)

#### Beat Timing (2 columns)
- `beat_start` (float, seconds, when this beat occurs)
- `beat_duration` (float, seconds, optional: `beats_start[i+1] - beats_start[i]`)

#### Beat Confidence (1 column)
- `beat_confidence` (float, confidence measure, 0-1)

**Total: ~7 columns per beat**

---

## Implementation Requirements

### 1. **Directory Traversal**
- Walk through the 10k MSD subset directory recursively.
- Identify all `.h5` files.
- For each file, extract and process.

### 2. **HDF5 Reading**
- Use `h5py` library to open and read .h5 files.
- Navigate group structure (typically `/metadata/songs`, `/analysis/songs`, etc.).
- Handle missing keys gracefully (some fields may not exist in all files).

### 3. **Error Handling**
- Skip corrupted .h5 files with a warning; do not crash.
- For missing fields, use sensible defaults:
  - `NaN` for float fields.
  - `0` for int counts.
  - Empty string for str fields.
- Log which files were skipped and why.

### 4. **CSV Writing Strategy**
- **songs_10k_features.csv:** Stream rows to CSV as you process files (use `csv.DictWriter` or pandas append mode) to avoid memory overload.
- **segments_long.csv & beats_long.csv:** Similarly stream as you process files.
- **artists_1500_features.csv:** After all songs are processed, load songs CSV, group by `artist_id`, aggregate, and write.

### 5. **Data Type Consistency**
- Ensure columns have consistent types across all rows.
- Float columns should handle NaN properly.
- Int columns should not have NaN; use 0 or -1 if missing.

### 6. **Logging & Progress**
- Print progress every N files processed (e.g., every 100 songs).
- Log summary statistics:
  - Total songs processed.
  - Total unique artists.
  - Total segments.
  - Total beats.
  - Any warnings or skipped files.

### 7. **Output Location**
- Save all CSVs to a configurable `output_dir` (default: `./output/`).
- Create the directory if it does not exist.
- File names must match exactly: `songs_10k_features.csv`, `artists_1500_features.csv`, `segments_long.csv`, `beats_long.csv`.

---

## Code Structure

Organize the Python file as follows:

1. **Imports section** – numpy, pandas, h5py, os, csv, logging, pathlib, etc.
2. **Configuration section** – paths, defaults, constants.
3. **Helper functions:**
   - `extract_song_features(h5_file_path)` – Read one .h5 file, return dict of song-level features.
   - `extract_segments(h5_file_path, track_id)` – Read segments arrays, return list of dicts (one per segment).
   - `extract_beats(h5_file_path, track_id)` – Read beats arrays, return list of dicts (one per beat).
4. **Main processing loop** – Walk directory, call helper functions, write to CSVs.
5. **Aggregation function** – Build `artists_1500_features.csv` from songs CSV.
6. **Main block** – Entry point with argument parsing (e.g., input MSD path, output directory).

---

## Execution

The script should be runnable as:

```bash
python msd_extraction.py --input_dir /path/to/msd_10k_subset --output_dir ./output
```

Or with default paths if not specified.

---

## Key Considerations

1. **Performance:** Processing 10k songs + segments/beats may take time. Consider:
   - Multiprocessing for independent .h5 files (optional).
   - Chunked CSV writing (do not load all into memory).

2. **Validation:** After extraction, basic sanity checks:
   - Number of rows in songs CSV should be ~10k.
   - Number of rows in artists CSV should be ~1500.
   - segments CSV rows >> 10k (many segments per song).
   - beats CSV rows >> 10k (many beats per song).

3. **Robustness:** Handle edge cases:
   - Songs with 0 segments (use NaN/0 for aggregate features).
   - Files with missing `artist_familiarity` / `artist_hotttnesss`.
   - Corrupted or incomplete .h5 files.

4. **Documentation:** Include inline comments and docstrings for all functions.

---

## Output Validation

After running, verify:

```python
import pandas as pd

songs = pd.read_csv('output/songs_10k_features.csv')
artists = pd.read_csv('output/artists_1500_features.csv')
segments = pd.read_csv('output/segments_long.csv')
beats = pd.read_csv('output/beats_long.csv')

print(f"Songs shape: {songs.shape}")  # Expected: (~10000, ~55)
print(f"Artists shape: {artists.shape}")  # Expected: (~1500, ~110)
print(f"Segments shape: {segments.shape}")  # Expected: (large number, ~34)
print(f"Beats shape: {beats.shape}")  # Expected: (large number, ~7)

print(f"\nSongs columns: {songs.columns.tolist()}")
print(f"\nArtists columns: {artists.columns.tolist()}")
print(f"\nSegments columns: {segments.columns.tolist()}")
print(f"\nBeats columns: {beats.columns.tolist()}")
```

---

## Notes

- This extraction is **data-loading only**; no feature engineering, scaling, or analysis is performed yet.
- The `artist_familiarity` and `artist_hotttnesss` columns are preserved in the CSVs for later evaluation, but should **NOT** be included when training K-Means/GMM clustering.
- The segments and beats CSVs are optional for immediate clustering but are valuable for future feature engineering or sequence-based modeling.
