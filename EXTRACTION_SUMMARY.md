# MSD Data Extraction - Summary

## Extraction Complete ✓

Successfully extracted features from the Million Song Dataset (10k subset) into structured CSV files for downstream analysis.

## Output Files

All files are located in the `output/` directory:

### 1. **songs_10k_features.csv**
- **Rows:** 10,000 songs
- **Columns:** 82 features per song
- **Contents:**
  - Identifiers & Metadata (track_id, artist_id, artist_name, title, year, release, duration, etc.)
  - Popularity metrics (artist_familiarity, artist_hotttnesss)
  - Scalar audio features (tempo, key, mode, loudness, danceability, energy, etc.)
  - Aggregated timbre features (24 columns: mean & std for 12 coefficients)
  - Aggregated pitch features (24 columns: mean & std for 12 coefficients)
  - Aggregated loudness envelope features (6 columns)
  - Sequence counts (num_segments, num_beats, num_bars, num_sections, num_tatums)

### 2. **artists_1500_features.csv**
- **Rows:** 3,888 unique artists
- **Columns:** 151 features per artist
- **Contents:**
  - Artist identifiers (artist_id, artist_name)
  - Song count per artist
  - Aggregated popularity metrics
  - Mean and std for all audio features across artist's songs

**Note:** The number of artists (3,888) is higher than initially expected (~1,500) because the 10k subset has more artist diversity.

### 3. **segments_long.csv**
- **Rows:** 8,577,406 segments (avg. 858 per song)
- **Columns:** 34 features per segment
- **Contents:**
  - Segment identifiers (track_id, segment_index, song_artist_id, song_title)
  - Timing information (segment_start, segment_duration, segment_confidence)
  - Timbre coefficients (12 values)
  - Pitch/chroma coefficients (12 values)
  - Loudness envelope (loudness_max, loudness_max_time, loudness_max_start)

### 4. **beats_long.csv**
- **Rows:** 4,809,945 beats (avg. 481 per song)
- **Columns:** 7 features per beat
- **Contents:**
  - Beat identifiers (track_id, beat_index, song_artist_id, song_title)
  - Timing information (beat_start, beat_duration, beat_confidence)

## Extraction Statistics

- **Processing time:** ~5.5 minutes
- **Songs processed:** 10,000
- **Songs failed:** 0
- **Total segments extracted:** 8,577,406
- **Total beats extracted:** 4,809,945
- **Unique artists:** 3,888
- **Average songs per artist:** 2.57
- **Artist with most songs:** Mario Rosenstock (13 songs)

## Data Quality

- **No duplicate track_ids or artist_ids**
- **Missing values:**
  - artist_familiarity: 4 songs (0.04%)
  - artist_hotttnesss: 0 songs
  - title: 1 song
- **All timbre and pitch features successfully extracted for all songs**
- **All required columns present in all CSV files**

## Usage

### Running the extraction:
```bash
python msd_extraction.py --input_dir millionsongsubset/MillionSongSubset --output_dir output
```

### Validating the output:
```bash
python validate_extraction.py output
```

### Loading the data:
```python
import pandas as pd

songs = pd.read_csv('output/songs_10k_features.csv')
artists = pd.read_csv('output/artists_1500_features.csv')
segments = pd.read_csv('output/segments_long.csv')
beats = pd.read_csv('output/beats_long.csv')

print(f"Songs: {songs.shape}")
print(f"Artists: {artists.shape}")
print(f"Segments: {segments.shape}")
print(f"Beats: {beats.shape}")
```

## Next Steps

The extracted data is ready for:

1. **Artist-level clustering** using the `artists_1500_features.csv` file
   - Exclude `artist_familiarity_mean` and `artist_hotttnesss_mean` from clustering features
   - Consider feature scaling/normalization
   - Apply K-Means, GMM, or other clustering algorithms

2. **Song-level analysis** using `songs_10k_features.csv`
   - Genre classification
   - Similarity search
   - Recommendation systems

3. **Sequence-based modeling** using `segments_long.csv` and `beats_long.csv`
   - Temporal pattern analysis
   - Advanced feature engineering
   - RNN/LSTM-based models

## Files Created

- `msd_extraction.py` - Main extraction script
- `validate_extraction.py` - Validation script
- `extraction.log` - Full extraction log
- `output/` - Directory containing all CSV outputs
  - `songs_10k_features.csv`
  - `artists_1500_features.csv`
  - `segments_long.csv`
  - `beats_long.csv`

## Notes

- The extraction preserves `artist_familiarity` and `artist_hotttnesss` for evaluation purposes, but these should NOT be used as clustering features (as per the original requirements)
- All audio analysis features are based on Echo Nest analysis
- Missing values are handled gracefully (NaN for floats, 0 for integers)
- The extraction is optimized for memory efficiency by streaming CSV writes
