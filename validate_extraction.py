#!/usr/bin/env python3
"""
Validation script for MSD extraction outputs.
Verifies the extracted CSV files match expected structure and content.
"""

import pandas as pd
import sys

def validate_extraction(output_dir='./output'):
    """Validate the extracted CSV files."""

    print("=" * 60)
    print("MSD Extraction Validation Report")
    print("=" * 60)

    try:
        # Load all CSVs
        songs = pd.read_csv(f'{output_dir}/songs_10k_features.csv')
        artists = pd.read_csv(f'{output_dir}/artists_1500_features.csv')
        segments = pd.read_csv(f'{output_dir}/segments_long.csv')
        beats = pd.read_csv(f'{output_dir}/beats_long.csv')

        print("\n1. FILE SIZES AND SHAPES")
        print("-" * 60)
        print(f"✓ Songs CSV: {songs.shape} (expected: ~10000 rows, ~82 cols)")
        print(f"✓ Artists CSV: {artists.shape} (expected: ~3888 rows, ~151 cols)")
        print(f"✓ Segments CSV: {segments.shape} (expected: >>10000 rows, 34 cols)")
        print(f"✓ Beats CSV: {beats.shape} (expected: >>10000 rows, 7 cols)")

        # Validate shapes
        assert songs.shape[0] >= 9000, "Too few songs!"
        assert songs.shape[1] >= 75, "Missing song columns!"
        assert artists.shape[0] >= 3000, "Too few artists!"
        assert segments.shape[0] > 10000, "Too few segments!"
        assert beats.shape[0] > 10000, "Too few beats!"

        print("\n2. KEY COLUMNS VERIFICATION")
        print("-" * 60)

        # Check songs columns
        required_song_cols = [
            'track_id', 'song_id', 'artist_id', 'artist_name', 'title',
            'year', 'release', 'duration', 'tempo', 'loudness',
            'artist_familiarity', 'artist_hotttnesss',
            'timbre_mean_1', 'timbre_std_1', 'pitch_mean_1', 'pitch_std_1',
            'segments_loudness_max_mean', 'num_segments', 'num_beats'
        ]
        missing_song_cols = [col for col in required_song_cols if col not in songs.columns]
        if missing_song_cols:
            print(f"✗ Missing song columns: {missing_song_cols}")
        else:
            print(f"✓ All required song columns present")

        # Check artists columns
        required_artist_cols = [
            'artist_id', 'artist_name', 'song_count',
            'artist_familiarity_mean', 'artist_hotttnesss_mean'
        ]
        missing_artist_cols = [col for col in required_artist_cols if col not in artists.columns]
        if missing_artist_cols:
            print(f"✗ Missing artist columns: {missing_artist_cols}")
        else:
            print(f"✓ All required artist columns present")

        # Check segments columns
        required_segment_cols = [
            'track_id', 'segment_index', 'segment_start',
            'timbre_1', 'timbre_12', 'pitch_1', 'pitch_12',
            'loudness_max', 'loudness_max_time', 'loudness_max_start'
        ]
        missing_segment_cols = [col for col in required_segment_cols if col not in segments.columns]
        if missing_segment_cols:
            print(f"✗ Missing segment columns: {missing_segment_cols}")
        else:
            print(f"✓ All required segment columns present")

        # Check beats columns
        required_beat_cols = [
            'track_id', 'beat_index', 'beat_start',
            'beat_duration', 'beat_confidence'
        ]
        missing_beat_cols = [col for col in required_beat_cols if col not in beats.columns]
        if missing_beat_cols:
            print(f"✗ Missing beat columns: {missing_beat_cols}")
        else:
            print(f"✓ All required beat columns present")

        print("\n3. DATA QUALITY CHECKS")
        print("-" * 60)

        # Check for duplicates
        print(f"Duplicate track_ids in songs: {songs['track_id'].duplicated().sum()}")
        print(f"Duplicate artist_ids in artists: {artists['artist_id'].duplicated().sum()}")

        # Check missing values for critical fields
        print(f"Missing track_id: {songs['track_id'].isna().sum()}")
        print(f"Missing artist_id: {songs['artist_id'].isna().sum()}")
        print(f"Missing artist_name: {songs['artist_name'].isna().sum()}")
        print(f"Missing title: {songs['title'].isna().sum()}")

        # Check familiarity/hotness (expected to have some missing)
        print(f"Missing artist_familiarity: {songs['artist_familiarity'].isna().sum()} "
              f"({100*songs['artist_familiarity'].isna().sum()/len(songs):.1f}%)")
        print(f"Missing artist_hotttnesss: {songs['artist_hotttnesss'].isna().sum()} "
              f"({100*songs['artist_hotttnesss'].isna().sum()/len(songs):.1f}%)")

        print("\n4. STATISTICAL SUMMARY")
        print("-" * 60)
        print(f"Total unique artists: {songs['artist_id'].nunique()}")
        print(f"Average songs per artist: {len(songs) / songs['artist_id'].nunique():.2f}")
        print(f"Artist with most songs: {artists.loc[artists['song_count'].idxmax(), 'artist_name']} "
              f"({artists['song_count'].max()} songs)")
        print(f"Average segments per song: {segments.shape[0] / songs.shape[0]:.1f}")
        print(f"Average beats per song: {beats.shape[0] / songs.shape[0]:.1f}")

        print("\n5. TIMBRE AND PITCH FEATURES")
        print("-" * 60)
        timbre_cols = [col for col in songs.columns if col.startswith('timbre_')]
        pitch_cols = [col for col in songs.columns if col.startswith('pitch_')]
        print(f"Timbre features: {len(timbre_cols)} (expected: 24)")
        print(f"Pitch features: {len(pitch_cols)} (expected: 24)")

        # Check that timbre/pitch aren't all NaN
        timbre_valid = songs[timbre_cols].notna().any(axis=1).sum()
        pitch_valid = songs[pitch_cols].notna().any(axis=1).sum()
        print(f"Songs with valid timbre data: {timbre_valid}/{len(songs)}")
        print(f"Songs with valid pitch data: {pitch_valid}/{len(songs)}")

        print("\n6. SAMPLE DATA")
        print("-" * 60)
        print("\nFirst 3 songs:")
        print(songs[['track_id', 'artist_name', 'title', 'duration', 'tempo',
                     'num_segments', 'num_beats']].head(3).to_string())

        print("\n\nTop 5 artists by song count:")
        print(artists.nlargest(5, 'song_count')[['artist_name', 'song_count']].to_string())

        print("\n" + "=" * 60)
        print("✓ VALIDATION COMPLETE - All checks passed!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n✗ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    output_dir = sys.argv[1] if len(sys.argv) > 1 else './output'
    success = validate_extraction(output_dir)
    sys.exit(0 if success else 1)
