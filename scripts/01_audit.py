# Standalone Python version of the corresponding analysis notebook.
# Run via: python run.py audit --data /path/to/archive.zip
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = Path(os.environ.get("SPOTIFY_OUTPUT_DIR", str(PROJECT_ROOT / "runs")))
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

def display(value):
    """Render notebook tables as text; figures are saved by the analysis itself."""
    if hasattr(value, "to_string"):
        print(value.to_string())
    else:
        print(value)

def main():
    from pathlib import Path
    from zipfile import ZipFile
    from functools import lru_cache
    from datetime import date
    import ast
    import hashlib
    import platform
    import re
    import pandas as pd
    import numpy as np

    candidates = [Path('archive.zip'), Path.home() / 'Downloads' / 'archive.zip']
    DATA_PATH = Path(os.environ.get("SPOTIFY_DATA_PATH", str(PROJECT_ROOT / "data" / "archive.zip")))
    THRESHOLD = 60
    AUDIO = ['danceability', 'energy', 'tempo', 'loudness', 'valence',
             'acousticness', 'instrumentalness', 'speechiness', 'liveness']
    STRUCTURE = ['duration_ms', 'key', 'mode', 'time_signature']
    FEATURES = AUDIO + STRUCTURE
    METADATA = ['id', 'name', 'artists', 'id_artists', 'release_date', 'explicit']
    EXPECTED = FEATURES + METADATA + ['popularity']
    NUMERIC = FEATURES + ['popularity', 'explicit']
    pd.set_option('display.max_rows', 40)
    pd.set_option('display.max_columns', 12)
    pd.set_option('display.width', 150)
    print({'Python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__})

    with ZipFile(DATA_PATH) as archive:
        display(pd.DataFrame([{'file': x.filename, 'bytes_uncompressed': x.file_size}
                              for x in archive.infolist()]))
        with archive.open('tracks.csv') as f:
            digest = hashlib.sha256()
            for block in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(block)
        with archive.open('tracks.csv') as f:
            # Preserve text such as a literal song title "NA"; detect blanks explicitly.
            raw = pd.read_csv(f, keep_default_na=False)
    print('tracks.csv SHA-256:', digest.hexdigest())
    print('Shape:', raw.shape)
    print('Missing required fields:', sorted(set(EXPECTED) - set(raw.columns)))
    assert set(EXPECTED).issubset(raw.columns), 'Required fields are missing.'
    display(raw.head(3))

    df = raw.copy()
    field_rows = []
    for col in raw.columns:
        blank = raw[col].isna() | raw[col].astype(str).str.strip().eq('')
        conversion_errors = 0
        if col in NUMERIC:
            df[col] = pd.to_numeric(raw[col], errors='coerce')
            conversion_errors = int((df[col].isna() & ~blank).sum())
        field_rows.append({'field': col, 'dtype': str(df[col].dtype),
                           'blank_count': int(blank.sum()),
                           'blank_pct': 100 * blank.mean(),
                           'numeric_parse_errors': conversion_errors})
    field_audit = pd.DataFrame(field_rows)
    display(field_audit.round(4))
    print('Rows missing any primary model feature:', int(df[FEATURES].isna().any(axis=1).sum()))
    print('Rows missing target:', int(df['popularity'].isna().sum()))

    display(df[NUMERIC].describe().T[['count','mean','std','min','25%','50%','75%','max']].round(4))
    scores = ['danceability','energy','valence','acousticness','instrumentalness','speechiness','liveness']
    range_checks = {c + ' outside [0,1]': int((df[c].notna() & ~df[c].between(0,1)).sum()) for c in scores}
    range_checks.update({
        'popularity outside integer [0,100]': int((~df.popularity.between(0,100) | df.popularity.mod(1).ne(0)).sum()),
        'key outside integer [-1,11]': int((~df.key.between(-1,11) | df.key.mod(1).ne(0)).sum()),
        'mode outside {0,1}': int((~df['mode'].isin([0,1])).sum()),
        'explicit outside {0,1}': int((~df.explicit.isin([0,1])).sum()),
        'duration nonpositive or noninteger': int(((df.duration_ms <= 0) | df.duration_ms.mod(1).ne(0)).sum()),
        'negative tempo': int((df.tempo < 0).sum()),
        'time_signature outside {0,1,2,3,4,5}': int((~df.time_signature.isin([0,1,2,3,4,5])).sum()),
        'nonfinite numeric cells': int((~np.isfinite(df[NUMERIC].to_numpy())).sum()),
    })
    display(pd.DataFrame(range_checks.items(), columns=['check','count']))
    review_flags = pd.DataFrame({
        'tempo_is_zero': df.tempo.eq(0),
        'time_signature_is_zero': df.time_signature.eq(0),
        'duration_under_30_seconds': df.duration_ms.lt(30000),
        'duration_over_20_minutes': df.duration_ms.gt(1200000),
    })
    display(review_flags.sum().rename('count').to_frame())
    display(df.nlargest(5, 'duration_ms')[['id','name','duration_ms','speechiness','popularity']])

    @lru_cache(maxsize=None)
    def parse_release(value):
        if not isinstance(value, str) or not re.fullmatch(r'\d{4}(?:-\d{2}(?:-\d{2})?)?', value):
            return (None, 'invalid')
        parts = [int(p) for p in value.split('-')]
        try:
            date(parts[0], parts[1] if len(parts) > 1 else 1, parts[2] if len(parts) > 2 else 1)
        except ValueError:
            return (None, 'invalid')
        return (parts[0], {1:'year', 2:'month', 3:'day'}[len(parts)])

    parsed_dates = df.release_date.map(parse_release)
    df['release_year'] = pd.array([x[0] for x in parsed_dates], dtype='Int64')
    precision = pd.Series([x[1] for x in parsed_dates], index=df.index)
    display(precision.value_counts().rename_axis('precision').rename('rows').to_frame())
    print('Recorded year range:', df.release_year.min(), 'to', df.release_year.max())
    print('Rows released after 2020:', int(df.release_year.gt(2020).sum()))
    print('Rows recorded before 1921:', int(df.release_year.lt(1921).sum()))
    display(df.loc[df.release_year.lt(1921), ['id','name','artists','release_date']])
    print('Latest full release dates:', sorted(df.loc[precision.eq('day'),'release_date'].unique())[-5:])

    valid_target = df.popularity.between(0,100) & df.popularity.mod(1).eq(0)
    target_scores = df.loc[valid_target,'popularity']
    threshold_table = pd.DataFrame([
        {'threshold': t, 'positive_count': int(target_scores.ge(t).sum()),
         'negative_count': int(target_scores.lt(t).sum()),
         'positive_pct': 100 * target_scores.ge(t).mean()}
        for t in [50,55,60,65,70]
    ])
    display(threshold_table.round(4))
    prevalence = target_scores.ge(THRESHOLD).mean()
    print(f'Primary positive prevalence: {prevalence:.8f} ({prevalence:.4%})')
    print(f'Always-negative accuracy: {1-prevalence:.4%}; positive recall: 0%')
    df['high_popularity'] = pd.Series(pd.NA, index=df.index, dtype='Int64')
    df.loc[valid_target,'high_popularity'] = target_scores.ge(THRESHOLD).astype(int)
    assert threshold_table.loc[threshold_table.threshold.eq(THRESHOLD), 'positive_count'].iloc[0] == df.high_popularity.sum()

    duplicate_audit = pd.DataFrame([
        {'rule':'all original columns', 'extra_rows': int(raw.duplicated().sum())},
        {'rule':'track id', 'extra_rows': int(raw.duplicated('id').sum())},
        {'rule':'exact name + artist-id list', 'extra_rows': int(raw.duplicated(['name','id_artists']).sum())},
        {'rule':'exact name + artist-id list + duration', 'extra_rows': int(raw.duplicated(['name','id_artists','duration_ms']).sum())},
        {'rule':'all 13 primary features', 'extra_rows': int(df.duplicated(FEATURES).sum())},
    ])
    display(duplicate_audit)
    feature_repeats = df.loc[df.duplicated(FEATURES, keep=False)]
    feature_groups = feature_repeats.groupby(FEATURES, dropna=False).agg(
        rows=('id','size'), label_values=('high_popularity','nunique'))
    print('Repeated feature groups:', len(feature_groups))
    print('Rows in repeated feature groups:', len(feature_repeats))
    print('Repeated feature groups with conflicting binary labels:', int(feature_groups.label_values.gt(1).sum()))
    name_duplicates = df.loc[df.duplicated(['name','id_artists'], keep=False)]
    display(name_duplicates[['name','id_artists','id','duration_ms','popularity']].head(8))

    @lru_cache(maxsize=None)
    def parse_artists(value):
        try:
            parsed = ast.literal_eval(value)
            if not isinstance(parsed, list) or not all(isinstance(x,str) for x in parsed):
                return None
            return tuple(parsed)
        except (ValueError, SyntaxError, TypeError):
            return None

    artist_ids = df.id_artists.map(parse_artists)
    artist_names = df.artists.map(parse_artists)
    def list_audit(series):
        return {'unparseable_rows': int(series.map(lambda x: x is None).sum()),
                'empty_lists': int(series.map(lambda x: x == ()).sum()),
                'rows_with_blank_entries': int(series.map(lambda x: x is not None and any(not s.strip() for s in x)).sum()),
                'multi_artist_rows': int(series.map(lambda x: x is not None and len(x)>1).sum())}
    display(pd.DataFrame({'id_artists': list_audit(artist_ids), 'artists': list_audit(artist_names)}).T)
    print('Distinct nonblank artist IDs:', len({a for ids in artist_ids if ids is not None for a in ids if a.strip()}))
    length_mismatch = sum(a is not None and b is not None and len(a)!=len(b) for a,b in zip(artist_ids,artist_names))
    print('Rows with artist name / ID list length mismatch:', length_mismatch)

    df['decade'] = df.release_year // 10 * 10
    decades = df.groupby('decade', dropna=False).agg(
        rows=('id','size'), valid_targets=('high_popularity','count'),
        positive_count=('high_popularity','sum'), positive_rate=('high_popularity','mean'))
    decades['positive_pct'] = decades.pop('positive_rate') * 100
    display(decades.round(4))
    valid_features = pd.Series(np.isfinite(df[FEATURES].to_numpy()).all(axis=1), index=df.index)
    valid_id = df.id.astype(str).str.strip().ne('')
    base_eligible = valid_features & valid_target & valid_id
    print('Basic eligible rows before special-value / version policy:', int(base_eligible.sum()))
    print('Eligible positive rows:', int(df.loc[base_eligible,'high_popularity'].sum()))
    assert int(decades.rows.sum()) == len(df)
    assert int(decades.positive_count.sum()) == int(df.high_popularity.sum())
    assert int(df.id.nunique()) + int(df.id.duplicated().sum()) == len(df)
    print('Row-count and target-count reconciliation checks passed.')


if __name__ == "__main__":
    main()
