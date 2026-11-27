# Source data

Download **Spotify Dataset 1921–2020, 600k+ Tracks** by Yamac Eren Ay:

https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks

Use the dataset page's Download option (Kaggle may require a free account). Put the downloaded archive here as `data/archive.zip`. Keep it zipped; it must contain `tracks.csv` at the ZIP root.

Alternatively, obtain the **original project archive.zip** from your teammate and provide its location with `--data`. The archive is not committed to Git. Do not place credentials in this repository.

From the repository root, check that you have the exact data used for the report:

```bash
python run.py check --data data/archive.zip
```

Expected SHA-256 for the uncompressed `tracks.csv`:

```text
8a08c199bd93438783e915470ec3ea1142c1267f2da48b7d1315d2b9f5d39cf9
```

If the source has been updated, request the original archive from your teammate. The analysis intentionally rejects a different source for model reproduction. `python run.py audit --data path/to/archive.zip` can inspect a different version, but the existing report's conclusions do not apply to that version.
