# SECOM Download Instructions

## Automatic download

From the project root, run:

```powershell
.\.venv\Scripts\python.exe -m src.data --config configs/base.yaml --download --validate
```

The command downloads the official UCI archive from:

```text
https://archive.ics.uci.edu/static/public/179/secom.zip
```

The archive and extracted raw files are stored under `data/raw/` and are excluded
from Git. Validation writes only metadata, hashes, and aggregate statistics to tracked
project files.

## Manual fallback

1. Open https://archive.ics.uci.edu/dataset/179/secom.
2. Select the dataset download link.
3. Save the ZIP as `data/raw/secom.zip`.
4. Extract `secom.data`, `secom_labels.data`, and `secom.names` into `data/raw/`.
5. Run the validation command:

```powershell
.\.venv\Scripts\python.exe -m src.data --config configs/base.yaml --validate
```

Expected label mapping from the official description is `-1=pass`, `1=fail`; the
loader still verifies the values in the downloaded file before normalizing the target.

## License and attribution

UCI distributes SECOM under Creative Commons Attribution 4.0 International (CC BY
4.0). Reuse must retain appropriate attribution to Michael McCann and Adrian Johnston,
the UCI Machine Learning Repository dataset page, and DOI `10.24432/C54305`.
