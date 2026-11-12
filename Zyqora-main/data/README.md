# data/

Images are not committed. Keep the sets separate:

| folder | purpose | used for |
|---|---|---|
| `organiser/` | the 10–20 organiser-provided labelled images + `labels.csv` | the final mandatory evaluation only, never for tuning |
| `dev/` | your own labelled development images + `labels.csv` | parity checks, threshold sweep (`sweep_threshold.py`) |
| calibration folder (anywhere) | unlabelled images for INT8 calibration, e.g. the Arm course Open Images subset | `export_int8.py` (use `--exclude` to prove no overlap with the test sets) |

`labels.csv` columns: `image_filename,ground_truth_category` (see `labels.csv.template`).
`image_filename` is the path relative to the image folder (e.g. `img01.jpg` or `sub/img01.jpg`).
Category names must match `config/mapping.json` (exact name or a listed alias).
