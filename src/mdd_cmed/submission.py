import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd


def validate_results_frame(results_df: pd.DataFrame, expected_rows: int | None = None) -> None:
    if expected_rows is not None and len(results_df) != expected_rows:
        raise AssertionError("results row count does not match test metadata")
    if list(results_df.columns) != ["predict"]:
        raise AssertionError(f"results.csv must contain exactly one predict column, got {list(results_df.columns)}")
    if results_df["predict"].isna().any():
        raise AssertionError("submission contains NaN predictions")
    if (results_df["predict"].astype(str).str.strip() == "").any():
        raise AssertionError("submission contains empty predictions")


def write_submission(
    predictions: pd.Series | list[str],
    output_dir: str | Path,
    expected_rows: int | None = None,
    make_compatibility_zip: bool = True,
) -> tuple[Path, Path, dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame({"predict": pd.Series(predictions).fillna("").astype(str).values})
    validate_results_frame(results_df, expected_rows=expected_rows)

    results_path = output_dir / "results.csv"
    zip_path = output_dir / "predict.zip"
    compatibility_zip_path = output_dir / "prediction.zip"
    manifest_path = output_dir / "submission_manifest.json"
    results_df.to_csv(results_path, index=False)

    zip_targets = [zip_path]
    if make_compatibility_zip:
        zip_targets.append(compatibility_zip_path)

    zipped_results = b""
    for out_zip in zip_targets:
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(results_path, arcname="results.csv")
        with zipfile.ZipFile(out_zip) as zf:
            names = zf.namelist()
            if names != ["results.csv"]:
                raise AssertionError(f"zip must contain only results.csv, got {names}")
            zipped_results = zf.read("results.csv")

    manifest = {
        "results_path": str(results_path),
        "zip_path": str(zip_path),
        "compatibility_zip_path": str(compatibility_zip_path) if make_compatibility_zip else None,
        "inner_files": ["results.csv"],
        "columns": list(results_df.columns),
        "rows": int(len(results_df)),
        "line_count_in_zip_results": int(zipped_results.count(b"\n")),
        "sha256_results_in_zip": hashlib.sha256(zipped_results).hexdigest(),
        "first_prediction": str(results_df.iloc[0]["predict"]) if len(results_df) else "",
        "last_prediction": str(results_df.iloc[-1]["predict"]) if len(results_df) else "",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return results_path, zip_path, manifest
