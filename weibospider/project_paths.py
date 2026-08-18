import json
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)

CONFIG_DIR = os.path.join(BASE_DIR, "config")
COOKIE_FILE = os.path.join(CONFIG_DIR, "cookie.txt")

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
CLEANED_DATA_DIR = os.path.join(DATA_DIR, "cleaned")
LABELS_DIR = os.path.join(DATA_DIR, "labels")
LABEL_FILE = os.path.join(LABELS_DIR, "to_label.csv")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
SENTIMENT_RESULTS_FILE = os.path.join(RESULTS_DIR, "sentiment_results_all.json")

MODELS_DIR = os.path.join(BASE_DIR, "models")
STAGE1_MODEL_DIR = os.path.join(MODELS_DIR, "stage1")
FINAL_MODEL_DIR = os.path.join(MODELS_DIR, "final")
MODEL_BACKUPS_DIR = os.path.join(MODELS_DIR, "backups")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
STAGE1_CHECKPOINT_DIR = os.path.join(CHECKPOINTS_DIR, "stage1")
FINAL_CHECKPOINT_DIR = os.path.join(CHECKPOINTS_DIR, "final")

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
ANALYSIS_DIR = os.path.join(REPORTS_DIR, "analysis")
CROSS_VALIDATION_DIR = os.path.join(REPORTS_DIR, "cross_validation")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def normalize_path(path):
    return path.replace("\\", "/")


def to_relative_path(path, start=REPO_ROOT):
    if path is None:
        return None

    if not isinstance(path, str):
        return path

    normalized_start = os.path.abspath(start)

    if os.path.isabs(path):
        normalized_path = os.path.abspath(path)
        try:
            relative_path = os.path.relpath(normalized_path, normalized_start)
        except ValueError:
            return normalize_path(normalized_path)

        if relative_path.startswith(".."):
            return normalize_path(normalized_path)
        return normalize_path(relative_path)

    return normalize_path(path)


def rewrite_json_path_fields(json_path, field_names, start=REPO_ROOT):
    if not os.path.exists(json_path):
        return False

    with open(json_path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)

    changed = False
    for field_name in field_names:
        field_value = payload.get(field_name)
        if isinstance(field_value, str):
            relative_value = to_relative_path(field_value, start=start)
            if relative_value != field_value:
                payload[field_name] = relative_value
                changed = True

    if changed:
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

    return changed
