import argparse
import json
import os
from collections import Counter

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from transformers import BertForSequenceClassification, BertTokenizer

try:
    from project_paths import BASE_DIR, FINAL_MODEL_DIR, LABEL_FILE, REPO_ROOT, to_relative_path
except ImportError:  # pragma: no cover
    from weibospider.project_paths import BASE_DIR, FINAL_MODEL_DIR, LABEL_FILE, REPO_ROOT, to_relative_path


DEFAULT_LABEL_NAMES = ["positive", "neutral", "negative"]


def build_label_maps(label_names=None):
    resolved_label_names = list(label_names or DEFAULT_LABEL_NAMES)
    label_map = {label: idx for idx, label in enumerate(resolved_label_names)}
    id_to_label = {idx: label for label, idx in label_map.items()}
    return resolved_label_names, label_map, id_to_label


def label_counter(labels, id_to_label):
    return {id_to_label[idx]: count for idx, count in Counter(labels).items()}


def compute_metrics_from_predictions(labels, pred_labels):
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels,
        pred_labels,
        average="weighted",
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        pred_labels,
        average="macro",
        zero_division=0,
    )
    accuracy = accuracy_score(labels, pred_labels)

    return {
        "accuracy": accuracy,
        "f1_weighted": weighted_f1,
        "precision_weighted": weighted_precision,
        "recall_weighted": weighted_recall,
        "f1_macro": macro_f1,
        "precision_macro": macro_precision,
        "recall_macro": macro_recall,
    }


def compute_trainer_metrics(pred):
    labels = pred.label_ids
    pred_labels = pred.predictions.argmax(-1)
    return compute_metrics_from_predictions(labels, pred_labels)


def build_classification_report(labels, pred_labels, label_names=None):
    resolved_label_names, _, _ = build_label_maps(label_names)
    report_text = classification_report(
        labels,
        pred_labels,
        target_names=resolved_label_names,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        labels,
        pred_labels,
        target_names=resolved_label_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    return report_text, report_dict


def save_evaluation_report(
    output_dir,
    eval_results,
    val_labels,
    pred_labels,
    classification_report_dict,
    id_to_label,
):
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "evaluation_report.json")
    payload = {
        "metrics": {key: float(value) for key, value in eval_results.items()},
        "true_label_distribution": label_counter(val_labels, id_to_label),
        "pred_label_distribution": label_counter(pred_labels, id_to_label),
        "classification_report": classification_report_dict,
    }
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    print(f"Evaluation report saved to: {to_relative_path(report_path, start=REPO_ROOT)}")
    return report_path


class BertSentimentEvaluator:
    def __init__(
        self,
        base_dir=BASE_DIR,
        label_names=None,
        default_model_dir=FINAL_MODEL_DIR,
        default_label_file=LABEL_FILE,
    ):
        self.base_dir = base_dir
        self.repo_root = os.path.dirname(self.base_dir)
        self.label_names, self.label_map, self.id_to_label = build_label_maps(label_names)
        self.default_model_dir = default_model_dir
        self.default_label_file = default_label_file

    def _resolve_path(self, path, default_path):
        if path is None:
            return default_path
        if os.path.isabs(path):
            return path

        base_candidate = os.path.join(self.base_dir, path)
        if os.path.exists(base_candidate):
            return base_candidate

        repo_candidate = os.path.join(self.repo_root, path)
        if os.path.exists(repo_candidate):
            return repo_candidate

        return base_candidate

    def _load_comment_dataframe(self, csv_file=None):
        csv_path = self._resolve_path(csv_file, self.default_label_file)
        df = pd.read_csv(csv_path, encoding="utf-8-sig")

        if len(df.columns) < 2:
            raise ValueError("CSV must contain at least two columns: id and text.")

        rename_map = {df.columns[0]: "row_id", df.columns[1]: "text"}
        if len(df.columns) > 2:
            rename_map[df.columns[2]] = "label"
        if len(df.columns) > 3:
            rename_map[df.columns[3]] = "note"

        df = df.rename(columns=rename_map)
        df["text"] = df["text"].fillna("").astype(str).str.strip()
        if "label" in df.columns:
            df["label"] = df["label"].fillna("").astype(str).str.strip()
        if "note" in df.columns:
            df["note"] = df["note"].fillna("").astype(str).str.strip()
        return df

    def _get_labeled_dataframe(self, csv_file=None):
        df = self._load_comment_dataframe(csv_file)
        if "label" not in df.columns:
            raise ValueError("CSV does not contain a label column.")
        df = df[df["label"].isin(self.label_map.keys())].copy()
        df = df[df["text"] != ""].copy()
        return df

    def _load_model_and_tokenizer(self, model_dir=None):
        resolved_model_dir = self._resolve_path(model_dir, self.default_model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BertForSequenceClassification.from_pretrained(resolved_model_dir).to(device)
        tokenizer = BertTokenizer.from_pretrained(resolved_model_dir)
        model.eval()
        return model, tokenizer, device

    def predict_texts_with_scores(self, texts, model_dir=None, batch_size=32):
        model, tokenizer, device = self._load_model_and_tokenizer(model_dir)

        pred_labels = []
        confidences = []
        margins = []
        entropies = []
        probabilities = []

        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=128,
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu()
                top_values, top_indices = torch.topk(probs, k=min(2, len(self.label_names)), dim=-1)

                pred_labels.extend(top_indices[:, 0].tolist())
                confidences.extend(top_values[:, 0].tolist())
                if top_values.shape[1] > 1:
                    margins.extend((top_values[:, 0] - top_values[:, 1]).tolist())
                else:
                    margins.extend([0.0] * top_values.shape[0])

                entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
                entropies.extend(entropy.tolist())
                probabilities.extend(probs.tolist())

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "pred_labels": pred_labels,
            "confidences": confidences,
            "margins": margins,
            "entropies": entropies,
            "probabilities": probabilities,
        }

    def evaluate_saved_model(
        self,
        model_dir=None,
        csv_file=None,
        batch_size=32,
        output_dir=None,
        save_report=False,
    ):
        df = self._get_labeled_dataframe(csv_file)
        texts = df["text"].tolist()
        labels = [self.label_map[label] for label in df["label"].tolist()]

        score_dict = self.predict_texts_with_scores(texts, model_dir=model_dir, batch_size=batch_size)
        pred_labels = score_dict["pred_labels"]

        metrics = compute_metrics_from_predictions(labels, pred_labels)
        _, report_dict = build_classification_report(labels, pred_labels, self.label_names)

        result = {
            **metrics,
            "true_label_distribution": label_counter(labels, self.id_to_label),
            "pred_label_distribution": label_counter(pred_labels, self.id_to_label),
            "classification_report": report_dict,
        }

        should_save_report = save_report or output_dir is not None
        if should_save_report:
            resolved_output_dir = self._resolve_path(
                output_dir,
                self._resolve_path(model_dir, self.default_model_dir),
            )
            save_evaluation_report(
                resolved_output_dir,
                metrics,
                labels,
                pred_labels,
                report_dict,
                self.id_to_label,
            )

        return result

    def predict(self, text, model_dir=None):
        model, tokenizer, device = self._load_model_and_tokenizer(model_dir)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self.id_to_label[prediction]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved BERT sentiment model.")
    parser.add_argument("--model-dir", default=FINAL_MODEL_DIR, help="Path to the saved model directory.")
    parser.add_argument("--csv-file", default=LABEL_FILE, help="CSV file with labeled evaluation data.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory used to save evaluation_report.json. If omitted, no file is saved.",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save evaluation_report.json to the model directory when --output-dir is not provided.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    evaluator = BertSentimentEvaluator()
    result = evaluator.evaluate_saved_model(
        model_dir=args.model_dir,
        csv_file=args.csv_file,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        save_report=args.save_report,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
