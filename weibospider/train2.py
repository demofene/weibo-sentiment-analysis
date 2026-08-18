import copy
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

def _build_missing_dependency_message(package_name):
    repo_root = Path(__file__).resolve().parent.parent
    requirements_file = repo_root / "requirements.txt"
    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    command_examples = [
        f"{sys.executable} -m pip install -r \"{requirements_file}\"",
        f"{sys.executable} -m pip install {package_name}",
    ]

    if venv_python.exists():
        command_examples.append(f"\"{venv_python}\" train_bert.py")

    return (
        f"Missing required dependency '{package_name}'.\n"
        f"Current interpreter: {sys.executable}\n"
        "This usually means the script is running with a different Python interpreter than the repo's virtual environment.\n"
        "Try one of these commands:\n"
        + "\n".join(f"  {command}" for command in command_examples)
    )


try:
    import pandas as pd
except ModuleNotFoundError as exc:
    if exc.name == "pandas":
        raise SystemExit(_build_missing_dependency_message("pandas")) from exc
    raise

try:
    import torch
    from torch.nn import CrossEntropyLoss
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        raise SystemExit(_build_missing_dependency_message("torch")) from exc
    raise

try:
    from datasets import load_dataset
except ModuleNotFoundError as exc:
    if exc.name == "datasets":
        raise SystemExit(_build_missing_dependency_message("datasets")) from exc
    raise

try:
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError as exc:
    if exc.name == "sklearn":
        raise SystemExit(_build_missing_dependency_message("scikit-learn")) from exc
    raise

try:
    from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
except ModuleNotFoundError as exc:
    if exc.name == "transformers":
        raise SystemExit(_build_missing_dependency_message("transformers")) from exc
    raise
try:
    from evaluate_bert import (
        build_classification_report,
        compute_trainer_metrics,
        save_evaluation_report,
    )
except ImportError:  # pragma: no cover
    from weibospider.evaluate_bert import (
        build_classification_report,
        compute_trainer_metrics,
        save_evaluation_report,
    )
try:
    from project_paths import (
        BASE_DIR,
        FINAL_CHECKPOINT_DIR,
        FINAL_MODEL_DIR,
        LABEL_FILE,
        REPO_ROOT,
        STAGE1_CHECKPOINT_DIR,
        STAGE1_MODEL_DIR,
        rewrite_json_path_fields,
        to_relative_path,
    )
except ImportError:  # pragma: no cover
    from weibospider.project_paths import (
        BASE_DIR,
        FINAL_CHECKPOINT_DIR,
        FINAL_MODEL_DIR,
        LABEL_FILE,
        REPO_ROOT,
        STAGE1_CHECKPOINT_DIR,
        STAGE1_MODEL_DIR,
        rewrite_json_path_fields,
        to_relative_path,
    )


class WeiboDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(device=logits.device, dtype=logits.dtype)

        loss = CrossEntropyLoss(weight=weight)(logits, labels)
        return (loss, outputs) if return_outputs else loss


class TwoStageSentimentTrainer:
    def __init__(self, base_dir=BASE_DIR, model_name="hfl/chinese-bert-wwm"):
        self.base_dir = base_dir
        self.repo_root = REPO_ROOT
        self.model_name = model_name          
        self.num_labels = 3
        self.label_names = ["positive", "neutral", "negative"]
        self.label_map = {label: idx for idx, label in enumerate(self.label_names)}
        self.id_to_label = {idx: label for label, idx in self.label_map.items()}
        self.binary_label_map = {"negative": 0, "positive": 1}

        # 默认路径保留，但方法参数允许覆盖
        self.stage1_dir = STAGE1_MODEL_DIR
        self.stage2_dir = FINAL_MODEL_DIR
        self.stage1_checkpoint_dir = STAGE1_CHECKPOINT_DIR
        self.stage2_checkpoint_dir = FINAL_CHECKPOINT_DIR
        self.label_file = LABEL_FILE
        self.logs_stage1_dir = os.path.join(self.base_dir, "reports", "logs", "stage1")
        self.logs_stage2_dir = os.path.join(self.base_dir, "reports", "logs", "stage2")

        self.tokenizer = self._load_tokenizer()         

    def _load_tokenizer(self):
        for candidate in (self.stage1_dir, self.stage2_dir, self.model_name):
            if candidate == self.model_name or os.path.exists(candidate):
                try:
                    return AutoTokenizer.from_pretrained(candidate)
                except OSError:
                    continue
        raise RuntimeError("Unable to load tokenizer from local model directories or base model.")

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

    def _label_counter(self, labels):
        return {self.id_to_label[idx]: count for idx, count in Counter(labels).items()}

    def _compute_class_weights(self, labels):
        counts = Counter(labels)
        total = len(labels)
        weights = [total / (self.num_labels * counts[idx]) for idx in range(self.num_labels)]
        return torch.tensor(weights, dtype=torch.float32)

    def _load_comment_dataframe(self, csv_file=None):
        csv_file = self._resolve_path(csv_file, self.label_file)
        df = pd.read_csv(csv_file, encoding="utf-8-sig")

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

    def load_weibo_binary_data(self):
        print("正在加载 weibo_senti_100k 数据集...")
        dataset = load_dataset("dirtycomputer/weibo_senti_100k")
        train_data = dataset["train"]
        texts = train_data["review"]
        binary_labels = train_data["label"]

        print(f"已加载 {len(texts)} 条二分类训练样本。")
        print(
            "二分类标签分布 - "
            f"negative: {sum(label == 0 for label in binary_labels)}, "
            f"positive: {sum(label == 1 for label in binary_labels)}"
        )
        return texts, binary_labels

    def load_your_validation_data(self, csv_file=None):
        df = self._get_labeled_dataframe(csv_file)
        texts = df["text"].tolist()
        labels = [self.label_map[label] for label in df["label"].tolist()]

        print(f"已加载 {len(texts)} 条人工标注样本。")
        print(f"标签分布: {df['label'].value_counts().to_dict()}")
        return texts, labels

    def tokenize_data(self, texts, labels):
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        return WeiboDataset(encodings, labels)

    def compute_metrics(self, pred):
        return compute_trainer_metrics(pred)

    def _save_evaluation_report(
        self,
        output_dir,
        eval_results,
        val_labels,
        pred_labels,
        classification_report_dict,
    ):
        return save_evaluation_report(
            output_dir,
            eval_results,
            val_labels,
            pred_labels,
            classification_report_dict,
            self.id_to_label,
        )

    def _sanitize_checkpoint_metadata(self, output_dir):
        if not os.path.exists(output_dir):
            return

        rewrite_json_path_fields(
            os.path.join(output_dir, "config.json"),
            ["_name_or_path"],
            start=self.repo_root,
        )

        for item in os.listdir(output_dir):
            if not item.startswith("checkpoint-"):
                continue

            checkpoint_dir = os.path.join(output_dir, item)
            rewrite_json_path_fields(
                os.path.join(checkpoint_dir, "config.json"),
                ["_name_or_path"],
                start=self.repo_root,
            )
            rewrite_json_path_fields(
                os.path.join(checkpoint_dir, "trainer_state.json"),
                ["best_model_checkpoint"],
                start=self.repo_root,
            )

    def _create_stage2_trainer(
        self,
        binary_model_path,
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        output_dir,               # 实则作为 checkpoint 目录使用
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        logging_steps=10,
        save_total_limit=2,
        save_strategy="epoch",
        load_best_model_at_end=True,
        seed=42,
    ):
        os.makedirs(output_dir, exist_ok=True)

        binary_model = AutoModelForSequenceClassification.from_pretrained(binary_model_path)
        three_class_model = self.build_three_class_model_from_binary(binary_model)

        train_dataset = self.tokenize_data(train_texts, train_labels)
        val_dataset = self.tokenize_data(val_texts, val_labels)
        class_weights = self._compute_class_weights(train_labels)
        warmup_steps = max(1, min(50, len(train_texts) // max(1, per_device_train_batch_size)))

        training_kwargs = {
            "output_dir": output_dir,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "warmup_steps": warmup_steps,
            "weight_decay": 0.01,
            "logging_dir": os.path.join(output_dir, "logs"),
            "logging_steps": max(1, logging_steps),
            "eval_strategy": "epoch",
            "save_strategy": save_strategy,
            "learning_rate": learning_rate,
            "fp16": torch.cuda.is_available(),
            "seed": seed,
            "report_to": "none",
        }
        if save_strategy != "no":
            training_kwargs["save_total_limit"] = save_total_limit
        if load_best_model_at_end:
            training_kwargs["load_best_model_at_end"] = True
            training_kwargs["metric_for_best_model"] = "f1_macro"
            training_kwargs["greater_is_better"] = True

        training_args = TrainingArguments(
            **training_kwargs,
        )

        trainer = WeightedTrainer(
            model=three_class_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            class_weights=class_weights,
        )
        return trainer, val_dataset

    def find_latest_checkpoint(self, output_dir):
        if not os.path.exists(output_dir):
            return None

        checkpoints = []
        for item in os.listdir(output_dir):
            if item.startswith("checkpoint-"):
                checkpoint_path = os.path.join(output_dir, item)
                state_file = os.path.join(checkpoint_path, "trainer_state.json")
                if os.path.exists(state_file):
                    checkpoints.append((checkpoint_path, os.path.getmtime(state_file)))

        if not checkpoints:
            return None

        checkpoints.sort(key=lambda item: item[1], reverse=True)
        return checkpoints[0][0]

    def stage1_train_binary(
        self,
        train_texts=None,
        train_labels=None,
        output_dir=None,
        checkpoint_output_dir=None,   # 新增：允许自定义 checkpoint 目录
    ):
        output_dir = self._resolve_path(output_dir, self.stage1_dir)
        checkpoint_output_dir = self._resolve_path(
            checkpoint_output_dir, self.stage1_checkpoint_dir
        )

        if train_texts is None or train_labels is None:
            train_texts, train_labels = self.load_weibo_binary_data()

        print("\n" + "=" * 60)
        print("第一阶段：二分类训练")
        print("=" * 60)

        checkpoint_dir = self.find_latest_checkpoint(checkpoint_output_dir)
        if checkpoint_dir:
            print(f"发现 checkpoint，从这里继续训练: {checkpoint_dir}")
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        else:
            print("从基础模型开始进行二分类训练。")
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)

        model.config.id2label = {0: "negative", 1: "positive"}
        model.config.label2id = self.binary_label_map.copy()

        stratify_labels = train_labels if len(set(train_labels)) > 1 else None
        train_texts_split, val_texts, train_labels_split, val_labels = train_test_split(
            train_texts,
            train_labels,
            test_size=0.1,
            random_state=42,
            stratify=stratify_labels,
        )

        train_dataset = self.tokenize_data(train_texts_split, train_labels_split)
        val_dataset = self.tokenize_data(val_texts, val_labels)

        training_args = TrainingArguments(
            output_dir=checkpoint_output_dir,            # 使用可配置的 checkpoint 目录
            num_train_epochs=2,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=self.logs_stage1_dir,
            logging_steps=500,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
            seed=42,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )

        try:
            if checkpoint_dir:
                trainer.train(resume_from_checkpoint=checkpoint_dir)
            else:
                trainer.train()
        except KeyboardInterrupt:
            print("\n训练被中断，正在保存当前模型。")
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            self._sanitize_checkpoint_metadata(checkpoint_output_dir)
            self._sanitize_checkpoint_metadata(output_dir)
            raise

        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self._sanitize_checkpoint_metadata(checkpoint_output_dir)
        self._sanitize_checkpoint_metadata(output_dir)
        print(f"第一阶段完成，模型已保存到: {to_relative_path(output_dir, start=self.repo_root)}")
        return trainer.model

    def build_three_class_model_from_binary(self, binary_model):
        config = copy.deepcopy(binary_model.config)
        config.num_labels = self.num_labels
        config.id2label = {idx: label for idx, label in self.id_to_label.items()}
        config.label2id = self.label_map.copy()

        three_class_model = AutoModelForSequenceClassification.from_config(config)

        # 通用获取基础编码器（BERT 是 .bert，RoBERTa 是 .roberta）
        base_model = getattr(binary_model, 'bert', None) or \
                     getattr(binary_model, 'roberta', None) or \
                     getattr(binary_model, 'model', None)  # 兜底
        target_base = getattr(three_class_model, 'bert', None) or \
                      getattr(three_class_model, 'roberta', None) or \
                      getattr(three_class_model, 'model', None)

        if base_model is None or target_base is None:
            raise RuntimeError("无法识别模型的基础编码器，暂不支持该模型。")

        target_base.load_state_dict(base_model.state_dict())

        with torch.no_grad():
            binary_classifier_weight = binary_model.classifier.weight.clone()
            binary_classifier_bias = binary_model.classifier.bias.clone()

            hidden_size = binary_classifier_weight.shape[1]
            three_class_weight = torch.zeros(
                self.num_labels, hidden_size, dtype=binary_classifier_weight.dtype
            )
            three_class_bias = torch.zeros(self.num_labels, dtype=binary_classifier_bias.dtype)

            negative_idx = self.binary_label_map["negative"]
            positive_idx = self.binary_label_map["positive"]

            three_class_weight[self.label_map["positive"]] = binary_classifier_weight[positive_idx]
            three_class_weight[self.label_map["negative"]] = binary_classifier_weight[negative_idx]
            three_class_weight[self.label_map["neutral"]] = (
                binary_classifier_weight[negative_idx] + binary_classifier_weight[positive_idx]
            ) / 2

            three_class_bias[self.label_map["positive"]] = binary_classifier_bias[positive_idx]
            three_class_bias[self.label_map["negative"]] = binary_classifier_bias[negative_idx]
            three_class_bias[self.label_map["neutral"]] = (
                binary_classifier_bias[negative_idx] + binary_classifier_bias[positive_idx]
            ) / 2

            three_class_model.classifier.weight = torch.nn.Parameter(three_class_weight)
            three_class_model.classifier.bias = torch.nn.Parameter(three_class_bias)

        return three_class_model

    def stage2_finetune_three_class(
        self,
        binary_model_path=None,
        your_data_file=None,
        output_dir=None,
        checkpoint_output_dir=None,
        num_train_epochs=5,
        learning_rate=2e-5,
        base_model_name=None,
    ):
        binary_model_path = self._resolve_path(binary_model_path, self.stage1_dir)
        your_data_file = self._resolve_path(your_data_file, self.label_file)
        output_dir = self._resolve_path(output_dir, self.stage2_dir)
        checkpoint_output_dir = self._resolve_path(checkpoint_output_dir, self.stage2_checkpoint_dir)

        print("\n" + "=" * 60)
        print("第二阶段：三分类微调")
        print("=" * 60)

        # 1. 准备模型
        if base_model_name or not os.path.exists(binary_model_path):
            if base_model_name is None:
                base_model_name = self.model_name
            print(f"跳过第一阶段，直接使用基座模型 {base_model_name} 创建三分类模型")
            model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name, num_labels=self.num_labels
            )
            # 标记为“未使用二分类模型”，后面会直接使用此 model
            use_binary = False
        else:
            print(f"从第一阶段模型创建三分类模型: {binary_model_path}")
            binary_model = AutoModelForSequenceClassification.from_pretrained(binary_model_path)
            model = self.build_three_class_model_from_binary(binary_model)
            use_binary = True

        # 2. 加载标注数据并划分
        your_texts, your_labels = self.load_your_validation_data(your_data_file)
        if len(your_texts) < 10:
            print("人工标注样本少于 10 条，暂时不建议开始微调。")
            return None

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            your_texts, your_labels, test_size=0.2, random_state=42, stratify=your_labels
        )

        print("\n数据划分：")
        print(f"训练集: {len(train_texts)} 条，标签分布: {self._label_counter(train_labels)}")
        print(f"验证集: {len(val_texts)} 条，标签分布: {self._label_counter(val_labels)}")

        # 3. 创建数据集和 Trainer
        train_dataset = self.tokenize_data(train_texts, train_labels)
        val_dataset = self.tokenize_data(val_texts, val_labels)
        class_weights = self._compute_class_weights(train_labels)
        warmup_steps = max(1, min(50, len(train_texts) // max(1, 16)))

        training_args = TrainingArguments(
            output_dir=checkpoint_output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir=os.path.join(checkpoint_output_dir, "logs"),
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),
            seed=42,
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=2,
        )

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            class_weights=class_weights,
        )

        # 4. 开始训练
        print("\n开始第二阶段微调...")
        trainer.train()

        # 5. 评估
        print("\n训练完成，正在评估验证集...")
        pred_output = trainer.predict(val_dataset)
        pred_labels = pred_output.predictions.argmax(-1)

        print(f"预测标签分布: {self._label_counter(pred_labels)}")
        print(f"真实标签分布: {self._label_counter(val_labels)}")

        report_text, report_dict = build_classification_report(
            val_labels, pred_labels, self.label_names
        )
        eval_results = trainer.evaluate()

        # 6. 保存最终模型到 output_dir（而非 checkpoint 目录）
        os.makedirs(output_dir, exist_ok=True)
        print(f"保存最终模型到: {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # 清理检查点中的绝对路径
        self._sanitize_checkpoint_metadata(checkpoint_output_dir)
        self._sanitize_checkpoint_metadata(output_dir)

        # 保存评估报告
        self._save_evaluation_report(
            output_dir, eval_results, val_labels, pred_labels, report_dict
        )

        print("\n" + report_text)
        print("=" * 60)
        print(f"验证集准确率: {eval_results['eval_accuracy']:.4f}")
        print(f"宏平均 F1: {eval_results['eval_f1_macro']:.4f}")
        print(f"模型已保存到: {output_dir}")

        return trainer.model
    
if __name__ == "__main__":
    import os

    # 可用的本地模型映射
    local_models = {
        "1": os.path.join(REPO_ROOT, "weibospider", "models", "final"),
        "2": os.path.join(REPO_ROOT, "weibospider", "models", "chinese-roberta-wwm-ext"),
        "3": os.path.join(REPO_ROOT, "weibospider", "models", "chinese-bert-base"),
    }

    print("请选择要微调的基础模型（本地路径）：")
    for key, path in local_models.items():
        print(f"{key}. {os.path.basename(path)} ({path})")
    choice = input("输入编号 (1/2/3): ").strip()
    base_model_path = local_models.get(choice)

    if not base_model_path or not os.path.exists(base_model_path):
        print(f"模型路径不存在: {base_model_path}")
        exit(1)

    # 根据模型标识生成专用输出目录，避免覆盖 final
    model_folder_name = os.path.basename(base_model_path.rstrip(os.sep))
    if model_folder_name == "final":
        model_folder_name = "bert-wwm"   # 避免用 "final" 做目录名，容易混淆
    output_dir = os.path.join(REPO_ROOT, "weibospider", "models", f"sentiment_{model_folder_name}")

    # 检查是否已有该目录，防止意外覆盖
    if os.path.exists(output_dir):
        overwrite = input(f"输出目录 {output_dir} 已存在，是否覆盖？(y/n): ").strip().lower()
        if overwrite != 'y':
            print("训练取消。")
            exit(0)

    # 创建 trainer 实例
    trainer = TwoStageSentimentTrainer(model_name=base_model_path)

    print("\n训练模式：")
    print("1. 直接三分类微调（跳过第一阶段，推荐）")
    print("2. 完整两阶段训练（先二分类预训练，再三分类微调）")
    mode = input("请输入 1 或 2: ").strip()

    if mode == "1":
        # 直接三分类微调，使用基座模型
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        trainer.stage2_finetune_three_class(
            base_model_name=base_model_path,
            output_dir=output_dir,
            checkpoint_output_dir=checkpoint_dir,
        )
    elif mode == "2":
        # 完整两阶段
        stage1_output = os.path.join(output_dir, "stage1")
        stage1_checkpoint = os.path.join(stage1_output, "checkpoints")
        print(f"\n第一阶段输出: {stage1_output}")
        trainer.stage1_train_binary(
            output_dir=stage1_output,
            checkpoint_output_dir=stage1_checkpoint,
        )
        # 第二阶段基于第一阶段产物
        stage2_checkpoint = os.path.join(output_dir, "checkpoints")
        trainer.stage2_finetune_three_class(
            binary_model_path=stage1_output,  # 使用第一阶段模型
            output_dir=output_dir,
            checkpoint_output_dir=stage2_checkpoint,
        )
    else:
        print("无效输入")