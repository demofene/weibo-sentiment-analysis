import json
import os
from collections import Counter
from datetime import datetime
import jieba
from collections import Counter

try:
    from clean_data import WeiboDataCleaner
    from evaluate_bert import BertSentimentEvaluator
    from project_paths import (
        BASE_DIR,
        CLEANED_DATA_DIR,
        FINAL_MODEL_DIR,
        RAW_DATA_DIR,
        REPO_ROOT,
        RESULTS_DIR,
        ensure_dir,
        to_relative_path,
    )
except ImportError:  # pragma: no cover
    from weibospider.clean_data import WeiboDataCleaner
    from weibospider.evaluate_bert import BertSentimentEvaluator
    from weibospider.project_paths import (
        BASE_DIR,
        CLEANED_DATA_DIR,
        FINAL_MODEL_DIR,
        RAW_DATA_DIR,
        REPO_ROOT,
        RESULTS_DIR,
        ensure_dir,
        to_relative_path,
    )


LABEL_DISPLAY_NAMES = {
    "positive": "正向",
    "neutral": "中性",
    "negative": "负向",
}

TENDENCY_DISPLAY_NAMES = {
    "positive": "整体偏正向",
    "neutral": "整体偏中性",
    "negative": "整体偏负向",
    "mixed": "观点分散",
}
# 模型显示名 → 文件夹路径映射
MODEL_NAME_MAP = {
    "bert-wwm": os.path.join(FINAL_MODEL_DIR, "..", "final"),      # 指向已有的 wwm 模型
    "roberta": os.path.join(FINAL_MODEL_DIR, "..", "sentiment_chinese-roberta-wwm-ext"),
    "seqgpt": os.path.join(BASE_DIR, "models", "seqgpt-560m"), 
}

def resolve_model_dir(model_name, fallback_model_dir):
    """根据模型名返回实际路径，若未匹配则使用传入的路径"""
    if model_name and model_name in MODEL_NAME_MAP:
        return MODEL_NAME_MAP[model_name]
    return fallback_model_dir

RUNTIME_OUTPUT_DIR = os.path.join(BASE_DIR, "runtime_outputs")
RUNTIME_CLEANED_DIR = os.path.join(RUNTIME_OUTPUT_DIR, "cleaned")
RUNTIME_RESULTS_DIR = os.path.join(RUNTIME_OUTPUT_DIR, "results")
RUNTIME_SENTIMENT_RESULTS_FILE = os.path.join(RUNTIME_RESULTS_DIR, "sentiment_results_all.json")


def log_message(logger, message):
    if logger is not None:
        logger(message)


def resolve_existing_path(path, search_dirs):
    if not path:
        raise ValueError("Path is required.")

    if os.path.isabs(path):
        if os.path.exists(path):
            return path
        raise FileNotFoundError(path)

    candidates = []
    for directory in search_dirs:
        candidates.append(os.path.join(directory, path))

    candidates.append(os.path.join(BASE_DIR, path))
    candidates.append(os.path.join(REPO_ROOT, path))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(path)


def build_file_record(path):
    is_directory = os.path.isdir(path)
    return {
        "name": os.path.basename(path),
        "absolute_path": path,
        "relative_path": to_relative_path(path, start=REPO_ROOT),
        "size_bytes": 0 if is_directory else os.path.getsize(path),
        "is_directory": is_directory,
        "updated_at": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds"),
    }


def list_recent_files(limit=12):
    ensure_dir(RAW_DATA_DIR)
    ensure_dir(RUNTIME_CLEANED_DIR)
    ensure_dir(RUNTIME_RESULTS_DIR)

    def collect(directories, suffixes):
        files = []
        seen = set()
        for directory in directories:
            if not os.path.isdir(directory):
                continue
            for item in os.listdir(directory):
                if item.startswith("_write_test") or item.startswith("write_test"):
                    continue
                path = os.path.join(directory, item)
                if not os.path.isfile(path):
                    continue
                if suffixes and not item.endswith(suffixes):
                    continue
                if path in seen:
                    continue
                seen.add(path)
                files.append(path)
        files.sort(key=os.path.getmtime, reverse=True)
        return [build_file_record(path) for path in files[:limit]]

    return {
        "raw": collect([RAW_DATA_DIR], (".jsonl",)),
        "cleaned": collect([RUNTIME_CLEANED_DIR, CLEANED_DATA_DIR], (".json", ".txt")),
        "results": collect([RUNTIME_RESULTS_DIR, RESULTS_DIR], (".json",)),
        "defaults": {
            "model_dir": build_file_record(FINAL_MODEL_DIR) if os.path.exists(FINAL_MODEL_DIR) else None,
        },
    }


def count_jsonl_records(file_path):
    with open(file_path, "r", encoding="utf-8") as fp:
        return sum(1 for _ in fp)


def summarize_cleaned_data(cleaned_data, total_raw_records):
    comment_type_counts = Counter(item.get("comment_type", "unknown") for item in cleaned_data)
    unique_users = {item.get("user_id") for item in cleaned_data if item.get("user_id")}
    text_lengths = [len(item.get("cleaned_content", "")) for item in cleaned_data if item.get("cleaned_content")]
    like_counts = [int(item.get("like_counts", 0) or 0) for item in cleaned_data]

    valid_records = len(cleaned_data)
    filtered_records = max(total_raw_records - valid_records, 0)
    valid_ratio = (valid_records / total_raw_records) if total_raw_records else 0.0

    return {
        "total_raw_records": total_raw_records,
        "valid_records": valid_records,
        "filtered_records": filtered_records,
        "valid_ratio": valid_ratio,
        "comment_type_distribution": dict(comment_type_counts),
        "unique_users": len(unique_users),
        "average_text_length": (sum(text_lengths) / len(text_lengths)) if text_lengths else 0.0,
        "average_like_counts": (sum(like_counts) / len(like_counts)) if like_counts else 0.0,
    }


def clean_raw_comment_file(raw_file, logger=None):
    raw_path = resolve_existing_path(raw_file, [RAW_DATA_DIR, BASE_DIR, REPO_ROOT])
    log_message(logger, f"Cleaning raw comment file: {to_relative_path(raw_path, start=REPO_ROOT)}")

    cleaner = WeiboDataCleaner(raw_path, output_dir=ensure_dir(RUNTIME_CLEANED_DIR))
    cleaned_data, stats = cleaner.process_comment_data()
    cleaner.save_results(cleaned_data, stats, "comment")

    summary = summarize_cleaned_data(cleaned_data, count_jsonl_records(raw_path))
    result = {
        "raw_file": build_file_record(raw_path),
        "cleaned_file": build_file_record(cleaner.cleaned_file),
        "stats_file": build_file_record(cleaner.stats_file),
        "summary": summary,
    }
    log_message(
        logger,
        f"Cleaning finished: {summary['valid_records']} valid comment(s) kept from {summary['total_raw_records']}.",
    )
    return result


def detect_overall_tendency(distribution):
    total = sum(distribution.values())
    if total == 0:
        return "mixed", TENDENCY_DISPLAY_NAMES["mixed"], 0.0

    ordered = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    dominant_label, dominant_count = ordered[0]
    dominant_ratio = dominant_count / total
    runner_up_count = ordered[1][1] if len(ordered) > 1 else 0
    runner_up_ratio = runner_up_count / total if total else 0.0

    if dominant_ratio < 0.45 or (dominant_ratio - runner_up_ratio) < 0.08:
        return "mixed", TENDENCY_DISPLAY_NAMES["mixed"], dominant_ratio

    return dominant_label, TENDENCY_DISPLAY_NAMES[dominant_label], dominant_ratio


def select_top_examples(predictions, label_name, limit):
    matching = [item for item in predictions if item["sentiment"] == label_name]
    matching.sort(key=lambda item: item["confidence"], reverse=True)
    return matching[:limit]

_STOP_WORDS = {
    "这个", "那个", "所以", "因为", "如果", "虽然", "然而",
    "这种", "那种", "这样", "那样", "就是", "不是", "现在", "只是",
    "已经", "还是", "然后", "可能", "应该", "自己", "什么", "怎么",
    "一个", "没有", "知道", "他们", "我们", "你们", "大家", "出来",
    "以后", "真是", "觉得", "还是", "这么", "那么", "一直", "还是",
    "时候", "有点", "真的", "特别", "非常", "比较", "可以", "需要",
}

def get_top_words(predictions, sentiment=None, topn=80):
    """从预测列表里提取某个情绪的高频词"""
    words = []
    for item in predictions:
        if sentiment is None or item["sentiment"] == sentiment:
            text = item["text"]
            seg_list = jieba.cut(text)
            words.extend([w for w in seg_list if len(w) > 1 and w not in _STOP_WORDS])
    return Counter(words).most_common(topn)

def analyze_cleaned_comments(cleaned_file, model_dir=FINAL_MODEL_DIR, model_name=None, batch_size=32, top_examples_per_label=5, logger=None):
    effective_model_dir = resolve_model_dir(model_name, model_dir)

    cleaned_path = resolve_existing_path(cleaned_file, [CLEANED_DATA_DIR, BASE_DIR, REPO_ROOT])
    model_path = resolve_existing_path(effective_model_dir, [BASE_DIR, REPO_ROOT, os.path.join(BASE_DIR, "models")])

    log_message(logger, f"Loading cleaned comments: {to_relative_path(cleaned_path, start=REPO_ROOT)}")
    with open(cleaned_path, "r", encoding="utf-8") as fp:
        cleaned_data = json.load(fp)

    comments = [item for item in cleaned_data if item.get("cleaned_content")]
    if not comments:
        raise ValueError("The cleaned file does not contain any comment text.")

    log_message(logger, f"Running sentiment model on {len(comments)} comment(s)")
    if model_name == "seqgpt":
        from evaluate_seqgpt import SeqGPTEvaluator
        evaluator = SeqGPTEvaluator(model_path)
    else:
        evaluator = BertSentimentEvaluator(default_model_dir=model_path)
    score_dict = evaluator.predict_texts_with_scores(
        [item["cleaned_content"] for item in comments],
        model_dir=model_path,
        batch_size=batch_size,
    )

    distribution = Counter()
    comment_type_distribution = Counter()
    predictions = []
    confidence_sum = 0.0
    margin_sum = 0.0
    entropy_sum = 0.0
    positive_index = evaluator.label_map["positive"]
    negative_index = evaluator.label_map["negative"]
    sentiment_index_sum = 0.0

    for index, item in enumerate(comments):
        label_name = evaluator.id_to_label[score_dict["pred_labels"][index]]
        confidence = float(score_dict["confidences"][index])
        margin = float(score_dict["margins"][index])
        entropy = float(score_dict["entropies"][index])
        probabilities = [float(value) for value in score_dict["probabilities"][index]]
        like_counts = int(item.get("like_counts", 0) or 0)

        prediction = {
            "text": item["cleaned_content"],
            "original_text": item.get("original_content", ""),
            "sentiment": label_name,
            "sentiment_display": LABEL_DISPLAY_NAMES[label_name],
            "confidence": confidence,
            "margin": margin,
            "entropy": entropy,
            "probabilities": {
                evaluator.id_to_label[label_idx]: probabilities[label_idx]
                for label_idx in range(len(probabilities))
            },
            "comment_type": item.get("comment_type", "unknown"),
            "like_counts": like_counts,
            "source_file": os.path.basename(cleaned_path),
        }
        predictions.append(prediction)
        distribution[label_name] += 1
        comment_type_distribution[prediction["comment_type"]] += 1
        confidence_sum += confidence
        margin_sum += margin
        entropy_sum += entropy
        sentiment_index_sum += probabilities[positive_index] - probabilities[negative_index]

    total = len(predictions)
    tendency_key, tendency_label, dominant_ratio = detect_overall_tendency(distribution)
    summary = {
        "total_comments": total,
        "distribution": {
            label: {
                "count": distribution.get(label, 0),
                "ratio": (distribution.get(label, 0) / total) if total else 0.0,
                "display_name": LABEL_DISPLAY_NAMES[label],
            }
            for label in evaluator.label_names
        },
        "comment_type_distribution": dict(comment_type_distribution),
        "overall_tendency": {
            "key": tendency_key,
            "label": tendency_label,
            "dominant_ratio": dominant_ratio,
        },
        "average_confidence": confidence_sum / total if total else 0.0,
        "average_margin": margin_sum / total if total else 0.0,
        "average_entropy": entropy_sum / total if total else 0.0,
        "sentiment_index": sentiment_index_sum / total if total else 0.0,
    }

    top_examples = {
        label: select_top_examples(predictions, label, top_examples_per_label)
        for label in evaluator.label_names
    }

    ensure_dir(RUNTIME_RESULTS_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(
        RUNTIME_RESULTS_DIR,
        f"{os.path.splitext(os.path.basename(cleaned_path))[0]}_analysis_{timestamp}.json",
    )
    payload = {
        "cleaned_file": to_relative_path(cleaned_path, start=REPO_ROOT),
        "model_dir": to_relative_path(model_path, start=REPO_ROOT),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "top_examples": top_examples,
        "predictions": predictions,
        "wordcloud": get_top_words(predictions, topn=100),
    }
    with open(result_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    with open(RUNTIME_SENTIMENT_RESULTS_FILE, "w", encoding="utf-8") as fp:
        json.dump(predictions, fp, ensure_ascii=False, indent=2)

    log_message(logger, f"Analysis finished: {tendency_label}, results saved to {to_relative_path(result_path, start=REPO_ROOT)}")
    return {
        "cleaned_file": build_file_record(cleaned_path),
        "model_dir": build_file_record(model_path),
        "result_file": build_file_record(result_path),
        "summary": summary,
        "top_examples": top_examples,
        "predictions": predictions,
        "wordcloud": get_top_words(predictions, topn=100),
    }

def analyze_keyword_tweets(keyword, model_dir=FINAL_MODEL_DIR, model_name=None, batch_size=32,
                           top_examples_per_label=5, logger=None, raw_file_path=None):
    effective_model_dir = resolve_model_dir(model_name, model_dir)
    model_path = resolve_existing_path(effective_model_dir, [BASE_DIR, REPO_ROOT])

    import glob
    from clean_data import WeiboDataCleaner
    from evaluate_bert import BertSentimentEvaluator
    from collections import Counter

    # 1. 优先使用传入的路径
    if raw_file_path and os.path.exists(raw_file_path):
        candidate_files = [raw_file_path]
        log_message(logger, f"使用指定的文件: {to_relative_path(raw_file_path, start=REPO_ROOT)}")
    else:
        # 原来的搜索逻辑（保留给独立分析用）
        pattern = os.path.join(RAW_DATA_DIR, f"tweet_spider_by_keyword_{keyword}_*.jsonl")
        candidate_files = glob.glob(pattern)
        if not candidate_files:
            all_files = [
                f for f in os.listdir(RAW_DATA_DIR)
                if f.startswith("tweet_spider_by_keyword_") and f.endswith(".jsonl") and keyword in f
            ]
            candidate_files = [os.path.join(RAW_DATA_DIR, f) for f in all_files]

        if not candidate_files:
            raise FileNotFoundError(f"未找到关键词 '{keyword}' 的原始微博数据，请先执行关键词抓取。")
    
    # 2. 读取所有微博正文
    tweets = []
    for fpath in candidate_files:
        log_message(logger, f"读取文件: {to_relative_path(fpath, start=REPO_ROOT)}")
        with open(fpath, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    tweets.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not tweets:
        raise ValueError(f"关键词 '{keyword}' 的原始文件中没有有效微博。")

    # 3. 清洗文本（直接复用 WeiboDataCleaner 的 clean_text 方法）
    cleaner = WeiboDataCleaner("")          # 输入空路径仅为了调用实例方法
    cleaned_items = []
    for t in tweets:
        raw_content = t.get("content", "")
        cleaned = cleaner.clean_text(raw_content).strip()
        if len(cleaned) >= 2:               # 过滤太短的文本
            cleaned_items.append({
                "text": cleaned,
                "original_text": raw_content,   # 保留原始内容用于展示
                "like_counts": t.get("attitudes_count", 0),    # 点赞数
                "comments_count": t.get("comments_count", 0),
                "reposts_count": t.get("reposts_count", 0),
                "created_at": t.get("created_at", ""),
                "user_name": t.get("user", {}).get("nick_name", ""),
            })

    if not cleaned_items:
        raise ValueError("清洗后无可分析的微博正文。")

    log_message(logger, f"共清洗出 {len(cleaned_items)} 条有效微博")

    # 4. 调用情感模型
    model_path = resolve_existing_path(effective_model_dir, [BASE_DIR, REPO_ROOT])
    if model_name == "seqgpt":
        from evaluate_seqgpt import SeqGPTEvaluator
        evaluator = SeqGPTEvaluator(model_path)
    else:
        evaluator = BertSentimentEvaluator(default_model_dir=model_path)
    texts = [item["text"] for item in cleaned_items]
    score_dict = evaluator.predict_texts_with_scores(
        texts, model_dir=model_path, batch_size=batch_size
    )

    distribution = Counter()
    predictions = []
    confidence_sum = 0.0
    margin_sum = 0.0
    entropy_sum = 0.0
    positive_index = evaluator.label_map["positive"]
    negative_index = evaluator.label_map["negative"]
    sentiment_index_sum = 0.0

    for idx, item in enumerate(cleaned_items):
        label_name = evaluator.id_to_label[score_dict["pred_labels"][idx]]
        confidence = float(score_dict["confidences"][idx])
        margin = float(score_dict["margins"][idx])
        entropy = float(score_dict["entropies"][idx])
        probabilities = [float(v) for v in score_dict["probabilities"][idx]]
        like_counts = int(item.get("like_counts", 0) or 0)

        prediction = {
            "text": item["text"],
            "original_text": item["original_text"],
            "sentiment": label_name,
            "sentiment_display": LABEL_DISPLAY_NAMES[label_name],
            "confidence": confidence,
            "margin": margin,
            "entropy": entropy,
            "probabilities": {
                evaluator.id_to_label[i]: probabilities[i] for i in range(len(probabilities))
            },
            "like_counts": like_counts,
            "user_name": item["user_name"],
            "created_at": item["created_at"],
        }
        predictions.append(prediction)
        distribution[label_name] += 1
        confidence_sum += confidence
        margin_sum += margin
        entropy_sum += entropy
        sentiment_index_sum += probabilities[positive_index] - probabilities[negative_index]

    total = len(predictions)
    tendency_key, tendency_label, dominant_ratio = detect_overall_tendency(distribution)

    summary = {
        "total_comments": total,   # 字段名保持一致，便于前端渲染
        "distribution": {
            label: {
                "count": distribution.get(label, 0),
                "ratio": (distribution.get(label, 0) / total) if total else 0.0,
                "display_name": LABEL_DISPLAY_NAMES[label],
            }
            for label in evaluator.label_names
        },
        "comment_type_distribution": {},  # 微博无此概念，置空
        "overall_tendency": {
            "key": tendency_key,
            "label": tendency_label,
            "dominant_ratio": dominant_ratio,
        },
        "average_confidence": confidence_sum / total if total else 0.0,
        "average_margin": margin_sum / total if total else 0.0,
        "average_entropy": entropy_sum / total if total else 0.0,
        "sentiment_index": sentiment_index_sum / total if total else 0.0,
    }

    top_examples = {
        label: select_top_examples(predictions, label, top_examples_per_label)
        for label in evaluator.label_names
    }

    # 保存结果到 runtime_outputs（便于回溯）
    ensure_dir(RUNTIME_RESULTS_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(
        RUNTIME_RESULTS_DIR,
        f"keyword_{keyword}_sentiment_{timestamp}.json",
    )
    with open(result_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "keyword": keyword,
                "model_dir": to_relative_path(model_path, start=REPO_ROOT),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "summary": summary,
                "top_examples": top_examples,
                "predictions": predictions,
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )

    log_message(logger, f"关键词分析完成：{tendency_label}，结果已保存")
    return {
        "summary": summary,
        "top_examples": top_examples,
        "result_file": build_file_record(result_path),
        "wordcloud": get_top_words(predictions, topn=100)
    }