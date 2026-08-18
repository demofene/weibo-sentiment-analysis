import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter

class SeqGPTEvaluator:
    def __init__(self, model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_path is None:
            from project_paths import BASE_DIR
            model_path = os.path.join(BASE_DIR, "models", "seqgpt-560m")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device == "cpu":
            self.model.to(self.device)
        self.model.eval()

        # 标签映射（与项目保持一致）
        self.label_names = ["positive", "neutral", "negative"]
        self.label_map = {label: idx for idx, label in enumerate(self.label_names)}
        self.id_to_label = {idx: label for label, idx in self.label_map.items()}

        # 情感词对应的 token id（用于概率计算）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.sentiment_tokens = {
            "正面": self.tokenizer.encode("正面", add_special_tokens=False)[-1],
            "中性": self.tokenizer.encode("中性", add_special_tokens=False)[-1],
            "负面": self.tokenizer.encode("负面", add_special_tokens=False)[-1],
        }

    def predict_texts_with_scores(self, texts, model_dir=None, batch_size=1):
        """
        与 BertSentimentEvaluator 接口一致，返回字典：
        {
            "pred_labels": list[int],
            "confidences": list[float],
            "margins": list[float],
            "entropies": list[float],
            "probabilities": list[list[float]]
        }
        """
        results = {
            "pred_labels": [],
            "confidences": [],
            "margins": [],
            "entropies": [],
            "probabilities": []
        }

        prompt_template = (
            "你是一个情感分析专家。请阅读下面的中文微博，并判断其情感倾向。"
            "只能回答以下三个词之一：正面、中性、负面。不要解释，不要标点。\n\n"
            "示例：\n"
            "文本：今天天气真好！\n答案：正面\n\n"
            "文本：这破手机又死机了。\n答案：负面\n\n"
            "文本：我在吃饭。\n答案：中性\n\n"
            "现在请判断：\n文本：{}\n答案："
        )

        for text in texts:
            prompt = prompt_template.format(text)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                    max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2,
                    do_sample=False,
                    num_beams=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            gen_tokens = outputs.sequences[0, inputs.input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            # 解析情感标签
            if "正面" in generated_text:
                label = "positive"
            elif "负面" in generated_text:
                label = "negative"
            else:
                label = "neutral"

            label_idx = self.label_map[label]
            results["pred_labels"].append(label_idx)

            # 使用第一个生成步的 logits 计算各类别概率
            scores = outputs.scores
            if len(scores) >= 1:
                first_logits = scores[0][0]  # shape: vocab_size
                probs = torch.nn.functional.softmax(first_logits, dim=-1)
                prob_pos = probs[self.sentiment_tokens["正面"]].item()
                prob_neg = probs[self.sentiment_tokens["负面"]].item()
                prob_neu = probs[self.sentiment_tokens["中性"]].item()
                raw_probs = [prob_pos, prob_neu, prob_neg]
                # 归一化
                normalized = torch.tensor(raw_probs)
                normalized = normalized / normalized.sum()
                confidence = normalized[label_idx].item()
                margin = normalized.max().item() - normalized.median().item()
                entropy = -torch.sum(normalized * torch.log(normalized + 1e-9)).item()
            else:
                normalized = torch.tensor([1/3, 1/3, 1/3])
                confidence = 0.5
                margin = 0.0
                entropy = 1.099  # 最大熵

            results["confidences"].append(confidence)
            results["margins"].append(margin)
            results["entropies"].append(entropy)
            results["probabilities"].append(normalized.tolist())

        return results