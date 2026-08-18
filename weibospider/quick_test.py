# quick_test.py
from transformers import BertTokenizer, BertForSequenceClassification
import torch
try:
    from project_paths import FINAL_MODEL_DIR
except ImportError:  # pragma: no cover
    from weibospider.project_paths import FINAL_MODEL_DIR

tokenizer = BertTokenizer.from_pretrained(FINAL_MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(FINAL_MODEL_DIR)
model.eval()

text = "是不是有病啊？"
inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(outputs.logits, dim=-1).item()

id_to_label = {0: '正面', 1: '中性', 2: '负面'}
print(f"预测结果: {id_to_label[pred]}")
print(f"正面概率: {probs[0][0]:.3f}")
print(f"中性概率: {probs[0][1]:.3f}")
print(f"负面概率: {probs[0][2]:.3f}")
