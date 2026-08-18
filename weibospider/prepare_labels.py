# prepare_labels.py (改进版)
import json
import os
import random
import csv
from snownlp import SnowNLP

try:
    from project_paths import CLEANED_DATA_DIR, LABELS_DIR, LABEL_FILE, ensure_dir
except ImportError:  # pragma: no cover
    from weibospider.project_paths import CLEANED_DATA_DIR, LABELS_DIR, LABEL_FILE, ensure_dir

def prepare_labeling_file(sample_size=300, selected_file=None):
    """
    从清洗后的数据中抽取样本，生成待标注的Excel文件
    """
    cleaned_dir = CLEANED_DATA_DIR
    if not os.path.exists(cleaned_dir):
        print("没有找到清洗数据文件夹 data/cleaned")
        return
    
    # 找到所有清洗后的 json 文件
    files = [f for f in os.listdir(cleaned_dir) if f.endswith('.json')]
    
    if not files:
        print("没有找到清洗后的数据文件")
        return
    
    # 如果没有指定文件，让用户选择
    if selected_file is None:
        print("\n找到以下数据文件:")
        for i, f in enumerate(files):
            filepath = os.path.join(cleaned_dir, f)
            with open(filepath, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            print(f"  {i+1}. {f} ({len(data)} 条)")
        
        choice = input("\n请选择要使用的文件编号 (直接回车选全部合并): ").strip()
        
        if choice == "":
            # 合并所有文件
            all_data = []
            for f in files:
                filepath = os.path.join(cleaned_dir, f)
                with open(filepath, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                    all_data.extend(data)
            print(f"\n合并后共 {len(all_data)} 条评论")
            data_to_use = all_data
            source_desc = "合并所有文件"
        else:
            try:
                idx = int(choice) - 1
                selected = files[idx]
                filepath = os.path.join(cleaned_dir, selected)
                with open(filepath, 'r', encoding='utf-8') as fp:
                    data_to_use = json.load(fp)
                source_desc = selected
                print(f"\n使用文件: {selected} ({len(data_to_use)} 条)")
            except (ValueError, IndexError):
                print("输入无效，使用最新文件")
                files.sort(reverse=True)
                filepath = os.path.join(cleaned_dir, files[0])
                with open(filepath, 'r', encoding='utf-8') as fp:
                    data_to_use = json.load(fp)
                source_desc = files[0]
    else:
        # 直接使用指定的文件
        filepath = os.path.join(cleaned_dir, selected_file)
        with open(filepath, 'r', encoding='utf-8') as fp:
            data_to_use = json.load(fp)
        source_desc = selected_file
        print(f"\n使用指定文件: {selected_file} ({len(data_to_use)} 条)")
    
    # 按情感初筛分类
    positive_samples = []
    negative_samples = []
    neutral_samples = []
    
    for item in data_to_use:
        text = item.get('cleaned_content', '')
        if not text or len(text) < 5:
            continue
        
        s = SnowNLP(text)
        score = s.sentiments
        
        if score > 0.6:
            positive_samples.append((text, item.get('comment_type', 'unknown')))
        elif score < 0.4:
            negative_samples.append((text, item.get('comment_type', 'unknown')))
        else:
            neutral_samples.append((text, item.get('comment_type', 'unknown')))
    
    # 各类别抽取相同数量
    sample_per_class = sample_size // 3
    sample = []
    
    sample.extend(positive_samples[:sample_per_class])
    sample.extend(negative_samples[:sample_per_class])
    sample.extend(neutral_samples[:sample_per_class])
    
    # 打乱顺序
    random.shuffle(sample)
    
    print(f"\n抽取样本分布:")
    print(f"  正面类候选: {len(positive_samples)} 条")
    print(f"  负面类候选: {len(negative_samples)} 条")
    print(f"  中性类候选: {len(neutral_samples)} 条")
    print(f"  最终抽取: {len(sample)} 条")
    print(f"  数据来源: {source_desc}")
    
    # 生成CSV文件
    ensure_dir(LABELS_DIR)
    output_file = LABEL_FILE
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['编号', '评论内容', '情感标签', '备注'])
        writer.writerow(['', '', '填写说明: positive/negative/neutral', '可选'])
        
        for idx, (text, comment_type) in enumerate(sample):
            writer.writerow([idx + 1, text, '', comment_type])
    
    print(f"\n已生成标注文件: {output_file}")

if __name__ == '__main__':
    # 可以指定文件名，如 prepare_labeling_file(selected_file='comment_xxx_cleaned.json')
    # 或者不指定，让程序交互选择
    prepare_labeling_file(sample_size=300)
