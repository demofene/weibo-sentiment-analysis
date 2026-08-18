# clean_data.py (改进版)
import json
import re
import os
from datetime import datetime
import pandas as pd

try:
    from project_paths import CLEANED_DATA_DIR, RAW_DATA_DIR, REPO_ROOT, ensure_dir, to_relative_path
except ImportError:  # pragma: no cover
    from weibospider.project_paths import CLEANED_DATA_DIR, RAW_DATA_DIR, REPO_ROOT, ensure_dir, to_relative_path

class WeiboDataCleaner:
    def __init__(self, input_file, output_dir=CLEANED_DATA_DIR):
        self.input_file = input_file
        self.output_dir = output_dir
        ensure_dir(output_dir)
        
        base_name = os.path.basename(input_file).replace('.jsonl', '')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.cleaned_file = f"{output_dir}/{base_name}_cleaned_{timestamp}.json"
        self.stats_file = f"{output_dir}/{base_name}_stats_{timestamp}.txt"
    
    def clean_text(self, text):
        """
        清洗文本内容
        """
        if not text:
            return ""
        
        # 1. 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 2. 去除URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # 3. 去除@用户
        text = re.sub(r'@[\u4e00-\u9fa5a-zA-Z0-9_-]+', '', text)
        
        # 4. 去除#话题#（保留话题文字，但去除#号）
        text = re.sub(r'#([^#]+)#', r'\1', text)

        # 5. 去除"回复:"、"回复@用户名:"等格式
        text = re.sub(r'^回复[:：]?\s*', '', text)
        text = re.sub(r'回复@[\u4e00-\u9fa5a-zA-Z0-9_-]+[:：]?\s*', '', text)

        # 6. 去除"图片评论"、"视频评论"等
        text = re.sub(r'图片评论|视频评论|转发微博|分享图片', '', text)
        
        # 7. 去除微博表情 [xxx] 格式
        text = re.sub(r'\[[^\[\]]+\]', '', text)
        
        # 8. 去除颜文字和其他特殊表情符号
        text = re.sub(r'[\(（][^\(\)（）]+[\)）]', '', text)
        text = re.sub(r'[\u2600-\u27BF]', '', text)
        
        # 9. 去除多余空格和换行
        text = re.sub(r'\s+', ' ', text)
        
        # 10. 去除转发标志 //@用户名: 
        text = re.sub(r'//\s*@[\u4e00-\u9fa5a-zA-Z0-9_-]+\s*[:：]', '', text)
        return text.strip()
    
    def extract_emojis_for_stats(self, text):
        """
        统计原文中的表情符号（用于统计报告）
        """
        emojis = re.findall(r'\[[^\[\]]+\]', text)
        return emojis
    
    def is_valid_comment(self, text):
        """
        判断评论是否有效
        """
        if not text or len(text) < 2:
            return False
        
        # 过滤纯表情的评论（清洗后为空）
        if text == '':
            return False
        
        # 过滤无意义评论
        meaningless = ['转发微博', '评论配图', '赞', '回复', '分享图片']
        if text in meaningless:
            return False
        
        if text.isdigit():
            return False
        
        return True
    
    def get_reply_info(self, raw):
        """
        获取回复对象的信息
        """
        reply = raw.get('reply_comment')
        if not reply:
            return None
        return {
            'user_id': str(reply.get('user', {}).get('id', '')),
            'user_name': reply.get('user', {}).get('screen_name', ''),
            'content': reply.get('text', '')[:100]  # 回复的内容预览
        }

    def process_comment_data(self):
        """
        处理评论数据
        """
        cleaned_data = []
        stats = {
            '总评论数': 0,
            '有效评论数': 0,
            '被过滤的评论数': 0,
            '直接评论数': 0,
            '回复评论数': 0,
            '移除的表情数量': 0,
            '评论长度统计': [],
            '点赞数统计': [],
            '用户数': set(),
            '时间分布': [],
            '常见表情': []
        }
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    raw = json.loads(line)
                    stats['总评论数'] += 1
                    
                    original_text = raw.get('content', '')
                    
                    # 统计原文中的表情
                    emojis = self.extract_emojis_for_stats(original_text)
                    stats['移除的表情数量'] += len(emojis)
                    if emojis:
                        stats['常见表情'].extend(emojis)
                    
                    # 清洗文本
                    cleaned_text = self.clean_text(original_text)
                    
                    # 判断是否为回复评论
                    is_reply = 'reply_comment' in raw and raw['reply_comment'] is not None

                    comment = {
                        'comment_id': raw.get('_id', ''),
                        'user_id': raw.get('comment_user', {}).get('_id', ''),
                        'user_name': raw.get('comment_user', {}).get('nick_name', ''),
                        'original_content': original_text,  # 保留原文，便于对比
                        'cleaned_content': cleaned_text,     # 清洗后的文本
                        'like_counts': raw.get('like_counts', 0),
                        'created_at': raw.get('created_at', ''),
                        'ip_location': raw.get('ip_location', ''),
                        'user_followers': raw.get('comment_user', {}).get('followers_count', 0),
                        'user_verified': raw.get('comment_user', {}).get('verified', False),
                        'comment_type': 'reply' if is_reply else 'direct',
                        'removed_emojis': emojis  # 记录被移除的表情
                    }
                    
                    # 有效性检查
                    if self.is_valid_comment(cleaned_text):
                        stats['有效评论数'] += 1
                        if is_reply:
                            stats['回复评论数'] += 1
                        else:
                            stats['直接评论数'] += 1
                        stats['评论长度统计'].append(len(cleaned_text))
                        stats['点赞数统计'].append(comment['like_counts'])
                        stats['用户数'].add(comment['user_id'])
                        if comment['created_at']:
                            stats['时间分布'].append(comment['created_at'][:10])
                        
                        cleaned_data.append(comment)
                    else:
                        stats['被过滤的评论数'] += 1
                        
                except Exception as e:
                    print(f"处理出错: {e}")
                    continue
        
        # 统计常见表情
        from collections import Counter
        if stats['常见表情']:
            stats['常见表情'] = Counter(stats['常见表情']).most_common(10)
        
        return cleaned_data, stats
    
    def save_results(self, cleaned_data, stats, data_type='comment'):
        """
        保存清洗后的数据和统计信息
        """
        # 保存清洗后的数据
        with open(self.cleaned_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        
        # 保存统计报告
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("数据清洗统计报告\n")
            f.write("原始文件: " + to_relative_path(self.input_file, start=REPO_ROOT) + "\n")
            f.write("清洗时间: " + str(datetime.now()) + "\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("数据量统计:\n")
            f.write("   总评论数: " + str(stats['总评论数']) + "\n")
            f.write("   有效评论数: " + str(stats['有效评论数']) + "\n")
            f.write("   被过滤评论数: " + str(stats['被过滤的评论数']) + "\n")
            f.write("   有效率: {:.2f}%\n".format(stats['有效评论数']/stats['总评论数']*100))
            f.write("   独立用户数: " + str(len(stats['用户数'])) + "\n")
            f.write("   移除的表情总数: " + str(stats['移除的表情数量']) + "\n")

            f.write("\n评论类型统计:\n")
            f.write("   直接评论: " + str(stats['直接评论数']) + " 条\n")
            f.write("   回复评论: " + str(stats['回复评论数']) + " 条\n")
            if stats['有效评论数'] > 0:
                f.write("   回复比例: {:.2f}%\n".format(stats['回复评论数']/stats['有效评论数']*100))
            
            if stats['评论长度统计']:
                avg_len = sum(stats['评论长度统计']) / len(stats['评论长度统计'])
                f.write("\n评论长度统计:\n")
                f.write("   平均长度: {:.1f} 字\n".format(avg_len))
                f.write("   最长评论: {} 字\n".format(max(stats['评论长度统计'])))
                f.write("   最短评论: {} 字\n".format(min(stats['评论长度统计'])))
            
            if stats['点赞数统计']:
                avg_like = sum(stats['点赞数统计']) / len(stats['点赞数统计'])
                f.write("\n点赞统计:\n")
                f.write("   平均点赞: {:.1f}\n".format(avg_like))
                f.write("   最多点赞: {}\n".format(max(stats['点赞数统计'])))
            
            if stats['时间分布']:
                from collections import Counter
                date_counts = Counter(stats['时间分布'])
                f.write("\n日期分布 (前10天):\n")
                for date, count in list(date_counts.most_common(10)):
                    f.write("   {}: {}条\n".format(date, count))
        
        print("\n清洗完成！")
        print("清洗后的数据: " + to_relative_path(self.cleaned_file, start=REPO_ROOT))
        print("统计报告: " + to_relative_path(self.stats_file, start=REPO_ROOT))

    def run(self):
        print(f"开始清洗文件: {self.input_file}")
        
        if 'comment' in self.input_file.lower():
            cleaned_data, stats = self.process_comment_data()
            self.save_results(cleaned_data, stats, 'comment')
        elif 'tweet' in self.input_file.lower():
            # 如果以后需要处理微博数据，可以类似添加
            print("暂不支持微博数据清洗")
        else:
            print("未知的数据类型")


def main():
    output_dir = RAW_DATA_DIR
    if not os.path.exists(output_dir):
        print("找不到原始数据文件夹 data/raw！")
        return
    
    files = [f for f in os.listdir(output_dir) if f.endswith('.jsonl')]
    
    if not files:
        print("没有找到数据文件！")
        return
    
    files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
    
    print("找到的数据文件：")
    for i, f in enumerate(files[:5]):
        size = os.path.getsize(os.path.join(output_dir, f)) / 1024
        print(f"{i+1}. {f} ({size:.1f} KB)")
    
    choice = input("\n请选择要清洗的文件编号: ").strip()
    if not choice:
        choice = '1'
    
    try:
        idx = int(choice) - 1
        selected_file = os.path.join(output_dir, files[idx])
        
        cleaner = WeiboDataCleaner(selected_file)
        cleaner.run()
        
    except (ValueError, IndexError):
        print("输入无效")

if __name__ == '__main__':
    main()
