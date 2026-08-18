# WeiboSpider - 微博舆情爬虫与情感分析系统

基于 Scrapy + BERT/RoBERTa + SeqGPT 的微博舆情采集与情感分析系统，支持微博评论、用户信息、关键词搜索爬取，中文文本清洗，三分类情感分析（正向/中性/负向），提供 CLI 和 Web 控制台两种使用方式。毕设项目。

---

## 项目结构

```
WeiboSpider-master/
├── scrapy.cfg
├── requirements.txt
├── backup/                            # 备用爬虫
└── weibospider/
    ├── settings.py                    # Scrapy 配置
    ├── project_paths.py               # 路径管理
    ├── run_spider.py                  # CLI 入口
    ├── middlewares.py                 # 代理中间件
    ├── pipelines.py                   # JSONL 输出
    ├── clean_data.py                  # 文本清洗
    ├── prepare_labels.py              # 标注候选生成
    ├── train_bert.py / train2.py      # 两阶段 BERT 微调
    ├── evaluate_bert.py               # BERT/RoBERTa 评估
    ├── evaluate_seqgpt.py             # SeqGPT 评估
    ├── pipeline_service.py            # 清洗+分析编排
    ├── quick_test.py                  # 快速推理测试
    ├── web_console.py                 # Web 控制台后端
    ├── config/cookie.txt              # 微博 Cookie
    ├── spiders/
    │   ├── common.py                  # base62解码、推文/用户解析
    │   ├── comment.py                 # 评论爬虫
    │   ├── tweet_by_keyword.py        # 关键词搜索爬虫
    │   └── user.py                    # 用户信息爬虫
    ├── data/
    │   ├── raw/                       # 原始 JSONL
    │   ├── cleaned/                   # 清洗后 JSON
    │   ├── labels/to_label.csv        # 300条人工标注
    │   └── results/                   # 分析结果
    ├── models/                        # 模型文件
    ├── reports/                       # 评估报告
    ├── runtime_outputs/               # Web控制台运行时输出
    └── webui/                         # 前端页面
```

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：Scrapy 2.5, transformers 4.46, torch>=2.4, jieba, pandas, scikit-learn, snownlp

### 配置 Cookie

1. 浏览器登录 weibo.com
2. F12 -> Network -> 任意请求 -> 复制完整 Cookie
3. 粘贴到 `weibospider/config/cookie.txt`

Cookie 过期后需手动重新获取。

### CLI 运行爬虫

```bash
cd weibospider

# 爬取评论
python run_spider.py comment --tweet-ids QxaWwkI8f --count 20

# 爬取用户信息
python run_spider.py user --user-ids 1749127163

# 关键词搜索
python run_spider.py tweet_by_keyword --keywords "丽江" --start-time "2026-01-01 00:00" --end-time "2026-01-03 23:00" --split-by-hour true
```

数据保存在 `data/raw/`，JSONL 格式。

### Web 控制台

```bash
cd weibospider
python web_console.py --host 127.0.0.1 --port 8765
```

浏览器打开 `http://127.0.0.1:8765`，支持一键全流程：爬取->清洗->情感分析->可视化（饼图+词云）。

---

## 情感分析模型

### 两阶段训练

阶段一：在 `dirtycomputer/weibo_senti_100k`（10万条微博）上做二分类（正/负）预训练。

阶段二：将二分类分类头通过权重插值扩展为三分类（正/中/负），在 300 条人工标注数据上细调。

权重插值公式：

```
W_neutral = (W_positive + W_negative) / 2
b_neutral = (b_positive + b_negative) / 2
```

相比随机初始化三分类头，该方法在小样本下收敛更快、泛化更好。

### 支持的模型

- bert-wwm：hfl/chinese-bert-wwm，判别式，两阶段微调
- roberta：hfl/chinese-roberta-wwm-ext，判别式，两阶段微调
- seqgpt：seqgpt-560m，生成式，3-shot 提示 + 首 token logit 分类，无需训练

### 训练与评估

```bash
python train_bert.py       # 两阶段训练
python evaluate_bert.py    # BERT 评估
python evaluate_seqgpt.py  # SeqGPT 评估
python quick_test.py       # 单条推理测试
```

---

## 倾向性判定

规则化判定（`pipeline_service.py::detect_overall_tendency`）：

- 某标签占比 >= 45% 且领先第二名 >= 8% -> 判定为该倾向
- 否则 -> "观点分散"

输出：整体偏正向 / 整体偏中性 / 整体偏负向 / 观点分散。同时输出平均置信度、边际、熵、情感指数。

---

## 爬虫技术

### 反爬策略

- Cookie 认证，模拟登录态
- 请求头伪装（Chrome UA、Referer 等）
- 下载延迟 1s，并发数 16
- 禁用重定向中间件
- IP 代理接口已预留（`middlewares.py`），需自行实现 `fetch_proxy()`

### 短 ID 解码

微博使用自定义 base62（字符集 0-9a-zA-Z）编码数字 ID。解码时 4 位一组分割，分别 base62 转十进制，拼接补零得到完整 mid。见 `spiders/common.py`。

### 关键词搜索时间拆分

`tweet_by_keyword.py` 支持按小时拆分大时间范围，逐段爬取，绕过搜索结果页数限制。

---

## 数据流程

```
微博 AJAX API
    -> Scrapy Spider (Cookie认证)
    -> data/raw/*.jsonl
    -> WeiboDataCleaner (10步清洗)
    -> data/cleaned/*.json
    -> BertSentimentEvaluator / SeqGPTEvaluator
    -> data/results/*.json (分布/倾向/Top示例/词云)
    -> Web Console (可视化)
```

---

## 核心工作

1. 两阶段微调 + 权重插值初始化三分类头，小样本场景下提升性能
2. BERT vs SeqGPT 双范式对比（判别式 vs 生成式）
3. 爬取-清洗-分析-可视化全流程 Web 控制台
4. 多维评估指标（置信度、边际、熵、情感指数）

---

## 不足与改进

- Cookie 需手动维护 -> 自动化登录
- IP 代理未实现 -> 接入代理池
- 标注数据仅 300 条 -> 半监督/主动学习扩充
- 仅三分类 -> 细粒度或方面级情感分析
- 单机爬取 -> Scrapy-Redis 分布式
- 仅支持微博 -> 对接其他平台

---

## License

MIT License
