# TechQA Chunk 切分 BM25 实验

## 实验范围

- 语料：`data/processed/techqa/corpus.jsonl` 全部 28,482 篇文档。
- 查询：`data/processed/techqa/train_queries.jsonl` 全部 450 条 TRAIN 查询。
- DEV：未读取、未评测。
- 索引文本：文档标题 + Chunk 正文。
- 文档聚合：每篇文档使用最高分 Chunk 作为代表。
- 同分规则：按文档 ID 升序。
- 评分：Recall@1、Recall@5、MRR@10，分母固定为 450。
- 重排：关闭。

## BM25 与分词

- 实现：Python `sqlite3` + SQLite FTS5 `bm25()`。
- SQLite：3.45.1。
- BM25：`k1=1.2`、`b=0.75`；FTS5 的实现固定使用这两个参数，脚本将其作为受检常量。
- 每套 Chunk 配置建立独立 FTS5 索引，分别重算 IDF 和平均块长。
- 分词：Python Unicode 正则 `[^\W_]+` 匹配连续字母或数字，并执行 `casefold()`。
- 下划线及其他标点均作为分隔符。
- 不做词干提取，不移除停用词。
- 查询中的重复词在构造 OR 检索表达式前去重。
- 参考文档 ID 只在检索完成后评分，不进入查询或索引。

## 基线检查

自然段落基线首先单独运行并通过以下检查：

- 实际处理 450 条 TRAIN 查询。
- 每条查询返回 10 个不同文档。
- 返回文档 ID 全部存在于共享语料。
- 逐查询排名文件为 450 行。

## 结果

正文块长按 Unicode 字符数统计，查询耗时为 450 条查询的平均端到端检索耗时。

| 配置 | Chunk 数 | 块长 P50 | 块长 P95 | Recall@1 | Recall@5 | MRR@10 | 建索引（秒） | 平均查询（毫秒） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 自然段落 | 595,996 | 67 | 562 | 0.4333 | 0.5667 | 0.4910 | 10.840 | 805.160 |
| 固定 600 / 0 | 182,835 | 600 | 600 | 0.4333 | 0.5933 | 0.5008 | 8.002 | 356.941 |
| 固定 600 / 150 | 229,636 | 600 | 600 | 0.4200 | 0.5822 | 0.4897 | 9.907 | 471.274 |
| 段落合并 600 / 0 | 211,869 | 517 | 600 | 0.4311 | 0.5800 | 0.4997 | 8.274 | 392.128 |
| 固定 1200 / 300 | 117,259 | 1,200 | 1,200 | 0.4378 | 0.5822 | 0.5019 | 8.640 | 294.566 |

计时是本次本机实测结果，主要用于本轮配置间相对比较。

## 指定对照

### 固定 600 / 0 与段落合并 600 / 0

固定切分的 Recall@1 高 0.0022、Recall@5 高 0.0133、MRR@10 高约 0.0011；平均查询也更快（356.9 ms 对 392.1 ms）。在相同 600 上限和零重叠下，本轮没有观察到段落边界带来检索指标提升。

### 固定 600 / 0 与固定 600 / 150

加入 150 字符重叠后，Chunk 数从 182,835 增至 229,636。Recall@1 下降 0.0133、Recall@5 下降 0.0111、MRR@10 下降约 0.0111，平均查询时间从 356.9 ms 增至 471.3 ms。本轮重叠没有带来收益。

### 固定 600 / 150 与固定 1200 / 300

两者均为 25% 重叠。1200 / 300 的 Recall@1 高 0.0178，Recall@5 相同，MRR@10 高约 0.0122；Chunk 数约减少 48.9%，平均查询从 471.3 ms 降至 294.6 ms。本轮长块配置整体更有竞争力。

以上只汇报本轮 BM25 基线结果，不在报告中决定进入联合实验的三套配置。

## 复现命令

本轮先单独运行自然段落基线：

```bash
python3 experiments/chunk/run_bm25_chunk_experiments.py \
  --configs natural_paragraph
```

基线验证通过后运行其余四组：

```bash
python3 experiments/chunk/run_bm25_chunk_experiments.py \
  --configs fixed_600_0 fixed_600_150 paragraph_merge_600_0 fixed_1200_300
```

也可以一次重跑全部五组：

```bash
python3 experiments/chunk/run_bm25_chunk_experiments.py
```

## 产物

- 汇总：`experiments/chunk/bm25_char_results/comparison.json`
- 各组指标：`experiments/chunk/bm25_char_results/<config>/metrics.json`
- 逐查询排名：`experiments/chunk/bm25_char_results/<config>/query_rankings.jsonl`
- 可重建索引：`data/processed/techqa/bm25_chunk_indexes/<config>.sqlite`

逐查询文件保存 top-10 不同文档、代表 Chunk、BM25 分数和参考文档在 top-10 内的排名；排名为 `null` 表示参考文档未进入 top-10。
