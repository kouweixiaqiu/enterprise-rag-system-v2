# TechQA 数据来源与核验记录

## 官方来源

- 发布方：IBM。
- 官方代码与数据说明：<https://github.com/IBM/techqa>。
- 数据论文：<https://aclanthology.org/2020.acl-main.117/>。
- 官方仓库当前指向的托管发布页：<https://huggingface.co/datasets/PrimeQA/TechQA/tree/main>。
- 官方代码仓库及托管发布页标注 Apache-2.0；原始数据包内另附 `CDLA-Permissive-v1.0.pdf`，语料使用应以包内许可为准。

官方发布包 `TechQA.tar.gz` 同时包含训练/开发问答数据与 Technotes 语料。官方仓库给出的基线文件名为：

- `training_Q_A.json`
- `dev_Q_A.json`
- `training_dev_technotes.json`

下载产物的固定版本、大小与 SHA-256 记录在 `data/manifests/techqa_source.json`。

## 本轮核验范围

本轮仅确认原始发布包的来源、版本和完整性，原样解包到 `data/raw/`，并确认其中的原始问答与文档语料文件可读。

以下工作明确留到后续数据处理阶段：

- TRAIN/DEV 分组与答案标记统计。
- 记录筛选和跨划分去重。
- 各规划条数的实测核验。
- 问答参考文档 ID 与语料的映射核验。

## 本地原始文件

保存位置：`data/raw/techqa/`。

归档 `TechQA.tar.gz`：

- 大小：2,959,973,525 字节。
- SHA-256：`6b094ef9a69718f727ce8d7e15c4d961e51032cefaa952e0d6af9d176d7ba118`。
- 与官方发布元数据完全一致。
- gzip 完整性检查及 tar 文件清单读取通过。

原样解包到 `data/raw/techqa/TechQA/` 后，以下文件均存在、非空且可读：

| 文件 | 大小（字节） | 用途 |
| --- | ---: | --- |
| `training_and_dev/training_Q_A.json` | 1,149,612 | TRAIN 原始问答 |
| `training_and_dev/dev_Q_A.json` | 560,864 | DEV 原始问答 |
| `training_and_dev/training_Q_A.txt` | 314,092 | TRAIN 文本版本 |
| `training_and_dev/training_dev_technotes.json` | 291,957,707 | TRAIN/DEV 参考文档语料 |
| `training_and_dev/training_dev_technotes.sections.json` | 239,015,621 | TRAIN/DEV 参考文档分段语料 |
| `technote_corpus/full_technote_collection.sections.json` | 6,422,031,925 | 完整 Technote 分段语料 |
| `technote_corpus/full_technote_collection.txt.bz2` | 833,121,612 | 完整 Technote 压缩文本语料 |
| `technote_corpus/technotes.id_title_text` | 2,722,434,043 | Technote ID、标题和正文 |
| `technote_corpus/technotes.text` | 2,349,812,432 | Technote 正文语料 |
| `validation/validation_questions.json` | 34,987 | Validation 问题 |
| `validation/validation_reference.json` | 40,697 | Validation 参考数据 |
| `validation/validation_technotes.json` | 10,448,626 | Validation 文档语料 |
| `validation/validation_example_output.json` | 14,617 | Validation 示例输出 |
| `README.txt` | 5,997 | 官方数据说明 |
| `CDLA-Permissive-v1.0.pdf` | 92,209 | 数据许可 |
| `technote_corpus/CDLA-Permissive-v1.0.pdf` | 92,209 | 语料许可副本 |
| `evaluation.py` | 12,914 | 官方评测脚本 |

解包目录总占用约 15 GB（包含保留的原始归档）。本轮未读取记录条数、未分组、未筛选、未去重，也未核验问答与文档 ID 映射。
