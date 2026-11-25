# Enterprise RAG System v2

基于 TechQA 共享技术文档语料构建并系统评估的一套企业技术支持 RAG。项目重点不是堆叠组件，而是在统一评测口径下依次比较切分、Dense 检索、BM25、RRF、Reranker、生成上下文和 Prompt，最终选择效果、成本与可维护性更均衡的完整链路。

## 最终结论

最终配置为：

```text
TechQA 共享语料（28,482 篇文档）
  → fixed_600_0 切分
  → perplexity-ai/pplx-embed-v1-0.6b
  → 无查询指令、归一化向量点积
  → 每篇文档取最高分块，保留 PPLX 原始文档排序
  → Top-5 文档
  → 标题 + 代表块 + 同文档前后相邻块
  → Qwen/Qwen2.5-7B-Instruct
  → prompt_v2，temperature=0，max_tokens=512
```

最终链不使用 BM25、RRF 或 Reranker。原因不是这些方案没有被测试，而是复杂链在统一的生成评测中没有稳定超过 PPLX Dense 原排序，同时增加了索引、推理时延、显存占用和维护成本。

最终 Prompt：[`configs/generation/answer_with_citations_direct_conditions.prompt`](configs/generation/answer_with_citations_direct_conditions.prompt)

## 最终 DEV 结果

所有参数选择均在 TRAIN 完成；锁定配置后，仅在去除 13 条 TRAIN 重复项的 147 道可回答 DEV 查询上进行最终评测。

### 文档级检索

| 指标 | 命中数 | 比例 |
|---|---:|---:|
| Recall@1 | 81/147 | 55.10% |
| Recall@5 | 116/147 | 78.91% |
| Recall@10 | 125/147 | 85.03% |
| Recall@20 | 129/147 | 87.76% |
| Recall@50 | 134/147 | 91.16% |
| MRR@10 | — | 0.6497 |

147 道查询均保存了 50 篇不同文档及其分数和代表块 ID。完整结果见 [`experiments/dense/results/pplx_embed_v1_0.6b_fixed_600_0_dev/summary.json`](experiments/dense/results/pplx_embed_v1_0.6b_fixed_600_0_dev/summary.json)。

### 带引用回答生成

| 结果 | 数量 |
|---|---:|
| 完整通过 | 84/147 |
| 未通过 | 63/147 |
| 完整通过率 | **57.14%** |

“完整通过”要求核心答案正确、必要步骤和适用条件齐全、产品与版本匹配，而且关键结论得到实际 Prompt 中证据的直接支持。逐题 Codex 评审见 [`dev_review.jsonl`](experiments/generation/results/siliconflow_dev_pplx_top5_neighbors_prompt_v2_147/dev_review.jsonl)。

63 道失败回答中，38 道主要因为实际上下文证据不足，24 道属于证据已经进入 Prompt 但生成模型遗漏或误用，1 道无法确定。这说明最终瓶颈同时存在于 Top-5 片段覆盖和生成模型的证据使用能力。

## 数据与实验口径

| 数据 | 数量 | 用途 |
|---|---:|---|
| TechQA 共享 Technote 文档 | 28,482 | 全量知识库 |
| 可回答 TRAIN | 450 | 所有参数选择和消融实验 |
| `ANSWERABLE=N` TRAIN | 150 | 拒答诊断，不作为主要选型集 |
| 原始可回答 DEV | 160 | 最终评测候选 |
| 去重后可回答 DEV | 147 | 锁定配置后的最终评测 |

数据准备规则和输入/输出哈希保存在 [`data/processed/techqa/summary.json`](data/processed/techqa/summary.json)，来源核验记录见 [`reports/data_profile/techqa_data_verification.md`](reports/data_profile/techqa_data_verification.md)。参考文档 ID 只用于排名完成后的指标计算，不参与查询构造、检索或生成。

检索统一采用文档级评测。同一文档有多个块时，取最高相似度块作为该文档代表块；分数相同时按固定文档 ID 顺序排序。

生成评审除语义正确外，还检查：

- 必要操作步骤和前置条件；
- 产品、版本、环境与错误码；
- 结论是否受实际输入证据支持；
- 引用是否指向支持对应结论的文本。

因此“参考文档进入 Top-5”和“Top-5 实际块足以回答问题”被作为两个不同指标处理。

## 方案筛选过程

### 1. Dense 模型：BGE Small → PPLX

最初基线为 `BAAI/bge-small-en-v1.5`。在 `fixed_600_0` 上，其 R@5 为 290/450，MRR@10 为 0.5331。增加 BGE 查询指令只改善了 R@1，却使 R@5 从 64.44% 降到 63.11%，因此未采用。

切换为 `perplexity-ai/pplx-embed-v1-0.6b` 后，相同切分下 R@5 提升到 335/450，R@50 提升到 400/450。PPLX 因此前排召回明显更强，成为后续 Dense 实验的基础模型。

### 2. 六种切分策略

正式比较了四种固定切分和两种段落合并切分：

| 切分 | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_600_0` | 254 | **335** | **359** | 376 | 400 | 414 | 0.6396 |
| `fixed_600_150` | 256 | 332 | 357 | 370 | 394 | **417** | 0.6389 |
| `fixed_1200_0` | **258** | 330 | 354 | 372 | 400 | 414 | **0.6399** |
| `fixed_1200_300` | 252 | 330 | 350 | 375 | 399 | 414 | 0.6305 |
| `paragraph_merge_600_0` | 252 | 331 | 355 | **377** | **403** | **417** | 0.6324 |
| `paragraph_merge_1200_0` | 255 | 328 | 347 | 369 | 401 | 415 | 0.6347 |

`fixed_1200_0` 的 R@1 略高，段落合并方案在较深候选池中有局部优势，但 `fixed_600_0` 的 R@5 和 R@10 最好。重叠切分增加了块数和编码成本，却没有改善前排召回。其他切分没有立即淘汰，而是继续进入 BM25、RRF 和生成联合实验。

### 3. BM25 与 RRF

六种切分分别测试 `k1={0.8,1.2,1.6}` 与 `b={0.5,0.75,1.0}`，共 54 组 BM25。BM25 单路整体不如 PPLX，但能补回少量 Dense 漏检文档，因此保留为融合输入。

初始等权 RRF（PPLX Top-50、BM25 Top-50、权重 1:1、`k=60`）使 R@5 从 335 降到 317，说明稀疏与 Dense 结果不能直接等权合并。随后运行了完整 648 组网格：

```text
6 种切分
× 9 套 BM25 参数
× BM25 输入深度 50/100
× PPLX 输入深度 50/100
× PPLX:BM25 权重 1:1、2:1、3:1
```

每种切分保留一条候选 RRF 链，再加上 `fixed_600_0 PPLX` 原排序，组成七条联合候选。RRF 能改变候选覆盖，但没有在最终生成质量上证明稳定优势。

### 4. Reranker

所有重排实验复用相同 PPLX 候选与代表块文本。

| 排序 | R@1 | R@5 | R@10 | MRR@10 |
|---|---:|---:|---:|---:|
| PPLX 原排序 | **254** | **335** | **359** | **0.6396** |
| Qwen3-Reranker-0.6B | 238 | 306 | 350 | 0.6023 |
| Qwen3-Reranker-4B GGUF Q4_K_M | 193 | 316 | 353 | 0.5486 |
| Qwen3-Reranker-4B BF16 | 196 | 326 | 352 | 0.5575 |

`BAAI/bge-reranker-base` 的 Top-20 R@5 仅为 285/450，也未保留。4B BF16 是重排版本中损失最少的，但相对 PPLX 仍然新进 Top-5 22 题、掉出 31 题，净减少 9 题，而且推理成本最高。

### 5. 生成上下文与 Prompt

固定 seed=42 的 20 题消融结果：

| 配置 | 完整通过 |
|---|---:|
| 仅代表块，初始 Prompt | 5/20 |
| 代表块＋相邻块，初始 Prompt | 8/20 |
| 代表块＋相邻块，`prompt_v2` | **9/20** |
| 代表块＋相邻块，`prompt_v3` | 7/20 |

相邻块能补回被切分边界拆开的条件和步骤，因此进入最终链。窗口外再加入额外相似块会增加噪声，没有形成稳定收益。`prompt_v3` 和更严格的范围检查 Prompt 也没有稳定超过较简洁的 `prompt_v2`。

### 6. 七链与三条完整链联合实验

七链证据评审发现，文档 Recall 与实际证据完整性并不总是一致。例如 `paragraph_merge_1200_0 RRF` 在固定 30 题中的重排前完整证据达到 19/30，而 `fixed_600_0 PPLX` 的参考文档 Top-5 命中更高，但完整证据只有 11/30。

4B BF16 曾将 `fixed_1200_0 RRF` 的完整证据从 16/30 提升到 21/30，因此该组合进入了完整 TRAIN 联合评测。最终比较三条链：

| 组 | 链路 | 完整通过 |
|---|---|---:|
| A | `fixed_1200_0 RRF` 原排序 | 168/450，37.33% |
| B | `fixed_1200_0 RRF + 4B BF16` | 162/450，36.00% |
| C | `paragraph_merge_1200_0 RRF` 原排序 | 168/450，37.33% |

B 相对 A 新通过 31 题、退步 37 题；相对 C 新通过 32 题、退步 38 题。重排确实能救回一些题，但破坏的正确答案更多，因此没有进入最终链。该评审采用匿名生成模型初评，再由 Codex 复核策略间状态不同的题和固定抽样题；完整汇总见 [`final_review/summary.json`](experiments/generation/results/siliconflow_train450_three_chain_top5/final_review/summary.json)。

### 7. 复杂链回测简单基线

在同一批 30 题、相同生成模型、`prompt_v2` 和解码参数下：

| 完整链 | 完整通过 |
|---|---:|
| `fixed_600_0 PPLX Dense + 相邻块` | **17/30** |
| `paragraph_merge_1200_0 RRF`，仅代表块 | 15/30 |
| `paragraph_merge_1200_0 RRF + 相邻块` | **19/30** |

随后在历史固定 100 题上直接进行 Codex 成对评审：

| 完整链 | 完整通过 |
|---|---:|
| `fixed_600_0 PPLX Dense + 相邻块` | **54/100** |
| `paragraph_merge_1200_0 RRF + 相邻块` | 52/100 |

两组都通过 44 题，Dense 独过 10 题，段落 RRF 独过 8 题。复杂链没有证明稳定超过简单 Dense，因此最终回到 `fixed_600_0 + PPLX`。

## 拒答实验

TechQA 的 `ANSWERABLE=N` 表示标注者未在原候选文档中找到答案，不等于 28,482 篇共享语料中绝对不存在答案。

150 道此类 TRAIN 查询中，Top-5 审核发现 58 道证据充分、37 道部分或待核实，只有 55 道没有证据；继续检查 Top-100 后，又发现 9 道充分证据和 10 道部分证据。最终仅保留 36 道“PPLX Top-100 范围内无充分证据”的操作性无答案集。

Dense Top-1 分数在这些题与可回答题之间高度重叠，简单阈值不能可靠区分。严格拒答 Prompt 的无答案拒答率很高，但会大量误拒可回答问题；平衡版仍存在明显取舍。在固定 30 道可回答题上，拒答 Prompt 使完整通过从 17 降到 6。因此拒答作为诊断实验保留，最终可回答 DEV 不启用拒答。

详细无答案集定义和限制见 [`evidence_audit_top100/summary.json`](experiments/generation/results/siliconflow_train150_unanswerable_pplx_fixed_600_0_top5_neighbors_refusal/evidence_audit_top100/summary.json)。

## 运行环境

最终 DEV 检索记录的主要环境：

- Ubuntu / WSL；
- Python 3.12.3；
- PyTorch 2.15.0 + CUDA 13.2；
- Sentence Transformers 6.3.0；
- NVIDIA GeForce RTX 4060 Laptop GPU 8 GB；
- 本地 Hugging Face 模型缓存，PPLX 离线运行；
- 生成接口为 SiliconFlow Chat Completions，API Key 从 `SILICONFLOW_API_KEY` 读取。

模型权重、原始数据、向量缓存和生成结果体积较大，是否纳入版本控制由 `.gitignore` 和实际发布方式决定。

## 复现关键流程

以下命令均从仓库根目录运行。脚本默认复用已经存在且配置一致的切块、向量和排名产物。

### 1. 准备 TechQA 数据

```bash
./.venv/bin/python scripts/prepare_techqa.py
```

输出位于 `data/processed/techqa/`，包括 TRAIN、去重 DEV、共享语料、排除清单和哈希汇总。

### 2. 运行六种切分的 PPLX TRAIN 检索

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
./.venv/bin/python experiments/dense/run_pplx_dense_train_chunk_grid.py \
  --batch-size 16
```

结果位于 `experiments/dense/results/pplx_embed_v1_0.6b_six_chunk_train/`。

### 3. 运行 54 组 BM25

```bash
./.venv/bin/python experiments/chunk/run_bm25_parameter_grid.py
```

结果位于 `experiments/chunk/bm25_parameter_grid_train/`。

### 4. 运行离线 RRF 网格

```bash
./.venv/bin/python experiments/hybrid/run_rrf_offline.py \
  --batch \
  --output-dir experiments/hybrid/results/pplx_bm25_rrf_grid_train
```

RRF 直接读取已有 BM25 与 PPLX Top-100 排名，不重新编码或检索。

### 5. 运行最终 DEV 检索

需要已完成的 `fixed_600_0` PPLX 文档向量缓存：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
./.venv/bin/python experiments/dense/run_pplx_dense_dev.py
```

脚本复用 28,482 篇文档、182,835 个块的向量，只编码 147 条 DEV 查询。

### 6. 运行最终 DEV 生成

```bash
read -rsp 'SILICONFLOW_API_KEY: ' SILICONFLOW_API_KEY
echo
export SILICONFLOW_API_KEY
./.venv/bin/python scripts/run_dev_generation.py
unset SILICONFLOW_API_KEY
```

该入口支持按题号跳过已经成功生成的记录。回答、实际上下文、日志和退出码位于 `experiments/generation/results/siliconflow_dev_pplx_top5_neighbors_prompt_v2_147/`。

## 主要目录

```text
configs/generation/       生成 Prompt
data/raw/                 TechQA 原始文件
data/processed/techqa/    规范化查询、共享语料、切块和向量缓存
experiments/chunk/        Chunk 与 BM25 实验
experiments/dense/        Dense、查询指令和 Reranker 实验
experiments/hybrid/       RRF 融合与候选分析
experiments/generation/   上下文、Prompt、生成、评审和拒答实验
reports/                  数据核验与早期实验报告
scripts/                  数据准备、生成、评审和汇总入口
```

## 项目限制

- 固定 seed=42 的 20、30 和 100 题样本被多次用于消融，存在小样本选择偏差。
- 最终 `fixed_600_0 Dense` 的完整 TRAIN 直接 Codex 对照使用固定 100 题，而不是再次逐题复核全部 450 题；最终方案另由 147 道锁定 DEV 验证。
- 三链 450 题评审采用匿名模型初评加 Codex 复核，并非 Codex 逐份重新评审全部 1,350 份回答。
- 当前项目针对 TechQA 技术支持语料优化，不声称配置能直接泛化到其他领域。
- 项目没有进行 Embedding、Reranker 或生成模型微调。

## 核心经验

1. 更大的块和更多重叠不一定改善 Top-5 召回。
2. BM25 能补充 Dense，但简单等权融合会破坏前排排序。
3. Reranker 能救回部分问题，也可能让更多正确证据掉出 Top-5。
4. 参考文档命中不等于实际证据完整。
5. 检索指标或证据完整性局部提升，不一定转化为最终回答通过率提升。
6. 相邻块比单纯扩大代表块更稳定地补充步骤和适用条件。
7. Prompt 约束过强可能产生误拒或新的无依据操作。
8. 最终方案应同时考虑回答质量、推理成本、可复现性和维护复杂度。

更完整的原始实施规划见 [`PROJECT_PLAN.md`](PROJECT_PLAN.md)。
