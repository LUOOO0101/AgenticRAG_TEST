---
name: knowledge-eval-dm
description: 对飞书 Bitable 中的知识问答记录做 Kimi 与 DeepSeek 双模型并行质量评估，基于 retrieved_chunks/evidence 批量打分、分别写回字段、生成两份模型评估报告和一致性对比报告。当用户要求“Knowledge Eval DM”“知识评估 DM”“双模型评估”“一致性对比”“回复质量打分”“RAG eval”“knowledge eval”，并提供 Bitable 链接或 app_token/table_id 时使用。
---

# Knowledge Eval DM

这个 skill 用本地脚本批量执行评估，不要把评分规则整段搬进对话里手工打分。

## 何时使用

满足以下条件时直接使用本 skill：

- 用户要批量评估知识库或 RAG 问答回复质量
- 数据源是飞书 Bitable
- 用户给出了 Bitable 链接，或明确给出 `app_token` 和 `table_id`

## 输入前提

目标表至少包含这些字段：

- `question`
- `answer`
- `scene`
- `intent`
- `retrieved_chunks`
- `evidence`

必需环境变量：

- `FEISHU_APP_ID`
- `FEISHU_APP_SECRET`
- `MAAS_EVAL_API_KEY`

一致性对比中的 Claude 辅助判断需要额外环境变量；不要把密钥写进脚本或 skill 文档：

- `ANTHROPIC_AUTH_TOKEN`
- `ANTHROPIC_BASE_URL`
- `ANTHROPIC_MODEL`（可选）
- 如需代理，由运行环境设置 `HTTPS_PROXY` / `HTTP_PROXY`

## 执行方式

先解析 Bitable 链接，拿到 `app_token` 和 `table_id`；如果用户已直接提供，则跳过解析。

在 skill 根目录执行。默认同时使用 Kimi `xopkimik25` 和 DeepSeek `xopdsv4pth`，并行评估同一批记录：

```bash
python3 scripts/knowledge_eval.py \
  --app-token <APP_TOKEN> \
  --table-id <TABLE_ID> \
  --max-workers 4 \
  --batch-size 20 \
  --verbose
```

只重新生成两份模型评估质量报告时执行：

```bash
python3 scripts/knowledge_eval.py \
  --app-token <APP_TOKEN> \
  --table-id <TABLE_ID> \
  --wiki-space-id <SPACE_ID> \
  --report-only
```

只重新生成一致性对比分析报告时执行：

```bash
python3 scripts/knowledge_eval.py \
  --app-token <APP_TOKEN> \
  --table-id <TABLE_ID> \
  --wiki-space-id <SPACE_ID> \
  --consistency-only
```

## 脚本行为

脚本会按以下流程执行：

1. **自动建字段**：检查目标表是否包含 Kimi 与 DeepSeek 两套输出字段，字段名为 `Kimi_<Dimension>_score/reason` 和 `DeepSeek_<Dimension>_score/reason`。用 `--skip-ensure-fields` 跳过。
2. **拉取记录**：默认整表全量评估；加 `--only-missing` 时只跳过两个模型均已写入 `Faithfulness_score` 前缀字段的记录。
3. **双模型并行评估**：Kimi `xopkimik25` 与 DeepSeek `xopdsv4pth` 同时进入线程池；并发单位是“模型 × 记录”。
4. 第一轮基于 `retrieved_chunks` 评估 `Faithfulness` 和 `Traceability`；第二轮基于 `evidence` 评估 `Accuracy`、`Relevance`、`Reason`、`Completeness`、`Action`、`Clarity`、`Refuse`。适用指标按 0.0-5.0 分制评分（保留 1 位小数），不涉及仍为 -1；评分标准不因模型变化而改变。
5. 按模型批量写回 9 个分数字段和对应 reason 字段。
6. 分别生成 `知识回复评估报告_<Bitable名称>_Kimi` 和 `知识回复评估报告_<Bitable名称>_DeepSeek` 两份飞书 Wiki / Docx 报告。
7. 生成 `<Bitable名称>_一致性对比分析报告` 报告：对比两个模型在各场景、各意图、各指标维度下的平均分数，两个模型都不涉及的指标（均显示为 `-`）可在总体表展示，结论显示为 `-`，但不纳入一致性统计和明细对比；其余指标中，两个模型得分相同或误差不超过当前评分分制满分的 10% 均视为一致；按各指标、各场景、各意图类型分别生成柱状图图片，每张柱状图下方放置对应的对比表格；一致性判断在本地完成，仅对超阈值分歧调用 Claude API 辅助判断哪个模型的分数与 reason 更合理，传给 Claude 的分歧样本包含具体问题、回答、当前指标对应的 evidence 或 retrieved_chunks，以及当前指标相关评分规则。

常用 flag：

- `--models kimi,deepseek`：选择评估模型，默认两个都跑
- `--only-missing`：增量模式，只跑缺少对应模型评分的任务
- `--skip-ensure-fields`：跳过自动建字段
- `--report-only`：只重生成两份模型评估质量报告，不生成一致性对比分析报告，也不重新评估
- `--consistency-only`：只重生成一致性对比分析报告，不重生成模型报告，也不重新评估
- `--skip-consistency`：跳过一致性对比报告
- `--reset-progress`：清掉断点续跑的进度文件

## 资源加载

脚本会按需读取这些参考文件：

- [references/rubric.md](./references/rubric.md)
- [references/scene-rules.md](./references/scene-rules.md)
- [references/intent-map.md](./references/intent-map.md)
- [references/prompt-template.md](./references/prompt-template.md)

只在需要修改评分标准或 prompt 模板时再读取对应文件，不要一次性全部展开进上下文。

## 运行产物

脚本运行时会在 `runtime/` 下生成临时文件：

- `progress.json`
- `failed_records.json`
- `errors/`

如果生成报告时创建柱状图图片，脚本会在图片插入飞书文档后删除本地临时图片。

这些都是运行态产物，不属于 skill 说明内容。
