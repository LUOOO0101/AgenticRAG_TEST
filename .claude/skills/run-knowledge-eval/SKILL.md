---
name: knowledge-eval-dm
description: 对飞书 Bitable 中的知识问答记录做 Kimi 与 DeepSeek 双模型并行质量评估，基于 retrieved_chunks/evidence 批量打分、分别写回字段、生成两份模型评估报告和一致性对比报告。当用户要求”Knowledge Eval DM””知识评估 DM””双模型评估””一致性对比””回复质量打分””RAG eval””knowledge eval”，并提供 Bitable 链接或 app_token/table_id 时使用。
---

# Knowledge Eval DM

批量评估知识库或 RAG 问答回复质量。用本地脚本执行评估，不要把评分规则整段搬进对话里手工打分。

## 何时使用

满足以下条件时直接使用本 skill：

- 用户要批量评估知识库或 RAG 问答回复质量
- 数据源是飞书 Bitable
- 用户给出了 Bitable 链接，或明确给出 `app_token` 和 `table_id`

## 前提条件

**必需环境变量：**

```powershell
$env:FEISHU_APP_ID = 'cli_xxx'
$env:FEISHU_APP_SECRET = 'xxx'
$env:MAAS_EVAL_API_KEY = 'xxx'
$env:ANTHROPIC_AUTH_TOKEN = 'xxx'  # 一致性分析需要
```

可选环境变量（代理、自定义配置）：

```powershell
$env:ANTHROPIC_BASE_URL = 'https://api.anthropic.com'  # 可选
$env:ANTHROPIC_MODEL = 'claude-opus-4-6'  # 可选，默认使用会话模型
$env:HTTPS_PROXY = 'http://proxy:port'  # 可选
```

**Bitable 表字段要求：**

目标表必须包含：`question`, `answer`, `scene`, `intent`, `retrieved_chunks`, `evidence`

## 快速开始（推荐使用驱动脚本）

从项目根目录执行，使用便捷驱动脚本 [.claude/skills/run-knowledge-eval/driver.py](.claude/skills/run-knowledge-eval/driver.py)：

**完整评估（双模型并行）：**

```powershell
python .claude/skills/run-knowledge-eval/driver.py “https://xxx.feishu.cn/base/<APP_TOKEN>?table=<TABLE_ID>” --verbose
```

**增量评估（只跑缺失记录）：**

```powershell
python .claude/skills/run-knowledge-eval/driver.py “https://xxx.feishu.cn/base/<APP_TOKEN>?table=<TABLE_ID>” --incremental --verbose
```

**只重新生成模型评估质量报告：**

```powershell
python .claude/skills/run-knowledge-eval/driver.py “https://xxx.feishu.cn/base/<APP_TOKEN>?table=<TABLE_ID>” --wiki-space-id 7xxx --report-only
```

**只重新生成一致性对比分析报告：**

```powershell
python .claude/skills/run-knowledge-eval/driver.py “https://xxx.feishu.cn/base/<APP_TOKEN>?table=<TABLE_ID>” --wiki-space-id 7xxx --consistency-only
```

驱动脚本会自动：
- 解析 Bitable 链接提取 `app_token` 和 `table_id`
- 检查必需的环境变量
- 构建正确的命令并执行
- 提供友好的错误提示

查看所有选项：

```powershell
python .claude/skills/run-knowledge-eval/driver.py --help
```

## 直接调用核心脚本（高级用法）

如果需要更精细的控制，可以直接调用 [scripts/knowledge_eval.py](../../scripts/knowledge_eval.py)：

**完整评估：**

```powershell
python scripts/knowledge_eval.py --app-token <APP_TOKEN> --table-id <TABLE_ID> --max-workers 4 --batch-size 20 --verbose
```

**增量评估：**

```powershell
python scripts/knowledge_eval.py --app-token <APP_TOKEN> --table-id <TABLE_ID> --only-missing --verbose
```

**只重新生成报告：**

```powershell
python scripts/knowledge_eval.py --app-token <APP_TOKEN> --table-id <TABLE_ID> --wiki-space-id <SPACE_ID> --report-only
```

**只重新生成一致性报告：**

```powershell
python scripts/knowledge_eval.py --app-token <APP_TOKEN> --table-id <TABLE_ID> --wiki-space-id <SPACE_ID> --consistency-only
```

## 评估流程

脚本按以下流程执行：

1. **自动建字段**：检查目标表是否包含 Kimi 与 DeepSeek 两套输出字段（`Kimi_<Dimension>_score/reason` 和 `DeepSeek_<Dimension>_score/reason`）。使用 `--skip-ensure-fields` 跳过。

2. **拉取记录**：默认整表全量评估；`--only-missing` 模式只跳过两个模型均已完成的记录。

3. **双模型并行评估**：Kimi (`xopkimik25`) 与 DeepSeek (`xopdsv4pth`) 同时进入线程池；并发单位是”模型 × 记录”。
   - 第一轮：基于 `retrieved_chunks` 评估 `Faithfulness` 和 `Traceability`
   - 第二轮：基于 `evidence` 评估 `Accuracy`, `Relevance`, `Reason`, `Completeness`, `Action`, `Clarity`, `Refuse`
   - 评分范围：0.0-5.0（保留 1 位小数），不涉及为 -1

4. **批量写回**：按模型分别写回 9 个分数字段和对应 reason 字段。

5. **生成报告**：
   - 两份模型评估质量报告：`知识回复评估报告_<Bitable名称>_Kimi` 和 `知识回复评估报告_<Bitable名称>_DeepSeek`
   - 一份一致性对比分析报告：`<Bitable名称>_一致性对比分析报告`（包含柱状图、对比表格、分歧样本的 Claude 辅助判断）

## 常用选项

| 选项 | 说明 |
|------|------|
| `--models kimi,deepseek` | 选择评估模型，默认两个都跑 |
| `--only-missing` | 增量模式，只跑缺少评分的记录 |
| `--skip-ensure-fields` | 跳过自动建字段 |
| `--report-only` | 只重新生成模型评估质量报告 |
| `--consistency-only` | 只重新生成一致性对比分析报告 |
| `--skip-consistency` | 跳过生成一致性报告 |
| `--reset-progress` | 清除断点续跑的进度文件 |
| `--max-workers N` | 设置并发数（默认 4） |
| `--batch-size N` | 批量写回大小（默认 20） |
| `--verbose` | 显示详细日志 |

## 评分标准文件

脚本会按需读取参考文件（位于 [references/](../../references/)）：

- [rubric.md](../../references/rubric.md) - 评估指标定义与评分标准
- [scene-rules.md](../../references/scene-rules.md) - 场景规则
- [intent-map.md](../../references/intent-map.md) - 意图映射
- [prompt-template.md](../../references/prompt-template.md) - Prompt 模板

只在需要修改评分标准或 prompt 模板时读取，不要一次性全部展开进上下文。

## 运行产物

脚本在 [runtime/](../../runtime/) 下生成临时文件：

- `progress.json` - 断点续跑进度
- `failed_records.json` - 失败记录
- `errors/` - 错误日志
- `charts/` - 柱状图图片（插入文档后自动删除）

## 故障排查

**环境变量缺失：**

```
✗ 错误: 缺少必需的环境变量: FEISHU_APP_ID, MAAS_EVAL_API_KEY
```

解决：设置所有必需的环境变量（见”前提条件”部分）。

**Bitable 链接解析失败：**

```
✗ 错误: 无法解析 Bitable 链接格式
```

解决：检查链接格式是否为 `https://xxx.feishu.cn/base/<app_token>?table=<table_id>`，或直接使用 `--app-token` 和 `--table-id` 参数。

**API 调用失败：**

检查网络连接和 API 密钥有效性。如果使用代理，确保设置了 `HTTPS_PROXY` 环境变量。

**断点续跑：**

脚本支持断点续跑。如果评估中断，重新运行相同命令会从上次中断处继续。使用 `--reset-progress` 强制从头开始。
