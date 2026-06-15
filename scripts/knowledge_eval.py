#!/usr/bin/env python3
"""
Knowledge Eval DM - 知识库 QA 质量评估脚本
独立运行，不依赖 agent session。
飞书 REST API + MaaS Kimi/DeepSeek 双模型评估
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
import statistics
import uuid
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
SKILL_DIR = SCRIPT_DIR.parent
REFERENCES_DIR = SKILL_DIR / "references"
RUNTIME_DIR = SKILL_DIR / "runtime"
PROGRESS_FILE = RUNTIME_DIR / "progress.json"
FAILED_FILE = RUNTIME_DIR / "failed_records.json"
ERRORS_DIR = RUNTIME_DIR / "errors"
CHARTS_DIR = RUNTIME_DIR / "charts"

FEISHU_BASE = "https://open.feishu.cn/open-apis"
MAAS_ENDPOINT = "https://maas-api.cn-huabei-1.xf-yun.com/v2/chat/completions"
MODEL_CONFIGS = {
    "kimi": {
        "label": "Kimi",
        "model": "xopkimik25",
        "field_prefix": "Kimi_",
        "api_key_env": "MAAS_EVAL_API_KEY_KIMI",
    },
    "deepseek": {
        "label": "DeepSeek",
        "model": "xopdsv4pth",
        "field_prefix": "DeepSeek_",
        "api_key_env": "MAAS_EVAL_API_KEY_DEEPSEEK",
    },
}
DEFAULT_MODEL_KEYS = ["kimi", "deepseek"]
MAX_SCORE = 5.0
SCORE_AXIS_MIN = 0.0
SCORE_AXIS_MAX = MAX_SCORE
CONSISTENCY_TOLERANCE = round(MAX_SCORE * 0.10, 2)
PARTIAL_CAP_SCORE = 2.5
ACCURACY_FAITHFULNESS_THRESHOLD = 4.2
ACCURACY_FAITHFULNESS_CAP_SCORE = 3.5
HIGH_QUALITY_THRESHOLD = 4.5
LOW_QUALITY_THRESHOLD = 2.5
DEFAULT_WIKI_SPACE_ID = "7329710185914761220"
DOCX_WRITE_QPS = 3
DOCX_CHILDREN_BATCH_SIZE = 20
DOCX_BATCH_UPDATE_SIZE = 30
DOCX_MAX_NATIVE_TABLE_COLUMNS = 100
DOCX_INITIAL_TABLE_COLUMNS = 9
DOCX_RED_TEXT_COLOR = 1
# 当一组均分中的最小值都已经高于此阈值时，不再标红 —— 整体水平已经够好，不需要再点名"最低"
LOW_SCORE_HIGHLIGHT_THRESHOLD = 4.2

DIMENSIONS = [
    "Faithfulness", "Traceability",
    "Accuracy", "Relevance", "Reason",
    "Completeness", "Action", "Clarity", "Refuse"
]
ROUND1_DIMENSIONS = ["Faithfulness", "Traceability"]
ROUND2_DIMENSIONS = ["Accuracy", "Relevance", "Reason", "Completeness", "Action", "Clarity", "Refuse"]


# ─── HTTP helpers ───

def http_request(url, data=None, headers=None, method=None, timeout=30):
    """Generic HTTP request using urllib."""
    if headers is None:
        headers = {}
    if data is not None and isinstance(data, (dict, list)):
        data = json.dumps(data).encode("utf-8")
        headers.setdefault("Content-Type", "application/json; charset=utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    ctx = ssl._create_unverified_context()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout, context=ctx)
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body) if body else {}
        except json.JSONDecodeError:
            parsed = {"raw_body": body}
        raise RuntimeError(f"HTTP {e.code}: {parsed}") from e


def retry_request(url, data=None, headers=None, method=None, timeout=30,
                  retries=3, label="request"):
    """HTTP request with exponential backoff retry."""
    last_err = None
    for attempt in range(retries):
        try:
            return http_request(url, data, headers, method, timeout)
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"  ⚠ {label} 第{attempt+1}次失败: {e}, {wait}s后重试...")
            time.sleep(wait)
    raise last_err


# ─── Feishu API ───

class TokenManager:
    """Manages feishu tenant_access_token with auto-refresh."""
    def __init__(self, app_id, app_secret):
        self.app_id = app_id
        self.app_secret = app_secret
        self.token = None
        self.expire_at = 0

    def get(self):
        """Get valid token, refresh if expired or close to expiry."""
        now = time.time()
        if self.token and now < self.expire_at - 300:  # 5 min buffer
            return self.token
        resp = retry_request(
            f"{FEISHU_BASE}/auth/v3/tenant_access_token/internal",
            data={"app_id": self.app_id, "app_secret": self.app_secret},
            label="tenant_token"
        )
        if resp.get("code") != 0:
            raise RuntimeError(f"获取 tenant_token 失败: {resp}")
        self.token = resp["tenant_access_token"]
        self.expire_at = now + resp.get("expire", 7200)
        return self.token


def feishu_headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8"
    }


def eval_field_name(dim, kind, model_config=None):
    prefix = (model_config or {}).get("field_prefix", "")
    return f"{prefix}{dim}_{kind}"


def fetch_records(token, app_token, table_id, verbose=False, only_missing=False, model_configs=None):
    """Fetch records. With only_missing=True, skip rows scored by all selected models."""
    all_records = []
    page_token = None
    page = 0

    while True:
        page += 1
        params = "page_size=100"
        if page_token:
            params += f"&page_token={urllib.request.quote(page_token)}"

        url = f"{FEISHU_BASE}/bitable/v1/apps/{app_token}/tables/{table_id}/records?{params}"

        resp = retry_request(url, headers=feishu_headers(token), label=f"fetch_page_{page}")

        if resp.get("code") != 0:
            raise RuntimeError(f"拉取记录失败: {resp.get('msg', resp)}")

        items = resp.get("data", {}).get("items") or []

        for item in items:
            if only_missing:
                fields = item.get("fields", {})
                configs = model_configs or [None]
                if all(fields.get(eval_field_name("Faithfulness", "score", cfg)) is not None for cfg in configs):
                    continue
            all_records.append(item)

        if verbose:
            label = "未评估" if only_missing else "累计"
            print(f"  页 {page}: 获取 {len(items)} 条, {label} {len(all_records)} 条")

        if not resp.get("data", {}).get("has_more"):
            break
        page_token = resp["data"].get("page_token")

    return all_records


def list_table_fields(token, app_token, table_id):
    """Return {field_name: field_dict} for all fields in the table."""
    fields = {}
    page_token = None
    while True:
        params = "page_size=100"
        if page_token:
            params += f"&page_token={urllib.request.quote(page_token)}"
        url = f"{FEISHU_BASE}/bitable/v1/apps/{app_token}/tables/{table_id}/fields?{params}"
        resp = retry_request(url, headers=feishu_headers(token), label="list_fields")
        if resp.get("code") != 0:
            raise RuntimeError(f"获取字段失败: {resp.get('msg', resp)}")
        for item in resp.get("data", {}).get("items", []):
            fields[item.get("field_name")] = item
        if not resp.get("data", {}).get("has_more"):
            break
        page_token = resp["data"].get("page_token")
    return fields


def create_table_field(token, app_token, table_id, field_name, field_type, property_=None):
    """Create a single field. field_type: 1=Text, 2=Number."""
    url = f"{FEISHU_BASE}/bitable/v1/apps/{app_token}/tables/{table_id}/fields"
    body = {"field_name": field_name, "type": field_type}
    if property_:
        body["property"] = property_
    resp = retry_request(url, data=body, headers=feishu_headers(token), label=f"create_field_{field_name}")
    if resp.get("code") != 0:
        raise RuntimeError(f"创建字段 {field_name} 失败: {resp.get('msg', resp)}")
    return resp.get("data", {}).get("field", {})


def ensure_eval_fields(token, app_token, table_id, verbose=False, model_configs=None):
    """Ensure eval output fields exist for every selected model."""
    existing = list_table_fields(token, app_token, table_id)
    created = 0
    def _create_or_refresh(field_name, field_type, property_=None):
        nonlocal existing, created
        try:
            create_table_field(token, app_token, table_id, field_name, field_type, property_)
            existing[field_name] = {"field_name": field_name}
            created += 1
            return True
        except RuntimeError as e:
            if "FieldNameDuplicated" not in str(e):
                raise
            existing = list_table_fields(token, app_token, table_id)
            return False

    for model_config in (model_configs or [None]):
        for dim in DIMENSIONS:
            score_name = eval_field_name(dim, "score", model_config)
            reason_name = eval_field_name(dim, "reason", model_config)
            if score_name not in existing:
                created_now = _create_or_refresh(score_name, 2, {"formatter": "0.0"})
                if verbose and created_now:
                    print(f"  ✓ 创建字段 {score_name} (Number)")
            if reason_name not in existing:
                created_now = _create_or_refresh(reason_name, 1)
                if verbose and created_now:
                    print(f"  ✓ 创建字段 {reason_name} (Text)")
    return created


def fetch_all_records(token, app_token, table_id, verbose=False):
    """Fetch ALL records (including scored ones) for report generation."""
    all_records = []
    page_token = None
    page = 0

    while True:
        page += 1
        params = "page_size=100"
        if page_token:
            params += f"&page_token={urllib.request.quote(page_token)}"

        url = f"{FEISHU_BASE}/bitable/v1/apps/{app_token}/tables/{table_id}/records?{params}"

        resp = retry_request(url, headers=feishu_headers(token), label=f"fetch_all_page_{page}")

        if resp.get("code") != 0:
            raise RuntimeError(f"拉取记录失败: {resp.get('msg', resp)}")

        items = resp.get("data", {}).get("items") or []
        all_records.extend(items)

        if verbose:
            print(f"  页 {page}: 获取 {len(items)} 条, 累计 {len(all_records)} 条")

        if not resp.get("data", {}).get("has_more"):
            break
        page_token = resp["data"].get("page_token")

    return all_records


def batch_update_records(token, app_token, table_id, records_data, verbose=False):
    """Batch update records in feishu bitable."""
    url = f"{FEISHU_BASE}/bitable/v1/apps/{app_token}/tables/{table_id}/records/batch_update"
    body = {"records": records_data}
    resp = retry_request(
        url, data=body, headers=feishu_headers(token),
        label="batch_update", timeout=30
    )
    if resp.get("code") != 0:
        raise RuntimeError(f"写回失败: {resp.get('msg', resp)}")
    return resp


# ─── MaaS API ───

def call_maas(api_key, prompt, model_config=None, verbose=False):
    """Call MaaS model via OpenAI-compatible API."""
    model_config = model_config or MODEL_CONFIGS["kimi"]
    api_key = model_config.get("api_key") or api_key
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json; charset=utf-8"
    }
    data = {
        "model": model_config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 8192
    }
    resp = retry_request(
        MAAS_ENDPOINT, data=data, headers=headers,
        timeout=120, retries=3, label=f"maas_eval_{model_config['label']}"
    )
    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    if verbose:
        usage = resp.get("usage", {})
        print(f"  {model_config['label']} tokens: in={usage.get('prompt_tokens',0)} out={usage.get('completion_tokens',0)}")
    return content


# ─── Prompt construction ───

def extract_field_text(field_val):
    """Extract text from feishu bitable field value."""
    if field_val is None:
        return ""
    if isinstance(field_val, str):
        return field_val
    if isinstance(field_val, (int, float)):
        return str(field_val)
    if isinstance(field_val, list):
        parts = []
        for item in field_val:
            if isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(field_val)



_FULLWIDTH_TO_ASCII = {
    "，": ",",
    "：": ":",
    "；": ";",
    "【": "[",
    "】": "]",
    "（": "(",
    "）": ")",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
}


def _normalize_jsonish(s):
    """Normalize loose JSON written with Chinese full-width punctuation.

    Only replaces punctuation OUTSIDE of double-quoted string literals so that
    legitimate Chinese punctuation inside content fields is preserved.
    """
    out = []
    in_str = False
    escape = False
    for ch in s:
        if in_str:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
                out.append(ch)
            else:
                out.append(_FULLWIDTH_TO_ASCII.get(ch, ch))
    return "".join(out)


def parse_feishu_json_field(field_val):
    """Parse a feishu bitable field that contains JSON data.
    
    Data should be standard JSON array: '[{...}, {...}]'
    Also handles rich text (list of {type:"text", text:"..."} segments).
    """
    if not field_val:
        return []
    
    if isinstance(field_val, str):
        field_val = field_val.strip()
        for candidate in (field_val, _normalize_jsonish(field_val)):
            try:
                result = json.loads(candidate)
                if isinstance(result, list):
                    return result
                elif isinstance(result, dict):
                    return [result]
            except (json.JSONDecodeError, TypeError):
                continue
        return []
    
    if not isinstance(field_val, list):
        return []
    
    # Rich text: extract text parts and recurse
    text_parts = []
    for item in field_val:
        if isinstance(item, dict):
            text = item.get("text", "")
            if text:
                text_parts.append(text)
    
    if not text_parts:
        return []
    
    full = "".join(text_parts)
    return parse_feishu_json_field(full)


def load_skill_files():
    """Load evaluation criteria files."""
    files = {}
    for name in ["rubric.md", "scene-rules.md", "intent-map.md", "prompt-template.md"]:
        path = REFERENCES_DIR / name
        if path.exists():
            files[name] = path.read_text(encoding="utf-8")
        else:
            print(f"  ⚠ 缺失评估准则文件: {path}")
            files[name] = ""
    return files


def _extract_prompt_template_block(prompt_template_text, index):
    """Extract prompt blocks from prompt-template.md using section headings."""
    headings = [
        "## 第一轮 Prompt：检索忠实性评估",
        "## 第二轮 Prompt：回答质量评估",
    ]
    if not (0 <= index < len(headings)):
        return ""

    start = prompt_template_text.find(headings[index])
    if start == -1:
        return ""

    end = len(prompt_template_text)
    if index + 1 < len(headings):
        next_start = prompt_template_text.find(headings[index + 1], start + len(headings[index]))
        if next_start != -1:
            end = next_start

    section = prompt_template_text[start:end]
    match = re.search(r"^```(?:[a-zA-Z0-9_-]+)?\s*$\n(.*?)\n^```\s*$", section, flags=re.S | re.M)
    if match:
        return match.group(1).strip()
    return ""


def _extract_round2_rubric(rubric_text):
    marker = "## 第二轮：回答质量指标"
    idx = rubric_text.find(marker)
    if idx != -1:
        return rubric_text[idx:].strip()
    return rubric_text.strip()


def _extract_dimension_rubric(rubric_text, dim):
    """Extract the rubric section for one evaluation dimension."""
    match = re.search(
        rf"^###\s+\d+\.\s+{re.escape(dim)}（.*?）\s*$\n(.*?)(?=^###\s+\d+\.|^##\s+|\Z)",
        rubric_text,
        flags=re.S | re.M,
    )
    return match.group(0).strip() if match else ""


def _extract_round2_common_rules(rubric_text):
    """Extract the shared second-round principles before the first dimension."""
    match = re.search(
        r"^## 第二轮：回答质量指标\s*$\n(.*?)(?=^###\s+\d+\.)",
        rubric_text,
        flags=re.S | re.M,
    )
    return match.group(0).strip() if match else ""


def _extract_intent_rule(intent_map_text, intent):
    """Extract the mapping row and shared usage rules for the current intent."""
    target_row = ""
    for line in intent_map_text.splitlines():
        if line.startswith("|") and intent and intent in line:
            target_row = line.strip()
            break
    usage_idx = intent_map_text.find("## 使用规则")
    usage_rules = intent_map_text[usage_idx:].strip() if usage_idx != -1 else ""
    return "\n".join(part for part in [target_row, usage_rules] if part)


def _fill_prompt_template(template_text, replacements):
    prompt = template_text
    for key, value in replacements.items():
        prompt = prompt.replace(f"{{{key}}}", value)
    return prompt


def get_scene_rules(scene_rules_full, scene):
    """Extract scene-specific rules from full scene-rules doc."""
    if not scene or not scene_rules_full:
        return scene_rules_full
    target = scene.strip()
    sections = {}
    current = None
    for line in scene_rules_full.split("\n"):
        if line.startswith("## ") and not line.startswith("### "):
            current = line[3:].strip()
            sections[current] = [line]
            continue
        if current is not None:
            sections[current].append(line)

    result = []
    seen = set()
    for name in ("通用场景", "通用", target):
        section = sections.get(name)
        if section:
            text = "\n".join(section)
            if text not in seen:
                result.append(text)
                seen.add(text)
    return "\n\n".join(result) if result else scene_rules_full


def build_record_payload(record, include_chunks=False, include_evidence=False):
    """Build normalized payload for a single record."""
    fields = record.get("fields", {})
    payload = {
        "record_id": record.get("record_id", ""),
        "question": extract_field_text(fields.get("question")),
        "answer": extract_field_text(fields.get("answer")),
        "scene": extract_field_text(fields.get("scene", "通用")) or "通用",
        "intent": extract_field_text(fields.get("intent", "")),
    }
    if include_chunks:
        payload["retrieved_chunks"] = extract_field_text(fields.get("retrieved_chunks"))
    if include_evidence:
        payload["evidence"] = extract_field_text(fields.get("evidence"))
    return payload


def build_round1_prompt(record, skill_files):
    """Build round 1 prompt for a single record."""
    scene = extract_field_text(record.get("fields", {}).get("scene", "通用")) or "通用"
    scene_rules = get_scene_rules(skill_files.get("scene-rules.md", ""), scene)
    record_json = json.dumps(
        [build_record_payload(record, include_chunks=True)],
        ensure_ascii=False,
        indent=2,
    )
    prompt_template = _extract_prompt_template_block(skill_files.get("prompt-template.md", ""), 0)
    if prompt_template:
        return _fill_prompt_template(prompt_template, {
            "SCENE_RULES_ROUND1": scene_rules,
            "RECORDS_JSON": record_json,
        })
    return f"""你是知识回复质量评估专家。现在只评估单条记录的第一轮：检索忠实性。

## 本轮只评估两个指标

1. Faithfulness：基于 retrieved_chunks 判断 answer 是否忠实于实际召回内容，不得新增无依据事实
2. Traceability：判断 answer 中事实句末标注的来源编号[^n]，是否真实对应到 retrieved_chunks 的对应片段 id，并支撑该句断言

注意：评估时只比对来源标注中的 n 与 retrieved_chunks 中的 id 是否一致，不评估脚注写法本身；只要 n 与 id 对应，即视为来源编号可对应。

## 评分依据

{skill_files.get("rubric.md", "")}

## 当前场景规则

{scene_rules}

## 输入记录

{record_json}

## 输出格式

严格输出单个 JSON 对象，不要输出数组，不要 markdown 代码块：

{{
  "record_id": "{record.get("record_id", "")}",
  "scores": {{
    "Faithfulness": {{"score": 4.8, "reason": "..."}},
    "Traceability": {{"score": -1, "reason": "不涉及"}}
  }}
}}

注意：
- 只输出 Faithfulness 和 Traceability
- score 只能是 0.0 到 5.0（保留 1 位小数）或 -1（不涉及）
- reason 必须是一句话
"""


def build_round2_prompt(record, skill_files):
    """Build round 2 prompt for a single record."""
    scene = extract_field_text(record.get("fields", {}).get("scene", "通用")) or "通用"
    scene_rules = get_scene_rules(skill_files.get("scene-rules.md", ""), scene)
    record_json = json.dumps(
        [build_record_payload(record, include_evidence=True)],
        ensure_ascii=False,
        indent=2,
    )
    prompt_template = _extract_prompt_template_block(skill_files.get("prompt-template.md", ""), 1)
    if prompt_template:
        return _fill_prompt_template(prompt_template, {
            "RUBRIC_ROUND2": _extract_round2_rubric(skill_files.get("rubric.md", "")),
            "SCENE": scene,
            "SCENE_RULES_ROUND2": scene_rules,
            "INTENT_MAP_CONTENT": skill_files.get("intent-map.md", ""),
            "RECORDS_JSON": record_json,
        })
    return f"""你是知识回复质量评估专家。现在只评估单条记录的第二轮：回答质量。

## 本轮评估指标

- Accuracy
- Relevance
- Reason
- Completeness
- Action
- Clarity
- Refuse

## 评分依据

{skill_files.get("rubric.md", "")}

## 当前场景规则

{scene_rules}

## 意图映射

{skill_files.get("intent-map.md", "")}

## 评估要求

1. Accuracy 严格基于 evidence 判断
2. 不涉及的指标填 score=-1, reason="不涉及"
3. 只输出本轮 7 个指标

## 输入记录

{record_json}

## 输出格式

严格输出单个 JSON 对象，不要输出数组，不要 markdown 代码块：

{{
  "record_id": "{record.get("record_id", "")}",
  "scores": {{
    "Accuracy": {{"score": 4.8, "reason": "..."}},
    "Relevance": {{"score": 5.0, "reason": "..."}},
    "Reason": {{"score": -1, "reason": "不涉及"}},
    "Completeness": {{"score": 4.2, "reason": "..."}},
    "Action": {{"score": -1, "reason": "不涉及"}},
    "Clarity": {{"score": 3.5, "reason": "..."}},
    "Refuse": {{"score": -1, "reason": "不涉及"}}
  }}
}}

注意：
- score 只能是 0.0 到 5.0（保留 1 位小数）或 -1（不涉及）
- reason 必须是一句话
"""


# ─── Progress management ───

def load_progress():
    """Load completed record IDs from progress file."""
    if PROGRESS_FILE.exists():
        try:
            data = json.loads(PROGRESS_FILE.read_text())
            return set(data.get("completed_ids", []))
        except:
            return set()
    return set()


def save_progress(completed_ids):
    """Save completed record IDs."""
    RUNTIME_DIR.mkdir(exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps({
        "completed_ids": list(completed_ids),
        "updated_at": datetime.now().isoformat()
    }, ensure_ascii=False, indent=2))


def clear_progress():
    """Remove progress file after a fully completed run."""
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        print("✓ 已清空进度文件")


def save_error(label, raw_response):
    """Save raw API response on parse failure."""
    ERRORS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(label))
    path = ERRORS_DIR / f"{ts}_{safe_label}.txt"
    path.write_text(raw_response, encoding="utf-8")
    print(f"  原始响应已保存: {path}")


def save_failed(failed_list):
    """Save failed records."""
    RUNTIME_DIR.mkdir(exist_ok=True)
    FAILED_FILE.write_text(json.dumps(failed_list, ensure_ascii=False, indent=2))


# ─── JSON parsing ───

def parse_eval_response(raw_text):
    """Parse model response as JSON payload, with fixup attempts."""
    text = raw_text.strip()

    # Remove markdown code block if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        elif lines[0].startswith("```"):
            lines = lines[1:]
        text = "\n".join(lines).strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object/array in text
    candidates = []
    array_start = text.find("[")
    array_end = text.rfind("]")
    if array_start != -1 and array_end != -1 and array_end > array_start:
        candidates.append(text[array_start:array_end + 1])
    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        candidates.append(text[obj_start:obj_end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # Try fixing truncated JSON by adding closing brackets
    for suffix in ["]", "}", "}]", "\"}}]", "\"}]}]"]:
        try:
            return json.loads(text + suffix)
        except:
            continue

    raise ValueError(f"无法解析 JSON: {text[:200]}...")


def parse_score_value(raw):
    """Parse score as -1 or a one-decimal 0.0-5.0 value."""
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value == -1:
        return -1
    if 0 <= value <= MAX_SCORE:
        return round(value, 1)
    return None


def is_valid_score_value(raw):
    value = parse_score_value(raw)
    if value is None:
        return False
    if value == -1:
        return True
    return 0 <= value <= MAX_SCORE


def validate_scores(result_list, expected_ids, expected_dims=None):
    """Validate parsed scores structure."""
    expected_dims = expected_dims or DIMENSIONS
    valid = []
    expected_id_set = set(expected_ids)
    for item in result_list:
        rid = item.get("record_id")
        scores = item.get("scores", {})
        if not rid or not scores or rid not in expected_id_set:
            continue
        # Check all dimensions exist
        all_ok = True
        for dim in expected_dims:
            if dim not in scores:
                all_ok = False
                break
            s = scores[dim]
            if "score" not in s or "reason" not in s:
                all_ok = False
                break
            parsed_score = parse_score_value(s["score"])
            if parsed_score is None:
                all_ok = False
                break
            s["score"] = parsed_score
        if all_ok:
            valid.append(item)
    return valid


def to_result_list(parsed_payload):
    """Normalize parsed payload to list form."""
    if isinstance(parsed_payload, list):
        return parsed_payload
    if isinstance(parsed_payload, dict):
        return [parsed_payload]
    raise ValueError(f"返回格式错误: {type(parsed_payload)}")


def build_failed_scores(dimensions, reason):
    """Build score map for a failed round."""
    return {dim: {"score": -1, "reason": reason} for dim in dimensions}


def apply_consistency_rules(scores):
    """Apply hard consistency rules after model scoring."""
    faith = scores.get("Faithfulness", {}).get("score")
    acc = scores.get("Accuracy", {}).get("score")
    rel = scores.get("Relevance", {}).get("score")
    trace = scores.get("Traceability", {}).get("score")

    if (
        faith is not None
        and faith < ACCURACY_FAITHFULNESS_THRESHOLD
        and acc not in (-1, None)
        and acc > ACCURACY_FAITHFULNESS_CAP_SCORE
    ):
        scores["Accuracy"] = {
            "score": ACCURACY_FAITHFULNESS_CAP_SCORE,
            "reason": "Faithfulness未达到4.2分，Accuracy最高3.5分"
        }

    if faith == 0 and trace == MAX_SCORE:
        scores["Traceability"] = {
            "score": PARTIAL_CAP_SCORE,
            "reason": "Faithfulness为0分时，Traceability不能为5.0分，下调为2.5分"
        }

    if faith == 0 and acc == 0 and rel not in (-1, None) and rel > PARTIAL_CAP_SCORE:
        scores["Relevance"] = {
            "score": PARTIAL_CAP_SCORE,
            "reason": "检索失败且事实错误，相关性最高2.5分"
        }
    return scores


def evaluate_round(record, round_name, prompt, expected_dims, maas_key, model_config, verbose=False):
    """Evaluate a single round with one retry on parsing/validation errors."""
    record_id = record.get("record_id", "")
    last_error = None

    for attempt in range(2):
        raw_response = None
        try:
            raw_response = call_maas(maas_key, prompt, model_config, verbose)
            parsed = parse_eval_response(raw_response)
            result_list = to_result_list(parsed)
            valid = validate_scores(result_list, [record_id], expected_dims)
            if not valid:
                raise ValueError("返回结果缺少必需指标或 record_id 不匹配")
            return valid[0]["scores"]
        except Exception as e:
            last_error = e
            if raw_response:
                save_error(f"{model_config['label']}_{record_id}_{round_name}_attempt{attempt + 1}", raw_response)
            if attempt == 0:
                time.sleep(1)

    return build_failed_scores(expected_dims, f"评估失败: {last_error}")


def evaluate_record(record, skill_files, maas_key, model_config, verbose=False):
    """Evaluate one record in two rounds and return full 9-dim result."""
    round1_prompt = build_round1_prompt(record, skill_files)
    round1_scores = evaluate_round(
        record,
        "round1",
        round1_prompt,
        ROUND1_DIMENSIONS,
        maas_key,
        model_config,
        verbose,
    )

    round2_prompt = build_round2_prompt(record, skill_files)
    round2_scores = evaluate_round(
        record,
        "round2",
        round2_prompt,
        ROUND2_DIMENSIONS,
        maas_key,
        model_config,
        verbose,
    )

    scores = {}
    scores.update(round1_scores)
    scores.update(round2_scores)
    apply_consistency_rules(scores)
    return {
        "record_id": record.get("record_id", ""),
        "model_key": model_config.get("key", model_config["label"].lower()),
        "model_label": model_config["label"],
        "scores": scores,
    }


# ─── Report generation ───

def analyze_scores(records):
    """Analyze scored records and return comprehensive statistics."""
    
    def _strip_punctuation(s):
        """Strip punctuation and whitespace, keep Chinese + English + digits."""
        return re.sub(r'[^\u4e00-\u9fffa-zA-Z0-9]', '', s)

    def _fuzzy_match(content, chunk_text):
        """Check if evidence content was retrieved in chunk text.
        
        Strategy: strip all punctuation/whitespace, keep Chinese + English + digits.
        1. Exact substring match after stripping.
        2. 10-char sliding window with 40% coverage threshold for partial matches.
        """
        if not content or not chunk_text:
            return False
        
        ev = _strip_punctuation(content)
        ct = _strip_punctuation(chunk_text)
        
        if len(ev) < 5:
            return False
        
        # 1. Exact substring
        if ev in ct:
            return True
        
        # 2. 10-char sliding window, 40% coverage
        seg_len = min(10, len(ev))
        if seg_len < 5:
            return False
        step = 5
        match_count = 0
        total_segs = 0
        for start in range(0, len(ev) - seg_len + 1, step):
            seg = ev[start:start + seg_len]
            total_segs += 1
            if seg in ct:
                match_count += 1
        if total_segs > 0 and match_count / total_segs >= 0.4:
            return True
        
        return False
    
    # Filter records with scores
    scored = []
    for rec in records:
        fields = rec.get("fields", {})
        if fields.get("Faithfulness_score") is not None:
            scored.append(rec)
    
    if not scored:
        return None
    
    analysis = {
        "total_records": len(scored),
        "dimensions": {},
        "intents": {},
        "zero_score_records": [],
        "retrieval_quality": {}
    }
    
    # Dimension-level analysis
    for dim in DIMENSIONS:
        scores = []
        count_5 = 0
        count_4 = 0
        count_3 = 0
        count_lt3 = 0
        count_0 = 0
        valid_count = 0
        
        for rec in scored:
            fields = rec.get("fields", {})
            score_raw = fields.get(f"{dim}_score")
            score = parse_score_value(score_raw)
            
            if score is not None and score != -1:
                scores.append(score)
                valid_count += 1
                if score == MAX_SCORE:
                    count_5 += 1
                if score >= 4.0:
                    count_4 += 1
                elif score >= 3.0:
                    count_3 += 1
                else:
                    count_lt3 += 1
                if score == 0:
                    count_0 += 1
        
        avg = statistics.mean(scores) if scores else 0
        analysis["dimensions"][dim] = {
            "avg": avg,
            "count_5": count_5,
            "count_4": count_4,
            "count_3": count_3,
            "count_lt3": count_lt3,
            "count_0": count_0,
            "valid_count": valid_count,
            "na_count": len(scored) - valid_count
        }
    
    # Overall metrics
    overall_scores = []
    high_quality = 0
    perfect = 0
    low_quality = 0
    
    for rec in scored:
        fields = rec.get("fields", {})
        rec_scores = []
        all_perfect = True
        
        for dim in DIMENSIONS:
            score_raw = fields.get(f"{dim}_score")
            score = parse_score_value(score_raw)
            
            if score is not None and score != -1:
                rec_scores.append(score)
                if score != MAX_SCORE:
                    all_perfect = False
        
        if rec_scores:
            avg = statistics.mean(rec_scores)
            overall_scores.append(avg)
            if avg >= HIGH_QUALITY_THRESHOLD:
                high_quality += 1
            if avg < LOW_QUALITY_THRESHOLD:
                low_quality += 1
            if all_perfect:
                perfect += 1
    
    analysis["overall_avg"] = statistics.mean(overall_scores) if overall_scores else 0
    analysis["high_quality_pct"] = (high_quality / len(scored) * 100) if scored else 0
    analysis["perfect_pct"] = (perfect / len(scored) * 100) if scored else 0
    analysis["low_quality_pct"] = (low_quality / len(scored) * 100) if scored else 0
    
    # Intent-level analysis
    intent_stats = {}
    for rec in scored:
        fields = rec.get("fields", {})
        intent = extract_field_text(fields.get("intent", "未知"))
        
        if intent not in intent_stats:
            intent_stats[intent] = {
                "count": 0,
                "dim_scores": {dim: [] for dim in DIMENSIONS},
                "zero_counts": {dim: 0 for dim in DIMENSIONS},
                "zero_record_ids": set(),
            }
        
        intent_stats[intent]["count"] += 1
        
        for dim in DIMENSIONS:
            score_raw = fields.get(f"{dim}_score")
            score = parse_score_value(score_raw)
            
            if score is not None and score != -1:
                intent_stats[intent]["dim_scores"][dim].append(score)
                if score == 0:
                    intent_stats[intent]["zero_counts"][dim] += 1
                    intent_stats[intent]["zero_record_ids"].add(rec.get("record_id"))
    
    # Calculate intent averages
    for intent, stats in intent_stats.items():
        stats["dim_avg"] = {}
        stats["overall_avg"] = []
        
        for dim in DIMENSIONS:
            scores = stats["dim_scores"][dim]
            stats["dim_avg"][dim] = statistics.mean(scores) if scores else -1
            if scores:
                stats["overall_avg"].extend(scores)
        
        stats["overall_avg"] = statistics.mean(stats["overall_avg"]) if stats["overall_avg"] else 0
        stats["total_zeros"] = len(stats["zero_record_ids"])
        del stats["zero_record_ids"]
    
    analysis["intents"] = intent_stats
    
    # Zero score diagnosis
    for rec in scored:
        fields = rec.get("fields", {})
        zero_dims = []
        
        for dim in DIMENSIONS:
            score_raw = fields.get(f"{dim}_score")
            score = parse_score_value(score_raw)
            
            if score == 0:
                zero_dims.append(dim)
        
        if zero_dims:
            question = extract_field_text(fields.get("question", ""))
            answer = extract_field_text(fields.get("answer", ""))
            intent = extract_field_text(fields.get("intent", "未知"))
            
            # Get scores for attribution analysis
            faith_score = 0
            trace_score = 0
            acc_score = 0
            comp_score = 0
            for _dim, _field in [("Faithfulness", "Faithfulness_score"),
                                  ("Traceability", "Traceability_score"),
                                  ("Accuracy", "Accuracy_score"),
                                  ("Completeness", "Completeness_score")]:
                try:
                    _val = parse_score_value(fields.get(_field))
                    if _val is None:
                        _val = -1
                except (ValueError, TypeError):
                    _val = -1
                if _dim == "Faithfulness": faith_score = _val
                elif _dim == "Traceability": trace_score = _val
                elif _dim == "Accuracy": acc_score = _val
                elif _dim == "Completeness": comp_score = _val
            
            # Analyze retrieval issue
            chunks_list = parse_feishu_json_field(fields.get("retrieved_chunks", ""))
            evidence_list = parse_feishu_json_field(fields.get("evidence", ""))
            retrieval_issue = ""
            try:
                if not chunks_list:
                    retrieval_issue = "未召回任何文档"
                elif not evidence_list:
                    retrieval_issue = "无参考答案可比对"
                else:
                    any_hit = False
                    chunk_ctxs = [ch.get("context", "") if isinstance(ch, dict) else str(ch) for ch in chunks_list]
                    for ev in evidence_list:
                        ev_content = ev.get("content", "") if isinstance(ev, dict) else str(ev)
                        if not ev_content:
                            continue
                        for ctx in chunk_ctxs:
                            if _fuzzy_match(ev_content, ctx):
                                any_hit = True
                                break
                        if any_hit:
                            break
                    if any_hit:
                        retrieval_issue = "召回部分相关但未充分利用"
                    else:
                        retrieval_issue = "召回文档与目标内容不匹配"
            except (json.JSONDecodeError, TypeError):
                retrieval_issue = "召回数据解析失败"

            chunk_ids = []
            for ch in chunks_list:
                if isinstance(ch, dict) and ch.get("id") is not None:
                    chunk_ids.append(str(ch.get("id")))
            cited_ids = sorted(set(re.findall(r"\[\^(\d+)\]", answer)))
            matched_chunk_ids = []
            try:
                for ev in evidence_list:
                    ev_content = ev.get("content", "") if isinstance(ev, dict) else str(ev)
                    if not ev_content:
                        continue
                    for ch in chunks_list:
                        ctx = ch.get("context", "") if isinstance(ch, dict) else str(ch)
                        if ctx and _fuzzy_match(ev_content, ctx):
                            ch_id = ch.get("id") if isinstance(ch, dict) else None
                            if ch_id is not None:
                                matched_chunk_ids.append(str(ch_id))
                matched_chunk_ids = sorted(set(matched_chunk_ids), key=lambda x: int(x) if x.isdigit() else x)
            except Exception:
                matched_chunk_ids = []
            
            # Classify problem type with root cause attribution
            # root_cause: "retrieval" / "retrieval_granularity" / "generation" / "both"
            # Key insight: when F>=2.5 but evidence NOT in chunks (any_hit=False),
            # the model got partial credit for faithfulness to SOME chunk content,
            # but the actual evidence paragraph was never retrieved.
            # This is a retrieval granularity problem, not a generation problem.
            evidence_retrieved = (retrieval_issue == "召回部分相关但未充分利用")
            
            if evidence_retrieved and trace_score == 0 and cited_ids:
                problem_type = "生成错误"
                root_cause = "generation"
            elif faith_score == 0 and trace_score == 0 and acc_score == 0:
                problem_type = "检索未命中"
                root_cause = "retrieval"
            elif faith_score == 0:
                problem_type = "检索未命中"
                root_cause = "retrieval"
            elif not evidence_retrieved and faith_score >= PARTIAL_CAP_SCORE:
                problem_type = "检索不完整"
                root_cause = "retrieval_granularity"
            elif evidence_retrieved and trace_score == 0:
                problem_type = "生成错误"
                root_cause = "generation"
            elif evidence_retrieved and faith_score >= MAX_SCORE and comp_score == 0:
                problem_type = "生成错误"
                root_cause = "generation"
            elif evidence_retrieved and faith_score >= PARTIAL_CAP_SCORE and acc_score == 0:
                problem_type = "生成错误"
                root_cause = "generation"
            else:
                problem_type = "生成错误"
                root_cause = "both"

            matched_chunk_text = "、".join(matched_chunk_ids) if matched_chunk_ids else "未命中"
            cited_id_text = "、".join(cited_ids) if cited_ids else "未标注"
            zero_dim_text = "、".join(zero_dims)
            if root_cause == "retrieval":
                root_detail = f"检索未命中：未召回到支撑该结论的相关片段，导致 {zero_dim_text} 失分。"
            elif root_cause == "retrieval_granularity":
                root_detail = f"检索不完整：已召回相关内容，但关键证据覆盖不全，当前命中片段 ID {matched_chunk_text}，导致 {zero_dim_text} 失分。"
            elif trace_score == 0 and cited_ids:
                root_detail = f"生成错误：支撑内容已召回，但引用来源编号 {cited_id_text} 与召回片段 ID 不一致，导致 Traceability 失分。"
            elif comp_score == 0:
                root_detail = f"生成错误：支撑内容已召回，但回答遗漏关键要点，导致 Completeness 失分。"
            elif acc_score == 0:
                root_detail = f"生成错误：支撑内容已召回，但回答内容与证据不一致，导致 Accuracy 失分。"
            else:
                root_detail = f"生成错误：支撑内容已召回，但回答出现内容错配或表达偏差，导致 {zero_dim_text} 失分。"

            analysis["zero_score_records"].append({
                "record_id": rec.get("record_id"),
                "question": question,
                "intent": intent,
                "zero_dims": zero_dims,
                "answer_summary": answer[:80] if answer else "",
                "retrieval_issue": retrieval_issue,
                "problem_type": problem_type,
                "root_cause": root_cause,
                "root_detail": root_detail,
                "root_summary": root_detail,
                "faith_score": faith_score,
                "trace_score": trace_score,
                "acc_score": acc_score,
                "comp_score": comp_score
            })
    
    # Retrieval quality (Recall & MRR)
    # Retrieval quality will be calculated below after zero_score_records
    
    # Calculate recall & MRR: global numerator/denominator, not per-record average
    total_evidence_count = 0
    total_evidence_hit = 0
    mrr_sum = 0.0
    mrr_count = 0

    for rec in scored:
        fields = rec.get("fields", {})
        evidence = parse_feishu_json_field(fields.get("evidence", ""))
        chunks = parse_feishu_json_field(fields.get("retrieved_chunks", ""))
        
        if not evidence:
            continue
        
        # Count valid evidence items (denominator = all evidence with content, no filtering)
        valid_ev = []
        for ev in evidence:
            if isinstance(ev, dict):
                content = ev.get("content", "")
            elif isinstance(ev, str):
                content = ev
            else:
                continue
            if content:
                valid_ev.append(content)
        
        if not valid_ev:
            continue
        
        total_evidence_count += len(valid_ev)
        mrr_count += 1
        
        # If no chunks retrieved, all evidence items are misses
        if not chunks:
            continue
        
        # Extract text from chunks
        chunk_texts = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                chunk_texts.append(chunk.get("context", "") or chunk.get("content", ""))
            elif isinstance(chunk, str):
                chunk_texts.append(chunk)
        
        # Recall: per-evidence granularity
        for content in valid_ev:
            for ct in chunk_texts:
                if _fuzzy_match(content, ct):
                    total_evidence_hit += 1
                    break
        
        # MRR: per-record, rank of first hit chunk across all evidence
        first_hit_rank = None
        for content in valid_ev:
            for idx, ct in enumerate(chunk_texts):
                if _fuzzy_match(content, ct):
                    rank = idx + 1
                    if first_hit_rank is None or rank < first_hit_rank:
                        first_hit_rank = rank
                    break
            if first_hit_rank == 1:
                break  # Can't do better than rank 1
        
        if first_hit_rank is not None:
            mrr_sum += 1.0 / first_hit_rank
    
    analysis["retrieval_quality"]["recall"] = total_evidence_hit / total_evidence_count if total_evidence_count > 0 else 0
    analysis["retrieval_quality"]["mrr"] = mrr_sum / mrr_count if mrr_count > 0 else 0
    
    return analysis


def _text_element(text, bold=False, text_color=None):
    style = {}
    if bold:
        style["bold"] = True
    if text_color is not None:
        style["text_color"] = int(text_color)
    text_run = {"content": str(text)}
    if style:
        text_run["text_element_style"] = style
    return {"text_run": text_run}


def _text_payload(text, bold=False):
    return {
        "style": {},
        "elements": [_text_element(text, bold=bold)]
    }


def _cell_text(text, limit=200):
    value = extract_field_text(text).replace("\n", " ").strip()
    if len(value) <= limit:
        return value
    return value[:limit - 3] + "..."


def make_cell(text, bold=False, text_color=None):
    return {
        "text": str(text),
        "bold": bool(bold),
        "text_color": text_color,
    }


def _normalize_table_cell(cell, limit=200):
    if isinstance(cell, dict):
        return {
            "text": _cell_text(cell.get("text", ""), limit=limit),
            "bold": bool(cell.get("bold", False)),
            "text_color": cell.get("text_color"),
        }
    return _cell_text(cell, limit=limit)


def _table_cell_text(cell):
    if isinstance(cell, dict):
        return str(cell.get("text", ""))
    return str(cell)


def _highlight_table_cells(rows, groups):
    normalized = [[_normalize_table_cell(cell) for cell in row] for row in rows]

    def _parse_value(text, kind):
        text = str(text).strip()
        if not text or text == "不涉及":
            return None
        try:
            if kind == "int":
                return int(float(text))
            return float(text)
        except (ValueError, TypeError):
            return None

    for group in groups:
        metric_kind = group.get("kind", "float")
        mode = group.get("mode", "min")
        columns = list(group.get("columns", []))
        values = []
        for row_idx, row in enumerate(normalized):
            for col_idx in columns:
                if col_idx >= len(row):
                    continue
                value = _parse_value(_table_cell_text(row[col_idx]), metric_kind)
                if value is not None:
                    values.append((row_idx, col_idx, value))
        if not values:
            continue
        unique_values = {v for _, _, v in values}
        if len(unique_values) <= 1:
            continue
        target = group.get("target")
        if target is None:
            target = min(unique_values) if mode == "min" else max(unique_values)
        elif metric_kind == "float":
            target = round(float(target), 2)
        if target not in unique_values:
            continue
        # 整体水平已经够好时不再标"最低"
        if mode == "min" and metric_kind == "float" and target > LOW_SCORE_HIGHLIGHT_THRESHOLD:
            continue
        for row_idx, col_idx, value in values:
            if value != target:
                continue
            cell = normalized[row_idx][col_idx]
            if not isinstance(cell, dict):
                cell = make_cell(cell)
            cell["bold"] = True
            cell["text_color"] = DOCX_RED_TEXT_COLOR
            normalized[row_idx][col_idx] = cell
    return normalized


def _avg_str(data):
    if data.get("valid_count", 0) <= 0:
        return "不涉及"
    return f"{data['avg']:.2f}"


def make_heading(text, level=1):
    return {"kind": "heading", "text": str(text), "level": max(1, min(3, int(level)))}


def make_text(text, bold=False):
    return {"kind": "text", "text": str(text), "bold": bool(bold)}


def make_bullet(text, bold=False):
    return {"kind": "bullet", "text": str(text), "bold": bool(bold)}


def make_callout(children, emoji_id="bar_chart", background_color=5):
    return {
        "kind": "callout",
        "emoji_id": emoji_id,
        "background_color": background_color,
        "children": list(children),
    }


def _normalize_callout_children(children):
    normalized = []
    for child in children:
        if child["kind"] == "bullet":
            normalized.append(make_text(child["text"], bold=child.get("bold", False)))
        else:
            normalized.append(child)
    return normalized


def make_table(headers, rows, column_width=None):
    safe_headers = [_cell_text(h) for h in headers]
    safe_rows = [[_normalize_table_cell(cell) for cell in row] for row in rows]
    widths = list(column_width) if column_width else None
    if len(safe_headers) > DOCX_MAX_NATIVE_TABLE_COLUMNS:
        return {
            "kind": "table_text",
            "headers": safe_headers,
            "rows": safe_rows,
        }
    return {
        "kind": "table",
        "headers": safe_headers,
        "rows": safe_rows,
        "column_width": widths,
    }


def make_code(text, language=1):
    return {
        "kind": "code",
        "text": str(text),
        "language": int(language),
    }


def make_image(path, alt=""):
    return {
        "kind": "image",
        "path": str(path),
        "alt": str(alt),
    }


def make_split_tables(headers, rows, column_width=None, repeat_prefix_columns=1):
    return [make_table(headers, rows, column_width)]


def make_divider():
    return {"kind": "divider"}


def build_block_payload(spec):
    """Convert logical block spec to Feishu docx payload."""
    kind = spec["kind"]

    if kind == "heading":
        level = spec["level"]
        block_type = 2 + level
        key = f"heading{level}"
        return {
            "block_type": block_type,
            key: _text_payload(spec["text"], bold=True)
        }

    if kind == "text":
        return {
            "block_type": 2,
            "text": _text_payload(spec["text"], bold=spec.get("bold", False))
        }

    if kind == "bullet":
        return {
            "block_type": 12,
            "bullet": _text_payload(spec["text"], bold=spec.get("bold", False))
        }

    if kind == "code":
        return {
            "block_type": 14,
            "code": {
                "style": {
                    "language": spec.get("language", 1),
                    "wrap": True,
                },
                "elements": [_text_element(spec["text"])],
            }
        }

    if kind == "callout":
        return {
            "block_type": 19,
            "callout": {
                "background_color": spec.get("background_color", 5),
                "emoji_id": spec.get("emoji_id", "bar_chart")
            }
        }

    if kind == "table":
        headers = spec["headers"]
        rows = spec["rows"]
        column_size = len(headers)
        payload = {
            "block_type": 31,
            "table": {
                "property": {
                    "row_size": len(rows) + 1,
                    "column_size": column_size,
                }
            }
        }
        if spec.get("column_width"):
            payload["table"]["property"]["column_width"] = spec["column_width"]
        return payload

    if kind == "divider":
        return {"block_type": 22, "divider": {}}

    raise ValueError(f"不支持的 block 类型: {kind}")


def expand_block_specs(block_specs):
    """Expand logical helper specs into uploadable block specs."""
    expanded = []
    for spec in block_specs:
        if spec["kind"] != "table_text":
            expanded.append(spec)
            continue

        header_line = " | ".join(spec["headers"])
        expanded.append(make_text(header_line, bold=True))
        for row in spec["rows"]:
            expanded.append(make_text(" | ".join(row)))
    return expanded


def _docx_request(token, url, data=None, method=None, timeout=30, label="docx"):
    resp = retry_request(
        url,
        data=data,
        headers=feishu_headers(token),
        method=method,
        timeout=timeout,
        label=label
    )
    if resp.get("code") != 0:
        raise RuntimeError(f"{label} 失败: {resp.get('msg', resp)}")
    return resp


def _docx_sleep():
    time.sleep(1.0 / DOCX_WRITE_QPS + 0.05)


def _docx_url(document_id, suffix="", query=None):
    url = f"{FEISHU_BASE}/docx/v1/documents/{document_id}{suffix}"
    params = ["document_revision_id=-1"]
    if query:
        params.extend(query)
    return url + "?" + "&".join(params)


def _docx_get_all_blocks(token, document_id):
    blocks = []
    page_token = ""
    while True:
        query = ["page_size=500"]
        if page_token:
            query.append(f"page_token={urllib.parse.quote(page_token)}")
        url = _docx_url(document_id, "/blocks", query=query)
        resp = _docx_request(token, url, method="GET", label="list_docx_blocks")
        data = resp.get("data", {})
        blocks.extend(data.get("items", []))
        if not data.get("has_more"):
            break
        page_token = data.get("page_token", "")
    return blocks


def _resolve_table_cell_text_blocks(token, document_id, cell_ids):
    cell_set = set(cell_ids)
    cell_to_text = {}

    for block in _docx_get_all_blocks(token, document_id):
        block_id = block.get("block_id", "")
        if block_id not in cell_set:
            continue

        children = block.get("children", [])
        if children:
            cell_to_text[block_id] = children[0]
            continue

        table_cell = block.get("table_cell", {})
        table_children = table_cell.get("children", [])
        if table_children:
            cell_to_text[block_id] = table_children[0]

    missing = [cid for cid in cell_ids if cid not in cell_to_text]
    for cell_id in missing:
        url = _docx_url(document_id, f"/blocks/{cell_id}/children", query=["page_size=50"])
        resp = _docx_request(token, url, method="GET", label="get_table_cell_children")
        items = resp.get("data", {}).get("items", [])
        if items:
            cell_to_text[cell_id] = items[0].get("block_id", "")

    return cell_to_text


def _cleanup_table_block(token, document_id, block_id, verbose=False):
    url = _docx_url(document_id, f"/blocks/{block_id}")
    try:
        _docx_request(token, url, method="DELETE", label="delete_table_block")
    except Exception as e:
        if verbose:
            print(f"  清理空表格失败: block_id={block_id}, error={e}")


def _update_text_block(token, document_id, block_id, text, bold=False):
    batch_url = _docx_url(document_id, "/blocks/batch_update")
    _docx_request(
        token,
        batch_url,
        data={
            "requests": [{
                "block_id": block_id,
                "update_text_elements": {
                    "elements": [_text_element(text or " ", bold=bold)]
                }
            }]
        },
        method="PATCH",
        label="update_text_block"
    )
    _docx_sleep()


def _insert_table_columns(token, document_id, table_block_id, column_count, verbose=False):
    if column_count <= 0:
        return
    batch_url = _docx_url(document_id, "/blocks/batch_update")
    for _ in range(column_count):
        resp = _docx_request(
            token,
            batch_url,
            data={
                "requests": [{
                    "block_id": table_block_id,
                    "insert_table_column": {
                        "column_index": -1
                    }
                }]
            },
            method="PATCH",
            label="insert_table_column"
        )
        _docx_sleep()
        if resp.get("code") != 0:
            _cleanup_table_block(token, document_id, table_block_id, verbose=verbose)
            raise RuntimeError(f"插入表格列失败: resp={resp}")


def _list_block_children(token, document_id, block_id, page_size=100):
    url = _docx_url(document_id, f"/blocks/{block_id}/children", query=[f"page_size={page_size}"])
    resp = _docx_request(token, url, method="GET", label="list_block_children")
    return resp.get("data", {}).get("items", [])


def _upload_callout_children(token, document_id, callout_block_id, child_specs, verbose=False):
    if not child_specs:
        return []

    existing_children = _list_block_children(token, document_id, callout_block_id, page_size=50)
    created_blocks = []
    remaining_specs = list(child_specs)

    # Feishu callout blocks often come with one empty paragraph child. Reuse it
    # instead of appending new children, otherwise the UI shows an empty first line.
    if remaining_specs and existing_children and remaining_specs[0]["kind"] == "text":
        first_child = existing_children[0]
        if first_child.get("block_type") == 2 and first_child.get("block_id"):
            _update_text_block(
                token,
                document_id,
                first_child["block_id"],
                remaining_specs[0]["text"],
                bold=remaining_specs[0].get("bold", False),
            )
            created_blocks.append(first_child)
            remaining_specs = remaining_specs[1:]

    if remaining_specs:
        created_blocks.extend(
            upload_block_children(
                token,
                document_id,
                callout_block_id,
                remaining_specs,
                verbose=verbose,
            )
        )

    return created_blocks


def _create_native_table_block(token, document_id, parent_block_id, table_spec, verbose=False):
    values = [table_spec["headers"]] + table_spec["rows"]
    row_count = len(values)
    col_count = len(table_spec["headers"])
    if row_count <= 0 or col_count <= 0:
        return None

    create_rows = min(row_count, 9)
    create_cols = min(col_count, DOCX_INITIAL_TABLE_COLUMNS)
    url = _docx_url(document_id, f"/blocks/{parent_block_id}/children")
    payload = {
        "children": [{
            "block_type": 31,
            "table": {
                "property": {
                    "row_size": create_rows,
                    "column_size": create_cols,
                    "header_row": True,
                }
            }
        }],
        "index": -1
    }
    if table_spec.get("column_width"):
        payload["children"][0]["table"]["property"]["column_width"] = table_spec["column_width"][:create_cols]

    resp = _docx_request(token, url, data=payload, method="POST", label="create_table_block")
    _docx_sleep()
    created_children = resp.get("data", {}).get("children", [])
    table_block = None
    for child in created_children:
        if child.get("block_type") == 31:
            table_block = child
            break
    if table_block is None and created_children:
        table_block = created_children[0]
    if table_block is None:
        raise RuntimeError("创建表格成功但未返回 table block")

    table_block_id = table_block.get("block_id", "")
    if not table_block_id:
        raise RuntimeError(f"表格 block 缺少 block_id: {table_block}")

    if row_count > create_rows:
        batch_url = _docx_url(document_id, "/blocks/batch_update")
        for row_index in range(create_rows, row_count):
            batch_payload = {
                "requests": [{
                    "block_id": table_block_id,
                    "insert_table_row": {
                        "row_index": row_index
                    }
                }]
            }
            resp = _docx_request(token, batch_url, data=batch_payload, method="PATCH", label="insert_table_row")
            _docx_sleep()
            if resp.get("code") != 0:
                _cleanup_table_block(token, document_id, table_block_id, verbose=verbose)
                raise RuntimeError(f"插入表格行失败: row_index={row_index}, resp={resp}")

    if col_count > create_cols:
        _insert_table_columns(token, document_id, table_block_id, col_count - create_cols, verbose=verbose)

    if table_spec.get("column_width"):
        batch_url = _docx_url(document_id, "/blocks/batch_update")
        for column_index, width in enumerate(table_spec["column_width"][:col_count]):
            _docx_request(
                token,
                batch_url,
                data={
                    "requests": [{
                        "block_id": table_block_id,
                        "update_table_property": {
                            "column_width": int(width),
                            "column_index": column_index,
                            "header_row": True
                        }
                    }]
                },
                method="PATCH",
                label="update_table_property"
            )
            _docx_sleep()

    children_url = _docx_url(document_id, f"/blocks/{table_block_id}/children", query=["page_size=500"])
    child_resp = _docx_request(token, children_url, method="GET", label="list_table_cells")
    cell_ids = [item.get("block_id", "") for item in child_resp.get("data", {}).get("items", []) if item.get("block_id")]
    expected_cells = row_count * col_count
    if len(cell_ids) < expected_cells:
        _cleanup_table_block(token, document_id, table_block_id, verbose=verbose)
        raise RuntimeError(f"表格单元格数量不足: got={len(cell_ids)} expected={expected_cells}")

    cell_to_text = _resolve_table_cell_text_blocks(token, document_id, cell_ids[:expected_cells])
    update_requests = []
    for idx, value in enumerate(cell for row in values for cell in row):
        cell_id = cell_ids[idx]
        text_block_id = cell_to_text.get(cell_id)
        if not text_block_id:
            continue
        cell_text = _table_cell_text(value) or " "
        bold = idx < col_count
        text_color = None
        if isinstance(value, dict):
            bold = bool(value.get("bold", bold))
            text_color = value.get("text_color")
        update_requests.append({
            "block_id": text_block_id,
            "update_text_elements": {
                "elements": [_text_element(cell_text, bold=bold, text_color=text_color)]
            }
        })

    if not update_requests:
        _cleanup_table_block(token, document_id, table_block_id, verbose=verbose)
        raise RuntimeError("未能构造任何表格单元格更新请求")

    batch_url = _docx_url(document_id, "/blocks/batch_update")
    for req_chunk in chunked(update_requests, DOCX_BATCH_UPDATE_SIZE):
        _docx_request(
            token,
            batch_url,
            data={"requests": req_chunk},
            method="PATCH",
            label="fill_table_cells"
        )
        _docx_sleep()

    if verbose:
        print(f"  table 已创建并填充: block_id={table_block_id}, rows={row_count}, cols={col_count}")
    return table_block




def _multipart_form_data(fields, files):
    boundary = "----KnowledgeEvalDM" + uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")
    for name, file_path in files.items():
        file_path = Path(file_path)
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{name}"; filename="{file_path.name}"\r\n'.encode("utf-8")
        )
        body.extend(b"Content-Type: image/png\r\n\r\n")
        body.extend(file_path.read_bytes())
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def upload_docx_image_media(token, image_path, image_block_id, verbose=False):
    image_path = Path(image_path)
    data, content_type = _multipart_form_data(
        {
            "file_name": image_path.name,
            "parent_type": "docx_image",
            "parent_node": image_block_id,
            "size": image_path.stat().st_size,
        },
        {"file": image_path},
    )
    headers = {"Authorization": f"Bearer {token}", "Content-Type": content_type}
    resp = retry_request(
        f"{FEISHU_BASE}/drive/v1/medias/upload_all",
        data=data,
        headers=headers,
        method="POST",
        timeout=120,
        retries=3,
        label="upload_docx_image",
    )
    if resp.get("code") != 0:
        raise RuntimeError(f"上传图片素材失败: {resp.get('msg', resp)}")
    file_token = resp.get("data", {}).get("file_token")
    if not file_token:
        raise RuntimeError(f"上传图片素材成功但未返回 file_token: {resp}")
    if verbose:
        print(f"  图片素材已上传: {image_path.name}")
    return file_token


def replace_docx_image(token, document_id, image_block_id, file_token):
    url = _docx_url(document_id, f"/blocks/{image_block_id}")
    return _docx_request(
        token,
        url,
        data={"replace_image": {"token": file_token}},
        method="PATCH",
        label="replace_image",
    )


def _create_image_block(token, document_id, parent_block_id, image_spec, verbose=False):
    image_path = Path(image_spec["path"])
    url = _docx_url(document_id, f"/blocks/{parent_block_id}/children")
    created_block = None
    try:
        resp = _docx_request(
            token,
            url,
            data={"children": [{"block_type": 27, "image": {}}], "index": -1},
            method="POST",
            label="create_image_block",
        )
        _docx_sleep()
        children = resp.get("data", {}).get("children", [])
        created_block = children[0] if children else None
        image_block_id = (created_block or {}).get("block_id")
        if not image_block_id:
            raise RuntimeError(f"创建图片 block 成功但未返回 block_id: {resp}")
        file_token = upload_docx_image_media(token, image_path, image_block_id, verbose=verbose)
        replace_docx_image(token, document_id, image_block_id, file_token)
        _docx_sleep()
        if verbose:
            print(f"  图片已插入文档: block_id={image_block_id}")
        return created_block
    finally:
        try:
            if image_path.exists():
                image_path.unlink()
                if verbose:
                    print(f"  已删除本地临时图片: {image_path}")
        except Exception as e:
            print(f"  ⚠ 删除本地临时图片失败 {image_path}: {e}")


def create_wiki_report_doc(token, title, space_id=DEFAULT_WIKI_SPACE_ID, verbose=False):
    """Create a wiki docx node and return node metadata."""
    url = f"{FEISHU_BASE}/wiki/v2/spaces/{space_id}/nodes"
    body = {
        "obj_type": "docx",
        "node_type": "origin",
        "title": title
    }
    resp = _docx_request(token, url, data=body, method="POST", label="create_wiki_node")
    node = resp["data"]["node"]
    if verbose:
        print(f"  wiki node 已创建: node_token={node.get('node_token')} obj_token={node.get('obj_token')}")
    return node


def fetch_table_name(token, app_token, table_id, verbose=False):
    page_token = None
    page = 0
    while True:
        page += 1
        params = "page_size=100"
        if page_token:
            params += f"&page_token={urllib.parse.quote(page_token)}"
        url = f"{FEISHU_BASE}/bitable/v1/apps/{app_token}/tables?{params}"
        resp = retry_request(url, headers=feishu_headers(token), label=f"list_tables_{page}")
        if resp.get("code") != 0:
            raise RuntimeError(f"拉取表格元数据失败: {resp.get('msg', resp)}")
        data = resp.get("data", {})
        for item in data.get("items", []):
            if item.get("table_id") == table_id:
                return item.get("name") or table_id
        if not data.get("has_more"):
            break
        page_token = data.get("page_token")
    return table_id


def fetch_bitable_title(token, app_token, verbose=False):
    url = f"{FEISHU_BASE}/bitable/v1/apps/{app_token}"
    resp = retry_request(url, headers=feishu_headers(token), label="get_bitable_app")
    if resp.get("code") != 0:
        raise RuntimeError(f"拉取 bitable 标题失败: {resp.get('msg', resp)}")
    data = resp.get("data", {})
    app = data.get("app", {})
    return app.get("name") or app_token


def upload_block_children(token, document_id, parent_block_id, block_specs, verbose=False):
    """Upload logical block specs under a parent block."""
    block_specs = expand_block_specs(block_specs)
    if not block_specs:
        return []

    all_created = []
    buffered_specs = []

    def flush_buffer():
        nonlocal buffered_specs
        if not buffered_specs:
            return
        url = _docx_url(document_id, f"/blocks/{parent_block_id}/children")
        for spec_chunk in chunked(buffered_specs, DOCX_CHILDREN_BATCH_SIZE):
            payload = {
                "children": [build_block_payload(spec) for spec in spec_chunk],
                "index": -1
            }
            resp = _docx_request(token, url, data=payload, method="POST", label="create_docx_blocks")
            _docx_sleep()
            created_blocks = resp.get("data", {}).get("children", [])
            if len(created_blocks) != len(spec_chunk):
                raise RuntimeError(f"创建 block 数量不匹配: created={len(created_blocks)} expected={len(spec_chunk)}")

            for spec, created in zip(spec_chunk, created_blocks):
                if spec["kind"] == "callout":
                    _upload_callout_children(
                        token,
                        document_id,
                        created["block_id"],
                        _normalize_callout_children(spec.get("children", [])),
                        verbose=verbose
                    )

            all_created.extend(created_blocks)
        buffered_specs = []

    for spec in block_specs:
        if spec["kind"] not in ("table", "image"):
            buffered_specs.append(spec)
            continue

        flush_buffer()
        if spec["kind"] == "table":
            created = _create_native_table_block(
                token,
                document_id,
                parent_block_id,
                spec,
                verbose=verbose
            )
        else:
            created = _create_image_block(
                token,
                document_id,
                parent_block_id,
                spec,
                verbose=verbose
            )
        if created:
            all_created.append(created)

    flush_buffer()

    if verbose:
        print(f"  parent={parent_block_id} 已创建 {len(all_created)} 个 blocks")
    return all_created


def generate_report_markdown(analysis):
    """Generate Feishu native block JSON tree from analysis."""
    if not analysis:
        return [make_text("无可用数据。")]

    dims = analysis["dimensions"]
    intents = analysis["intents"]
    zero_records = analysis["zero_score_records"]

    active_dims = [(k, v) for k, v in dims.items() if v["valid_count"] > 0]
    if active_dims:
        dim_sorted = sorted(active_dims, key=lambda x: x[1]["avg"], reverse=True)
    else:
        dim_sorted = [(dim, dims[dim]) for dim in DIMENSIONS]
    top1_dim, top1_data = dim_sorted[0]
    top2_dim, top2_data = dim_sorted[1] if len(dim_sorted) > 1 else (top1_dim, top1_data)
    bottom1_dim, bottom1_data = dim_sorted[-1]
    bottom2_dim, bottom2_data = dim_sorted[-2] if len(dim_sorted) > 1 else (bottom1_dim, bottom1_data)

    intent_sorted = sorted(intents.items(), key=lambda x: x[1]["overall_avg"], reverse=True)
    best_intent, best_data = intent_sorted[0] if intent_sorted else ("未知", {"overall_avg": 0, "count": 0, "total_zeros": 0, "dim_avg": {d: -1 for d in DIMENSIONS}})
    worst_by_zeros = sorted(intents.items(), key=lambda x: x[1]["total_zeros"], reverse=True)
    worst_intent, worst_data = worst_by_zeros[0] if worst_by_zeros else ("未知", {"overall_avg": 0, "count": 0, "total_zeros": 0, "dim_avg": {d: -1 for d in DIMENSIONS}})
    most_common = max(intents.items(), key=lambda x: x[1]["count"]) if intents else ("未知", {"count": 0, "overall_avg": 0})

    top1_2_perfect = top1_data["count_5"] + top2_data["count_5"]
    top1_2_valid = top1_data["valid_count"] + top2_data["valid_count"]
    top1_2_pct = (top1_2_perfect / top1_2_valid * 100) if top1_2_valid > 0 else 0

    def best_dim_for_intent(data):
        return max(data["dim_avg"].items(), key=lambda x: x[1] if x[1] != -1 else -999)

    def worst_dim_for_intent(data):
        return min(data["dim_avg"].items(), key=lambda x: x[1] if x[1] != -1 else 999)

    best_top_dim = best_dim_for_intent(best_data)
    worst_bottom_dim = worst_dim_for_intent(worst_data)
    comp_data = dims.get("Completeness", {})
    faith_data = dims.get("Faithfulness", {})
    trace_data = dims.get("Traceability", {})

    blocks = []
    blocks.append(make_heading("1. 评估概览", level=1))
    overview_callout = [
        make_text("关键结论", bold=True),
        make_bullet(f"最强项：{top1_dim}（{top1_data['avg']:.2f}）和 {top2_dim}（{top2_data['avg']:.2f}），约 {top1_2_pct:.1f}% 记录 5.0 分"),
        make_bullet(f"最佳意图：{best_intent}，{best_top_dim[0]} 得分 {best_top_dim[1]:.2f}，0 分 {best_data['total_zeros']} 条"),
        make_bullet(f"最弱指标：{bottom1_dim}（{bottom1_data['avg']:.2f}）和 {bottom2_dim}（{bottom2_data['avg']:.2f}），其中 {bottom1_data['count_0']} 条为 0 分"),
        make_bullet(f"重灾区意图：{worst_intent}，{worst_bottom_dim[0]} 仅 {worst_bottom_dim[1]:.2f}，0 分 {worst_data['total_zeros']}/{max(worst_data['count'], 1)}"),
    ]
    blocks.append(make_callout(overview_callout, emoji_id="bar_chart"))

    intent_dist = "，".join([f"{k}({v['count']})" for k, v in intents.items()]) if intents else "无"
    blocks.append(make_heading("1.1 测评项目", level=2))
    blocks.append(make_table(
        ["项目", "内容"],
        [
            ["评估总条数", str(analysis["total_records"])],
            ["维度", "9 个（Faithfulness, Traceability, Accuracy, Relevance, Reason, Completeness, Action, Clarity, Refuse）"],
            ["量表", "0.0-5.0 分（保留 1 位小数；-1=不涉及）"],
            ["意图分布", intent_dist],
        ],
        column_width=[180, 760]
    ))

    blocks.append(make_heading("1.2 综合得分", level=2))
    blocks.append(make_table(
        ["指标", "数值"],
        [
            ["综合均分", f"{analysis['overall_avg']:.2f}"],
            ["高质量占比（≥4.5）", f"{analysis['high_quality_pct']:.1f}%"],
            ["全满分占比", f"{analysis['perfect_pct']:.1f}%"],
            ["低质量占比（<2.5）", f"{analysis['low_quality_pct']:.1f}%"],
        ],
        column_width=[220, 220]
    ))

    recall = analysis["retrieval_quality"]["recall"]
    mrr = analysis["retrieval_quality"]["mrr"]
    blocks.append(make_heading("1.3 检索召回质量", level=2))
    blocks.append(make_table(
        ["指标", "数值", "说明"],
        [
            ["Recall", f"{recall:.2%}", "Evidence 片段被 retrieved_chunks 覆盖的比例"],
            ["MRR", f"{mrr:.2%}", "首个命中 chunk 的平均倒数排名"],
        ],
        column_width=[140, 140, 560]
    ))

    blocks.append(make_divider())
    blocks.append(make_heading("2. 各维度得分分布", level=1))
    dimension_callout = [
        make_text("关键分析", bold=True),
        make_bullet(f"最佳维度：{top1_dim}（{top1_data['avg']:.2f}），{top1_data['count_5']} 条 5.0 分"),
        make_bullet(f"核心短板：{bottom1_dim}（{bottom1_data['avg']:.2f}），{bottom1_data['count_0']} 条 0 分"),
    ]
    if comp_data.get("count_0", 0) > 0:
        dimension_callout.append(make_bullet(f"完整性问题：{comp_data['count_0']} 条记录 Completeness=0，回答遗漏关键信息"))
    if faith_data.get("count_0", 0) > 0 or trace_data.get("count_0", 0) > 0:
        dimension_callout.append(make_bullet(f"检索忠实度：Faithfulness {faith_data['count_0']} 条 0 分，Traceability {trace_data['count_0']} 条 0 分"))
    blocks.append(make_callout(dimension_callout, emoji_id="mag"))

    blocks.append(make_heading("检索忠实性维度", level=2))
    retrieval_avg_values = [dims[dim]["avg"] for dim in ["Faithfulness", "Traceability"] if dims[dim].get("valid_count", 0) > 0]
    retrieval_avg_target = min(retrieval_avg_values) if len(set(retrieval_avg_values)) > 1 else None
    retrieval_zero_values = [dims[dim]["count_0"] for dim in ["Faithfulness", "Traceability"]]
    retrieval_zero_target = max(retrieval_zero_values) if len(set(retrieval_zero_values)) > 1 else None
    retrieval_rows = []
    for dim in ["Faithfulness", "Traceability"]:
        d = dims[dim]
        retrieval_rows.append([dim, _avg_str(d), str(d["count_5"]), str(d["count_4"]), str(d["count_3"]), str(d["count_lt3"]), str(d["count_0"]), str(d["valid_count"])])
    retrieval_rows = _highlight_table_cells(retrieval_rows, [
        {"columns": [1], "mode": "min", "kind": "float", "target": retrieval_avg_target},
        {"columns": [6], "mode": "max", "kind": "int", "target": retrieval_zero_target},
    ])
    blocks.append(make_table(
        ["维度", "均分", "5.0分", "≥4.0", "3.0-3.9", "<3.0", "0分", "有效条数"],
        retrieval_rows,
        column_width=[160, 100, 80, 80, 90, 80, 80, 120]
    ))

    blocks.append(make_heading("回答质量维度", level=2))
    quality_dim_names = ["Accuracy", "Relevance", "Reason", "Completeness", "Action", "Clarity", "Refuse"]
    quality_avg_values = [dims[dim]["avg"] for dim in quality_dim_names if dims[dim].get("valid_count", 0) > 0]
    quality_avg_target = min(quality_avg_values) if len(set(quality_avg_values)) > 1 else None
    quality_zero_values = [dims[dim]["count_0"] for dim in quality_dim_names]
    quality_zero_target = max(quality_zero_values) if len(set(quality_zero_values)) > 1 else None
    quality_rows = []
    for dim in quality_dim_names:
        d = dims[dim]
        quality_rows.append([dim, _avg_str(d), str(d["count_5"]), str(d["count_4"]), str(d["count_3"]), str(d["count_lt3"]), str(d["count_0"]), str(d["valid_count"]), str(d["na_count"])])
    quality_rows = _highlight_table_cells(quality_rows, [
        {"columns": [1], "mode": "min", "kind": "float", "target": quality_avg_target},
        {"columns": [6], "mode": "max", "kind": "int", "target": quality_zero_target},
    ])
    blocks.append(make_table(
        ["维度", "均分", "5.0分", "≥4.0", "3.0-3.9", "<3.0", "0分", "有效条数", "不涉及"],
        quality_rows,
        column_width=[160, 100, 80, 80, 90, 80, 80, 120, 100]
    ))

    blocks.append(make_heading("典型案例", level=2))
    if zero_records:
        miss_count = sum(1 for r in zero_records if r.get("problem_type") == "检索未命中")
        partial_count = sum(1 for r in zero_records if r.get("problem_type") == "检索不完整")
        generation_count = sum(1 for r in zero_records if r.get("problem_type") == "生成错误")

        parts = []
        if miss_count:
            parts.append(f"检索未命中 {miss_count} 条")
        if partial_count:
            parts.append(f"检索不完整 {partial_count} 条")
        if generation_count:
            parts.append(f"生成错误 {generation_count} 条")
        blocks.append(make_text(f"0 分记录归因：{'、'.join(parts)}（详见第 4 章）"))

        shown_causes = set()
        cause_label = {"retrieval": "检索未命中", "retrieval_granularity": "检索不完整", "generation": "生成错误", "both": "生成错误"}
        for rec in zero_records:
            rc = rec.get("root_cause", "both")
            if rc in shown_causes:
                continue
            shown_causes.add(rc)
            zero_dims_str = "、".join(rec["zero_dims"])
            blocks.append(make_bullet(f"{cause_label.get(rc, rc)} | {rec['problem_type']}：{_cell_text(rec['question'], 80)}"))
            blocks.append(make_text(f"{zero_dims_str} = 0 分 | {rec.get('root_detail', '')}"))
    else:
        blocks.append(make_text("无 0 分记录。"))

    blocks.append(make_divider())
    blocks.append(make_heading("3. 按意图类型分析", level=1))
    blocks.append(make_callout([
        make_text("关键分析", bold=True),
        make_bullet(f"最优意图：{best_intent}（均分 {best_data['overall_avg']:.2f}），共 {best_data['count']} 条"),
        make_bullet(f"重灾区意图：{worst_intent}（均分 {worst_data['overall_avg']:.2f}），0 分 {worst_data['total_zeros']} 条"),
        make_bullet(f"数量最多：{most_common[0]}（{most_common[1]['count']} 条），均分 {most_common[1]['overall_avg']:.2f}"),
    ], emoji_id="pushpin"))

    blocks.append(make_heading("各意图类型指标均分", level=2))
    intent_avg_rows = []
    for intent, data in sorted(intents.items(), key=lambda x: x[1]["overall_avg"], reverse=True):
        row = [intent, str(data["count"])]
        for dim in DIMENSIONS:
            avg = data["dim_avg"][dim]
            row.append(f"{avg:.2f}" if avg != -1 else "不涉及")
        row.append(str(data["total_zeros"]))
        intent_avg_rows.append(row)
    intent_avg_rows = _highlight_table_cells(intent_avg_rows, [
        {"columns": list(range(2, 2 + len(DIMENSIONS))), "mode": "min", "kind": "float"},
        {"columns": [2 + len(DIMENSIONS)], "mode": "max", "kind": "int"},
    ])
    blocks.extend(make_split_tables(
        ["意图类型", "条数"] + DIMENSIONS + ["0分记录数"],
        intent_avg_rows or [["无", "0"] + ["不涉及"] * len(DIMENSIONS) + ["0"]],
        column_width=[160, 80] + [90] * len(DIMENSIONS) + [90],
        repeat_prefix_columns=2
    ))

    blocks.append(make_heading("0 分在各维度的分布", level=2))
    zero_dist_rows = []
    for intent, data in sorted(intents.items(), key=lambda x: x[1]["total_zeros"], reverse=True):
        row = [intent] + [str(data["zero_counts"][dim]) for dim in DIMENSIONS]
        zero_dist_rows.append(row)
    zero_dist_rows = _highlight_table_cells(zero_dist_rows, [
        {"columns": list(range(1, 1 + len(DIMENSIONS))), "mode": "max", "kind": "int"}
    ])
    blocks.extend(make_split_tables(
        ["意图类型"] + DIMENSIONS,
        zero_dist_rows or [["无"] + ["0"] * len(DIMENSIONS)],
        column_width=[160] + [90] * len(DIMENSIONS),
        repeat_prefix_columns=1
    ))

    blocks.append(make_divider())
    blocks.append(make_heading("4. 问题诊断：0 分记录分析", level=1))
    blocks.append(make_text(f"共 {len(zero_records)} 条记录出现了至少一个维度 0 分。"))
    if zero_records:
        retrieval_miss = [r for r in zero_records if r.get("problem_type") == "检索未命中"]
        retrieval_partial = [r for r in zero_records if r.get("problem_type") == "检索不完整"]
        generation_caused = [r for r in zero_records if r.get("problem_type") == "生成错误"]

        diagnosis_callout = [make_text("问题归因", bold=True)]
        if retrieval_miss:
            diagnosis_callout.append(make_bullet(f"检索未命中：{len(retrieval_miss)} 条，未召回到支撑结论的证据，导致后续指标失分"))
        if retrieval_partial:
            diagnosis_callout.append(make_bullet(f"检索不完整：{len(retrieval_partial)} 条，召回到部分相关内容，但关键证据覆盖不全"))
        if generation_caused:
            diagnosis_callout.append(make_bullet(f"生成错误：{len(generation_caused)} 条，证据已召回且覆盖充分，但生成阶段出现内容错配或来源编号错配"))
        blocks.append(make_callout(diagnosis_callout, emoji_id="mag"))

        by_type = {}
        for rec in zero_records:
            by_type.setdefault(rec["problem_type"], []).append(rec)

        type_order = ["检索未命中", "检索不完整", "生成错误"]
        cause_label = {"retrieval": "检索未命中", "retrieval_granularity": "检索不完整", "generation": "生成错误", "both": "生成错误"}

        for ptype in type_order:
            if ptype not in by_type:
                continue
            recs = by_type[ptype]
            rc = recs[0].get("root_cause", "both")
            blocks.append(make_heading(f"{ptype}（{len(recs)} 条，{cause_label.get(rc, rc)}）", level=2))
            blocks.append(make_text(recs[0].get("root_detail", "")))

            rows = []
            for rec in recs:
                zero_dims_str = "、".join(rec["zero_dims"])
                f_score = rec.get("faith_score", -1)
                t_score = rec.get("trace_score", -1)
                a_score = rec.get("acc_score", -1)
                c_score = rec.get("comp_score", -1)
                rows.append([
                    _cell_text(rec["question"], 80),
                    f"F={f_score} T={t_score} A={a_score} C={c_score}",
                    zero_dims_str,
                    rec.get("root_summary", ""),
                ])
            blocks.append(make_table(
                ["回复问题", "关键分数", "0分指标", "问题归因"],
                rows,
                column_width=[340, 180, 180, 360]
            ))

    blocks.append(make_divider())
    blocks.append(make_heading("5. 改进建议", level=1))

    suggestions_retrieval = []
    if analysis["retrieval_quality"]["recall"] < 0.7:
        suggestions_retrieval.append("召回率偏低，优化检索策略（混合检索、重排序）")
    if trace_data.get("count_0", 0) > 0:
        suggestions_retrieval.append(f"{trace_data['count_0']} 条 Traceability=0，检查来源标注逻辑")
    if faith_data.get("count_0", 0) > 0:
        suggestions_retrieval.append(f"{faith_data['count_0']} 条 Faithfulness=0，审查 chunk 质量和相关性")
    if not suggestions_retrieval:
        suggestions_retrieval.append("检索质量整体良好，保持现有策略")

    retrieval_children = [make_text("检索层（短期）", bold=True)]
    retrieval_children.extend(make_bullet(item) for item in suggestions_retrieval)
    blocks.append(make_callout(retrieval_children, emoji_id="pushpin"))

    suggestions_generation = []
    if comp_data.get("count_0", 0) > 0:
        suggestions_generation.append(f"{comp_data['count_0']} 条 Completeness=0，强化完整性检查")
    if dims.get("Clarity", {}).get("count_0", 0) > 0:
        suggestions_generation.append(f"{dims['Clarity']['count_0']} 条 Clarity=0，优化表达清晰度")
    if dims.get("Action", {}).get("count_0", 0) > 0:
        suggestions_generation.append(f"{dims['Action']['count_0']} 条 Action=0，补充操作步骤指导")
    if not suggestions_generation:
        suggestions_generation.append("生成质量整体优秀，持续优化 prompt")

    generation_children = [make_text("生成层（中期）", bold=True)]
    generation_children.extend(make_bullet(item) for item in suggestions_generation)
    blocks.append(make_callout(generation_children, emoji_id="pushpin"))

    blocks.append(make_divider())
    blocks.append(make_heading("6. 附录：评估维度说明", level=1))
    blocks.append(make_table(
        ["维度", "定义"],
        [
            ["Faithfulness", "回答是否忠实于 RAG 召回的文档"],
            ["Traceability", "来源编号是否能对应召回片段并支撑断言"],
            ["Accuracy", "事实是否正确"],
            ["Relevance", "是否围绕问题回答"],
            ["Reason", "推理逻辑是否完整"],
            ["Completeness", "是否覆盖所有关键点"],
            ["Action", "操作步骤是否清晰可执行"],
            ["Clarity", "表达是否清晰无歧义"],
            ["Refuse", "敏感内容是否合规拒答"],
        ],
        column_width=[180, 620]
    ))

    return blocks


# ─── Main logic ───

def build_update_records(eval_results, model_config=None):
    """Convert eval results to feishu batch_update format."""
    records = []
    for item in eval_results:
        fields = {}
        scores = item["scores"]
        for dim in DIMENSIONS:
            s = scores[dim]
            fields[eval_field_name(dim, "score", model_config)] = s["score"]
            fields[eval_field_name(dim, "reason", model_config)] = s["reason"]
        records.append({
            "record_id": item["record_id"],
            "fields": fields
        })
    return records


def project_records_for_model(records, model_config):
    """Return records whose eval fields are mapped from a model prefix to base names."""
    projected = []
    for rec in records:
        fields = dict(rec.get("fields", {}))
        for dim in DIMENSIONS:
            score_name = eval_field_name(dim, "score", model_config)
            reason_name = eval_field_name(dim, "reason", model_config)
            if score_name in fields:
                fields[f"{dim}_score"] = fields.get(score_name)
            if reason_name in fields:
                fields[f"{dim}_reason"] = fields.get(reason_name)
        copied = dict(rec)
        copied["fields"] = fields
        projected.append(copied)
    return projected


def resolve_model_configs(model_keys=None):
    keys = model_keys or DEFAULT_MODEL_KEYS
    configs = []
    for key in keys:
        if key not in MODEL_CONFIGS:
            raise ValueError(f"未知模型: {key}; 可选: {', '.join(MODEL_CONFIGS)}")
        cfg = dict(MODEL_CONFIGS[key])
        cfg["key"] = key
        configs.append(cfg)
    return configs


def generate_report_file(token_mgr, app_token, table_id, verbose=False, wiki_space_id=DEFAULT_WIKI_SPACE_ID, model_config=None, table_name=None):
    """Generate analysis report and upload it as a Feishu wiki docx document."""
    token = token_mgr.get()
    all_records = fetch_all_records(token, app_token, table_id, verbose)
    print(f"✓ 拉取 {len(all_records)} 条记录")

    if model_config:
        all_records = project_records_for_model(all_records, model_config)
    analysis = analyze_scores(all_records)
    if not analysis:
        print("✗ 无可用评估数据，无法生成报告")
        return None

    report_blocks = generate_report_markdown(analysis)
    try:
        table_name = table_name or fetch_bitable_title(token, app_token, verbose=verbose)
    except Exception as e:
        table_name = table_id
        if verbose:
            print(f"  拉取 bitable 标题失败，回退为 table_id: {e}")
    report_title = f"知识回复评估报告_{table_name}_{model_config['label']}" if model_config else f"知识回复评估报告_{table_name}"
    node = create_wiki_report_doc(token, report_title, space_id=wiki_space_id, verbose=verbose)
    document_id = node.get("obj_token")
    if not document_id:
        raise RuntimeError(f"创建 wiki 节点成功但未返回 document_id/obj_token: {node}")

    upload_block_children(token, document_id, document_id, report_blocks, verbose=verbose)
    wiki_url = f"https://my.feishu.cn/wiki/{node['node_token']}"
    print(f"✓ 报告已上传: {wiki_url}")
    print(f"  document_id: {document_id}")
    print(f"  wiki_space_id: {wiki_space_id}")
    return {
        "title": report_title,
        "node_token": node["node_token"],
        "document_id": document_id,
        "wiki_url": wiki_url,
    }


def chunked(items, size):
    """Yield fixed-size chunks."""
    for i in range(0, len(items), size):
        yield items[i:i + size]



def _score_value(fields, dim, model_config):
    raw = fields.get(eval_field_name(dim, "score", model_config))
    return parse_score_value(raw)


def _mean(values):
    values = [v for v in values if v is not None and v != -1]
    return statistics.mean(values) if values else None


def _group_dimension_avgs(records, model_config, group_field):
    grouped = {}
    for rec in records:
        fields = rec.get("fields", {})
        group = extract_field_text(fields.get(group_field)) or "未填写"
        if group not in grouped:
            grouped[group] = {dim: [] for dim in DIMENSIONS}
        for dim in DIMENSIONS:
            score = _score_value(fields, dim, model_config)
            if score is not None and score != -1:
                grouped[group][dim].append(score)
    result = {}
    for group, dim_values in grouped.items():
        result[group] = {dim: (_mean(vals) if vals else None) for dim, vals in dim_values.items()}
    return result


def _fmt_avg(value):
    return "-" if value is None else f"{value:.2f}"

def _norm_consistency_score(value):
    return None if value is None or value == -1 else value


def is_both_not_applicable(a, b):
    return a == -1 and b == -1


def consistency_diff(a, b):
    a_norm = _norm_consistency_score(a)
    b_norm = _norm_consistency_score(b)
    if a_norm is None and b_norm is None:
        return None
    if a_norm is None or b_norm is None:
        return None
    return abs(a_norm - b_norm)


def is_consistent_score_pair(a, b):
    diff = consistency_diff(a, b)
    return diff is not None and diff <= CONSISTENCY_TOLERANCE


def _fmt_diff(value):
    return "-" if value is None else f"{value:.2f}"


def build_scene_bar_chart_spec(records, model_a, model_b):
    """Build a VChart grouped bar chart spec for Feishu chart components."""
    scene_a = _group_dimension_avgs(records, model_a, "scene")
    scene_b = _group_dimension_avgs(records, model_b, "scene")
    data = []
    for scene in sorted(set(scene_a) | set(scene_b)):
        a_avg = _mean([v for v in scene_a.get(scene, {}).values() if v is not None])
        b_avg = _mean([v for v in scene_b.get(scene, {}).values() if v is not None])
        if a_avg is not None:
            data.append({"scene": scene, "model": model_a["label"], "score": round(a_avg, 4)})
        if b_avg is not None:
            data.append({"scene": scene, "model": model_b["label"], "score": round(b_avg, 4)})
    return {
        "type": "bar",
        "title": {"visible": True, "text": "各场景综合均分对比"},
        "data": [{"id": "scene_score", "values": data}],
        "xField": ["scene", "model"],
        "yField": "score",
        "seriesField": "model",
        "axes": [
            {"orient": "bottom", "type": "band", "title": {"visible": True, "text": "场景"}},
            {"orient": "left", "type": "linear", "min": SCORE_AXIS_MIN, "max": SCORE_AXIS_MAX, "title": {"visible": True, "text": "模型均分"}},
        ],
        "legends": {"visible": True, "orient": "top"},
        "label": {"visible": True, "formatMethod": "datum => datum.score.toFixed(2)"},
    }




def _record_dimension_avg(records, model_config, dim):
    values = []
    for rec in records:
        score = _score_value(rec.get("fields", {}), dim, model_config)
        if score is not None and score != -1:
            values.append(score)
    return _mean(values)


def _group_overall_avgs(records, model_config, group_field):
    grouped = _group_dimension_avgs(records, model_config, group_field)
    result = {}
    for group, dim_avgs in grouped.items():
        result[group] = _mean([v for v in dim_avgs.values() if v is not None])
    return result


def generate_grouped_bar_chart_image(title, x_label, categories, model_a, model_b, a_values, b_values, filename_prefix):
    """Generate a grouped bar chart PNG and return its local path."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_prefix = re.sub(r"[^a-zA-Z0-9_-]+", "_", filename_prefix).strip("_") or "chart"
    image_path = CHARTS_DIR / f"{safe_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.png"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in font_candidates:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False

    width = max(9, min(20, 0.9 * max(1, len(categories)) + 5))
    fig, ax = plt.subplots(figsize=(width, 5.8), dpi=160)
    x = list(range(len(categories)))
    bar_width = 0.36
    ax.bar([i - bar_width / 2 for i in x], a_values, bar_width, label=model_a["label"], color="#BAD1E4")
    ax.bar([i + bar_width / 2 for i in x], b_values, bar_width, label=model_b["label"], color="#F8C9C9")
    ax.set_title(title, fontsize=15, pad=14)
    ax.set_xlabel(x_label)
    ax.set_ylabel("模型均分")
    ax.set_ylim(SCORE_AXIS_MIN, SCORE_AXIS_MAX)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)
    return image_path


def generate_dimension_bar_chart_image(records, model_a, model_b):
    categories = list(DIMENSIONS)
    a_values = [(_record_dimension_avg(records, model_a, dim) or 0) for dim in categories]
    b_values = [(_record_dimension_avg(records, model_b, dim) or 0) for dim in categories]
    return generate_grouped_bar_chart_image(
        "各指标总体均分对比",
        "指标",
        categories,
        model_a,
        model_b,
        a_values,
        b_values,
        "consistency_dimension_bar",
    )


def generate_scene_bar_chart_image(records, model_a, model_b):
    scene_a = _group_overall_avgs(records, model_a, "scene")
    scene_b = _group_overall_avgs(records, model_b, "scene")
    categories = sorted(set(scene_a) | set(scene_b))
    a_values = [(scene_a.get(scene) or 0) for scene in categories]
    b_values = [(scene_b.get(scene) or 0) for scene in categories]
    return generate_grouped_bar_chart_image(
        "各场景综合均分对比",
        "场景",
        categories,
        model_a,
        model_b,
        a_values,
        b_values,
        "consistency_scene_bar",
    )


def generate_intent_bar_chart_image(records, model_a, model_b):
    intent_a = _group_overall_avgs(records, model_a, "intent")
    intent_b = _group_overall_avgs(records, model_b, "intent")
    categories = sorted(set(intent_a) | set(intent_b))
    a_values = [(intent_a.get(intent) or 0) for intent in categories]
    b_values = [(intent_b.get(intent) or 0) for intent in categories]
    return generate_grouped_bar_chart_image(
        "各意图类型综合均分对比",
        "意图类型",
        categories,
        model_a,
        model_b,
        a_values,
        b_values,
        "consistency_intent_bar",
    )

def _avg_diff_rows(records, model_a, model_b, group_field):
    a_stats = _group_dimension_avgs(records, model_a, group_field)
    b_stats = _group_dimension_avgs(records, model_b, group_field)
    rows = []
    for group in sorted(set(a_stats) | set(b_stats)):
        for dim in DIMENSIONS:
            a = a_stats.get(group, {}).get(dim)
            b = b_stats.get(group, {}).get(dim)
            if a is None and b is None:
                continue
            if is_both_not_applicable(a, b):
                continue
            diff = consistency_diff(a, b)
            status = "一致" if is_consistent_score_pair(a, b) else "需关注"
            rows.append([group, dim, _fmt_avg(a), _fmt_avg(b), _fmt_diff(diff), status])
    return rows


def collect_inconsistent_records(records, model_a, model_b, limit=80, skill_files=None):
    items = []
    skill_files = skill_files or {}
    rubric_text = skill_files.get("rubric.md", "")
    scene_rules_text = skill_files.get("scene-rules.md", "")
    intent_map_text = skill_files.get("intent-map.md", "")
    for rec in records:
        fields = rec.get("fields", {})
        for dim in DIMENSIONS:
            a = _score_value(fields, dim, model_a)
            b = _score_value(fields, dim, model_b)
            if is_both_not_applicable(a, b):
                continue
            if is_consistent_score_pair(a, b):
                continue
            if a is None or b is None:
                continue
            scene = extract_field_text(fields.get("scene"))
            intent = extract_field_text(fields.get("intent"))
            item = {
                "question": extract_field_text(fields.get("question"))[:500],
                "answer": extract_field_text(fields.get("answer"))[:3000],
                "scene": scene,
                "intent": intent,
                "dimension": dim,
                model_a["label"]: {
                    "score": a,
                    "reason": extract_field_text(fields.get(eval_field_name(dim, "reason", model_a)))[:300],
                },
                model_b["label"]: {
                    "score": b,
                    "reason": extract_field_text(fields.get(eval_field_name(dim, "reason", model_b)))[:300],
                },
            }
            if dim in ROUND1_DIMENSIONS:
                item["evaluation_basis"] = {
                    "retrieved_chunks": extract_field_text(fields.get("retrieved_chunks"))[:6000]
                }
            else:
                item["evaluation_basis"] = {
                    "evidence": extract_field_text(fields.get("evidence"))[:6000]
                }
            if skill_files:
                item["rules"] = {
                    "round2_common_rules": _extract_round2_common_rules(rubric_text) if dim in ROUND2_DIMENSIONS else "",
                    "dimension_rubric": _extract_dimension_rubric(rubric_text, dim),
                    "scene_rules": get_scene_rules(scene_rules_text, scene),
                    "intent_rule": _extract_intent_rule(intent_map_text, intent),
                }
            items.append(item)
    return items[:limit]


def call_claude_for_consistency(inconsistent_items, verbose=False):
    if not inconsistent_items:
        return f"未发现超过 {CONSISTENCY_TOLERANCE:.2f} 分阈值的记录级分歧，无需调用 Claude 辅助判断。"
    token = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("CLAUDE_API_KEY")
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    model = os.environ.get("ANTHROPIC_MODEL") or os.environ.get("CLAUDE_MODEL") or "MaaS_Cl_Sonnet_4.6_20260217"
    if not token or not base_url:
        return "未设置 ANTHROPIC_AUTH_TOKEN/ANTHROPIC_BASE_URL，跳过 Claude 辅助判断；本报告仅包含本地一致性统计。"

    prompt = f"""你是知识问答评估一致性审查专家。下面是 Kimi 与 DeepSeek 在同一批记录上的评分分歧样本。每条样本包含 question、answer、当前指标对应的 evaluation_basis、实际评分 rules，以及两个模型的分数与 reason。

请严格依据样本中的 rules 和 evaluation_basis，判断哪个模型的分数与 reason 更合理。Faithfulness / Traceability 只基于 retrieved_chunks 判断；其他指标只基于 evidence、question 和 answer 判断。不要重新定义评分标准，也不要因为某个 reason 写得更详细就认为该模型更合理。

分歧样本 JSON：
{json.dumps(inconsistent_items, ensure_ascii=False, indent=2)}

输出要求：
1. 先给总体判断。
2. 按场景/意图/指标总结哪个模型更准确。
3. 列出需要人工复核的高风险分歧。
"""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 4096,
        "tool_choice": "none",
    }

    def claude_request_unverified():
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(base_url, data=data, headers=headers, method="POST")
        context = ssl._create_unverified_context()
        try:
            resp = urllib.request.urlopen(req, timeout=180, context=context)
            text = resp.read().decode("utf-8")
            return json.loads(text) if text else {}
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(text) if text else {}
            except json.JSONDecodeError:
                parsed = {"raw_body": text}
            raise RuntimeError(f"HTTP {e.code}: {parsed}") from e

    try:
        last_err = None
        for attempt in range(2):
            try:
                resp = claude_request_unverified()
                break
            except Exception as e:
                last_err = e
                wait = 2 ** attempt
                print(f"  ⚠ claude_consistency 第{attempt+1}次失败: {e}, {wait}s后重试...")
                time.sleep(wait)
        else:
            raise last_err
        content = resp.get("choices", [{}])[0].get("message", {}).get("content")
        if content:
            return content.strip()
        # Handle tool_calls response (extract arguments text as fallback)
        tool_calls = resp.get("choices", [{}])[0].get("message", {}).get("tool_calls")
        if tool_calls:
            parts = []
            for tc in tool_calls:
                args = tc.get("function", {}).get("arguments", "")
                if isinstance(args, str):
                    try:
                        parsed_args = json.loads(args)
                        parts.append(parsed_args.get("prompt", "") or parsed_args.get("description", "") or args)
                    except json.JSONDecodeError:
                        parts.append(args)
            if parts:
                return "\n".join(parts).strip() or f"Claude 返回了工具调用但无可用文本内容"
        if "content" in resp and isinstance(resp["content"], list):
            return "\n".join(part.get("text", "") for part in resp["content"] if isinstance(part, dict)).strip()
        return f"Claude 返回格式未识别: {str(resp)[:500]}"
    except Exception as e:
        if verbose:
            print(f"  Claude 一致性分析失败: {e}")
        return f"Claude 辅助判断调用失败: {e}"


def generate_consistency_report_blocks(records, model_a, model_b, claude_summary):
    scene_rows = _avg_diff_rows(records, model_a, model_b, "scene")
    intent_rows = _avg_diff_rows(records, model_a, model_b, "intent")
    inconsistent_items = collect_inconsistent_records(records, model_a, model_b, limit=30)
    total_pairs = 0
    inconsistent_pairs = 0
    for rec in records:
        fields = rec.get("fields", {})
        for dim in DIMENSIONS:
            a = _score_value(fields, dim, model_a)
            b = _score_value(fields, dim, model_b)
            if a is None or b is None:
                continue
            if is_both_not_applicable(a, b):
                continue
            total_pairs += 1
            if not is_consistent_score_pair(a, b):
                inconsistent_pairs += 1
    consistency_pct = ((total_pairs - inconsistent_pairs) / total_pairs * 100) if total_pairs else 0

    rows_by_dim = []
    for dim in DIMENSIONS:
        a_values = []
        b_values = []
        for rec in records:
            fields = rec.get("fields", {})
            a = _score_value(fields, dim, model_a)
            b = _score_value(fields, dim, model_b)
            if a is not None and a != -1:
                a_values.append(a)
            if b is not None and b != -1:
                b_values.append(b)
        a_avg = _mean(a_values)
        b_avg = _mean(b_values)
        diff = consistency_diff(a_avg, b_avg)
        status = "-" if a_avg is None and b_avg is None else ("一致" if is_consistent_score_pair(a_avg, b_avg) else "需关注")
        rows_by_dim.append([dim, _fmt_avg(a_avg), _fmt_avg(b_avg), _fmt_diff(diff), status])

    sample_rows = []
    for item in inconsistent_items[:20]:
        sample_rows.append([
            item.get("question", ""), item.get("scene", ""), item.get("intent", ""), item["dimension"],
            str(item[model_a["label"]]["score"]), item[model_a["label"]]["reason"],
            str(item[model_b["label"]]["score"]), item[model_b["label"]]["reason"],
        ])

    dimension_chart_path = generate_dimension_bar_chart_image(records, model_a, model_b)
    scene_chart_path = generate_scene_bar_chart_image(records, model_a, model_b)
    intent_chart_path = generate_intent_bar_chart_image(records, model_a, model_b)

    blocks = [
        make_heading("一致性对比分析报告", level=1),
        make_callout([
            make_text("模型一致性概览", bold=True),
            make_bullet(f"对比模型：{model_a['label']} vs {model_b['label']}"),
            make_bullet(f"可接受误差：{CONSISTENCY_TOLERANCE:.2f} 分；有效指标对：{total_pairs}；超阈值分歧：{inconsistent_pairs}"),
            make_bullet(f"记录级指标一致率：{consistency_pct:.1f}%"),
        ], emoji_id="bar_chart"),
        make_heading("各指标总体均分柱状图", level=2),
        make_image(dimension_chart_path, alt="各指标 Kimi 与 DeepSeek 总体均分柱状图"),
        make_heading("各指标总体均分对比", level=2),
        make_table(["指标", model_a["label"], model_b["label"], "差异", "结论"], rows_by_dim, column_width=[170, 120, 120, 120, 120]),
        make_heading("各场景综合均分柱状图", level=2),
        make_image(scene_chart_path, alt="各场景 Kimi 与 DeepSeek 综合均分柱状图"),
        make_heading("各场景各指标均分对比", level=2),
    ]
    blocks.extend(make_split_tables(["场景", "指标", model_a["label"], model_b["label"], "差异", "结论"], scene_rows, column_width=[180, 150, 110, 110, 100, 120]))
    blocks.append(make_heading("各意图类型综合均分柱状图", level=2))
    blocks.append(make_image(intent_chart_path, alt="各意图类型 Kimi 与 DeepSeek 综合均分柱状图"))
    blocks.append(make_heading("各意图各指标均分对比", level=2))
    blocks.extend(make_split_tables(["意图", "指标", model_a["label"], model_b["label"], "差异", "结论"], intent_rows, column_width=[180, 150, 110, 110, 100, 120]))
    blocks.append(make_heading("Claude 辅助判断", level=2))
    blocks.append(make_callout([make_text("分歧合理性分析", bold=True), make_text(claude_summary or "无 Claude 分析结果")], emoji_id="mag"))
    if sample_rows:
        blocks.append(make_heading("分歧样本", level=2))
        blocks.extend(make_split_tables(
            ["问题", "场景", "意图", "指标", f"{model_a['label']}分", f"{model_a['label']}原因", f"{model_b['label']}分", f"{model_b['label']}原因"],
            sample_rows,
            column_width=[260, 120, 120, 120, 80, 260, 80, 260],
        ))
    return blocks


def cleanup_local_generated_files(block_specs, verbose=False):
    """Delete local files referenced by generated image blocks if they still exist."""
    for spec in block_specs or []:
        if spec.get("kind") == "image":
            path = Path(spec.get("path", ""))
            try:
                if path.exists():
                    path.unlink()
                    if verbose:
                        print(f"  已删除本地生成内容: {path}")
            except Exception as e:
                print(f"  ⚠ 删除本地生成内容失败 {path}: {e}")
        for child in spec.get("children", []) or []:
            cleanup_local_generated_files([child], verbose=verbose)


def generate_consistency_report_file(token_mgr, app_token, table_id, model_configs, verbose=False, wiki_space_id=DEFAULT_WIKI_SPACE_ID, table_name=None):
    if len(model_configs) < 2:
        print("仅选择了一个模型，跳过一致性对比报告。")
        return None
    model_a, model_b = model_configs[0], model_configs[1]
    token = token_mgr.get()
    records = fetch_all_records(token, app_token, table_id, verbose)
    skill_files = load_skill_files()
    inconsistent_items = collect_inconsistent_records(records, model_a, model_b, limit=80, skill_files=skill_files)
    claude_summary = call_claude_for_consistency(inconsistent_items, verbose=verbose)
    blocks = generate_consistency_report_blocks(records, model_a, model_b, claude_summary)
    try:
        try:
            table_name = table_name or fetch_bitable_title(token, app_token, verbose=verbose)
        except Exception as e:
            table_name = table_id
            if verbose:
                print(f"  拉取 bitable 标题失败，回退为 table_id: {e}")
        report_title = f"{table_name}_一致性对比分析报告"
        node = create_wiki_report_doc(token, report_title, space_id=wiki_space_id, verbose=verbose)
        document_id = node.get("obj_token")
        if not document_id:
            raise RuntimeError(f"创建 wiki 节点成功但未返回 document_id/obj_token: {node}")
        upload_block_children(token, document_id, document_id, blocks, verbose=verbose)
        wiki_url = f"https://my.feishu.cn/wiki/{node['node_token']}"
        print(f"✓ 一致性对比报告已上传: {wiki_url}")
        return {"title": report_title, "node_token": node["node_token"], "document_id": document_id, "wiki_url": wiki_url}
    finally:
        cleanup_local_generated_files(blocks, verbose=verbose)


def main():
    parser = argparse.ArgumentParser(description="Knowledge Eval DM 双模型 QA 质量评估")
    parser.add_argument("--app-token", required=True, help="飞书 Bitable app_token")
    parser.add_argument("--table-id", required=True, help="飞书 Bitable table_id")
    parser.add_argument("--wiki-space-id", default=DEFAULT_WIKI_SPACE_ID, help="飞书知识库 space_id")
    parser.add_argument("--batch-size", type=int, default=20, help="每次批量写回飞书的记录数")
    parser.add_argument("--max-workers", type=int, default=4, help="并发评估请求数；并发单位为模型×记录")
    parser.add_argument("--models", default=",".join(DEFAULT_MODEL_KEYS), help="逗号分隔模型 key，默认 kimi,deepseek")
    parser.add_argument("--dry-run", action="store_true", help="只拉数据不评估不写回")
    parser.add_argument("--verbose", action="store_true", help="详细日志")
    parser.add_argument("--reset-progress", action="store_true", help="清除进度重新开始")
    parser.add_argument("--report-only", action="store_true", help="仅生成两份模型评估质量报告，不评估也不生成一致性报告")
    parser.add_argument("--consistency-only", action="store_true", help="仅生成一致性对比分析报告，不评估也不生成模型报告")
    parser.add_argument("--skip-consistency", action="store_true", help="跳过一致性对比报告")
    parser.add_argument("--only-missing", action="store_true", help="增量模式：只跳过所有选定模型均已评分的记录")
    parser.add_argument("--skip-ensure-fields", action="store_true", help="跳过自动建字段")
    args = parser.parse_args()

    model_keys = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    model_configs = resolve_model_configs(model_keys)

    print(f"{'='*50}")
    print(f"Knowledge Eval DM 双模型评估 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    model_desc = ", ".join(f"{m['label']}({m['model']})" for m in model_configs)
    print(f"模型: {model_desc}")
    print(f"{'='*50}")

    app_id = os.environ.get("FEISHU_APP_ID")
    app_secret = os.environ.get("FEISHU_APP_SECRET")
    maas_key = os.environ.get("MAAS_EVAL_API_KEY")
    for model_config in model_configs:
        model_config["api_key"] = os.environ.get(model_config.get("api_key_env", "")) or maas_key

    missing = []
    if not app_id: missing.append("FEISHU_APP_ID")
    if not app_secret: missing.append("FEISHU_APP_SECRET")
    if not args.report_only and not args.consistency_only:
        missing_model_keys = [m["label"] for m in model_configs if not m.get("api_key")]
        if missing_model_keys:
            missing.append("MAAS_EVAL_API_KEY or " + "/".join(m.get("api_key_env", "") for m in model_configs))
    if missing:
        print(f"✗ 缺少环境变量: {', '.join(missing)}")
        sys.exit(1)
    print("✓ 环境变量检查通过")

    print("获取飞书 token...")
    token_mgr = TokenManager(app_id, app_secret)
    token_mgr.get()
    print("✓ tenant_access_token 获取成功")

    try:
        table_name = fetch_bitable_title(token_mgr.get(), args.app_token, verbose=args.verbose)
    except Exception as e:
        table_name = args.table_id
        if args.verbose:
            print(f"  拉取 bitable 标题失败，回退为 table_id: {e}")

    if args.consistency_only:
        print("\n[仅生成一致性对比分析报告模式]")
        generate_consistency_report_file(token_mgr, args.app_token, args.table_id, model_configs, args.verbose, args.wiki_space_id, table_name=table_name)
        return

    if args.report_only:
        print("\n[仅生成模型评估质量报告模式]")
        for model_config in model_configs:
            generate_report_file(token_mgr, args.app_token, args.table_id, args.verbose, args.wiki_space_id, model_config=model_config, table_name=table_name)
        return

    if not args.skip_ensure_fields:
        print("检查评估输出字段...")
        created = ensure_eval_fields(token_mgr.get(), args.app_token, args.table_id, args.verbose, model_configs=model_configs)
        print(f"✓ 已创建 {created} 个缺失字段" if created else "✓ 字段齐全")

    mode_label = "未完整评估" if args.only_missing else "全量"
    print(f"拉取{mode_label}记录 (app={args.app_token}, table={args.table_id})...")
    records = fetch_records(token_mgr.get(), args.app_token, args.table_id, args.verbose, only_missing=args.only_missing, model_configs=model_configs)
    print(f"✓ 共 {len(records)} 条待评估")

    if args.reset_progress and PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        print("✓ 已清除进度文件")

    completed_ids = load_progress()
    tasks = []
    for rec in records:
        record_id = rec.get("record_id", "")
        fields = rec.get("fields", {})
        for model_config in model_configs:
            task_id = f"{model_config['key']}:{record_id}"
            if task_id in completed_ids:
                continue
            if args.only_missing and fields.get(eval_field_name("Faithfulness", "score", model_config)) is not None:
                continue
            tasks.append((model_config, rec, task_id))

    if not tasks:
        print("没有待评估任务，直接生成报告。")
        clear_progress()
        for model_config in model_configs:
            generate_report_file(token_mgr, args.app_token, args.table_id, args.verbose, args.wiki_space_id, model_config=model_config, table_name=table_name)
        if not args.skip_consistency:
            generate_consistency_report_file(token_mgr, args.app_token, args.table_id, model_configs, args.verbose, args.wiki_space_id, table_name=table_name)
        return

    print("加载评估准则...")
    skill_files = load_skill_files()
    print("✓ 评估准则加载完成")
    print(f"评估任务数: {len(tasks)} = 模型×记录；并发数: {args.max_workers}；写回批大小: {args.batch_size}")

    if args.dry_run:
        print("\n[Dry-run 模式] 跳过评估和写回")
        for idx, (model_config, rec, _) in enumerate(tasks, 1):
            print(f"  任务 {idx}: {model_config['label']} / {rec.get('record_id', '?')}")
        return

    succeeded = 0
    failed = 0
    failed_list = []
    pending_by_model = {cfg["key"]: [] for cfg in model_configs}
    config_by_key = {cfg["key"]: cfg for cfg in model_configs}
    write_idx = 0

    def flush_model_results(model_key, force=False):
        nonlocal succeeded, failed, write_idx
        pending = pending_by_model[model_key]
        model_config = config_by_key[model_key]
        while pending and (force or len(pending) >= args.batch_size):
            chunk = pending[:args.batch_size]
            del pending[:args.batch_size]
            write_idx += 1
            try:
                update_data = build_update_records(chunk, model_config=model_config)
                batch_update_records(token_mgr.get(), args.app_token, args.table_id, update_data, args.verbose)
                task_ids = {f"{model_key}:{item['record_id']}" for item in chunk}
                completed_ids.update(task_ids)
                save_progress(completed_ids)
                succeeded += len(chunk)
                print(f"  ✓ 写回批次 {write_idx}: {model_config['label']} {len(chunk)} 条")
            except Exception as e:
                failed += len(chunk)
                for item in chunk:
                    failed_list.append({"record_id": item["record_id"], "model": model_config["label"], "reason": f"写回失败: {str(e)[:160]}"})
                print(f"  ✗ 写回批次 {write_idx} 失败: {e}")

    print("\n开始双模型并行评估，并按模型分字段写回飞书...")
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        future_map = {
            executor.submit(evaluate_record, rec, skill_files, maas_key, model_config, args.verbose): (model_config, rec, task_id)
            for model_config, rec, task_id in tasks
        }
        for idx, future in enumerate(as_completed(future_map), 1):
            model_config, rec, task_id = future_map[future]
            record_id = rec.get("record_id", "?")
            try:
                result = future.result()
                pending_by_model[model_config["key"]].append(result)
                print(f"  ✓ {model_config['label']} 评估完成 {record_id} ({idx}/{len(tasks)})")
                flush_model_results(model_config["key"])
            except Exception as e:
                failed += 1
                failed_list.append({"record_id": record_id, "model": model_config["label"], "reason": str(e)[:200]})
                print(f"  ✗ {model_config['label']} 评估失败 {record_id}: {e}")

    for model_config in model_configs:
        if pending_by_model[model_config["key"]]:
            print(f"\n补写最后一批飞书: {model_config['label']}")
            flush_model_results(model_config["key"], force=True)

    print(f"\n{'='*50}")
    print("评估完成!")
    print(f"  成功任务: {succeeded}")
    print(f"  失败任务: {failed}")
    print(f"  总任务: {len(tasks)}")
    print(f"{'='*50}")

    if failed_list:
        save_failed(failed_list)
        print(f"失败记录已保存: {FAILED_FILE}")
    elif succeeded == len(tasks):
        clear_progress()

    print(f"\n{'='*50}")
    print("生成模型评估报告...")
    print(f"{'='*50}")
    for model_config in model_configs:
        generate_report_file(token_mgr, args.app_token, args.table_id, args.verbose, args.wiki_space_id, model_config=model_config, table_name=table_name)

    if not args.skip_consistency:
        print(f"\n{'='*50}")
        print("生成一致性对比报告...")
        print(f"{'='*50}")
        generate_consistency_report_file(token_mgr, args.app_token, args.table_id, model_configs, args.verbose, args.wiki_space_id, table_name=table_name)


if __name__ == "__main__":
    main()
