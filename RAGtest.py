#!/usr/bin/env python3
# ============================================================
# 单条 query 测试命令（不走本地 CSV，直接打印结果）
# ------------------------------------------------------------
# DeepSeek V4 Pro:
#   python3 "/Users/snowmeral/Downloads/测试脚本/RAG知识库API调用_知识库.py" \
#     --query "你的问题" --llm-provider deepseek --deepseek-variant pro
#
# DeepSeek V4 Flash:
#   python3 "/Users/snowmeral/Downloads/测试脚本/RAG知识库API调用_知识库.py" \
#     --query "你的问题" --llm-provider deepseek --deepseek-variant flash
#
# 讯飞 MaaS DeepSeek:
#   python3 "/Users/snowmeral/Downloads/测试脚本/RAG知识库API调用_知识库.py" \
#     --query "你的问题" --llm-provider xfyun
# ============================================================
import asyncio
import base64
import csv
import hashlib
import hmac
import json
import re
import ssl
import time
import urllib.request
from email.utils import formatdate
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urlparse, urlunparse

# 知识库检索配置
RAG_API_BASE = "https://knowledge-retrieval.cn-huabei-1.xf-yun.com/v2/aiui/cbm/chunk/query"
RAG_APPID = "a3afa08b"  # 在飞云开放平台创建的 appid
RAG_API_KEY = "2dfd59e2b9b6209b8766842a9e7e99cf"
RAG_API_SECRET = "YmI0ZDViNzI2ZjdmODNmYjkzZDI5N2E5"

# 检索参数
# # 医疗
RAG_REPO_ID = "insight_qwen06b_100_msw86"  # 知识库名称（repoId）
RAG_GROUP_ID = "group_5c75ccf4855bab3dbfa260d8"  # 文档分组ID
RAG_TOP_N = 5
RAG_TOP_K = 20
RAG_RETRIEVAL_METHOD = "hybrid"  # hybrid | vector | keywords
RAG_RERANK_METHOD = "qwen3-reranker-0.6b"  # spark | gte | search | qwen3-reranker-0.6b | skip

# # 金融
# RAG_REPO_ID = "insight_qwen06b_100_msw86"  # 知识库名称（repoId）
# RAG_GROUP_ID = "group_7d7631f6dcecd114d7feba1c"  # 文档分组ID
# RAG_TOP_N = 5
# RAG_TOP_K = 20
# RAG_RETRIEVAL_METHOD = "hybrid"  # hybrid | vector | keywords
# RAG_RERANK_METHOD = ("qwen3-reranker-0.6b")  # spark | gte | search | qwen3-reranker-0.6b | skip

# # 教育
# RAG_REPO_ID = "insight_qwen06b_100_msw86"  # 知识库名称（repoId）
# RAG_GROUP_ID = "group_b5a97a3397ada7583ec3bdb6" # 文档分组ID
# RAG_TOP_N = 5
# RAG_TOP_K = 20
# RAG_RETRIEVAL_METHOD = "hybrid"  # hybrid | vector | keywords
# RAG_RERANK_METHOD = "qwen3-reranker-0.6b"  # spark | gte | search | qwen3-reranker-0.6b | skip

# # 工业
# RAG_REPO_ID = "insight_qwen06b_100_msw86"  # 知识库名称（repoId）
# RAG_GROUP_ID = "group_a0909d136f04165eacc9cecf"  # 文档分组ID
# RAG_TOP_N = 5
# RAG_TOP_K = 20
# RAG_RETRIEVAL_METHOD = "hybrid"  # hybrid | vector | keywords
# RAG_RERANK_METHOD = "qwen3-reranker-0.6b"  # spark | gte | search | qwen3-reranker-0.6b | skip

# # 法律
# RAG_REPO_ID = "insight_qwen06b_100_msw86"  # 知识库名称（repoId）
# RAG_GROUP_ID = "group_7d7631f6dcecd114d7feba1c"  # 文档分组ID
# RAG_TOP_N = 5
# RAG_TOP_K = 20
# RAG_RETRIEVAL_METHOD = "hybrid"  # hybrid | vector | keywords
# RAG_RERANK_METHOD = "qwen3-reranker-0.6b"  # spark | gte | search | qwen3-reranker-0.6b | skip

# 综合
# RAG_REPO_ID = "insight_qwen06b_100_msw86"  # 知识库名称（repoId）
# RAG_GROUP_ID = "group_cd228ee59df8c254eb67c8ed"  # 文档分组ID
# RAG_TOP_N = 5
# RAG_TOP_K = 20
# RAG_RETRIEVAL_METHOD = "hybrid"  # hybrid | vector | keywords
# RAG_RERANK_METHOD = "qwen3-reranker-0.6b"  # spark | gte | search | qwen3-reranker-0.6b | skip


# 大模型配置
# 通过 LLM_PROVIDER 切换 provider：
#   "xfyun"    -> 讯飞 MaaS DeepSeek（原配置）
#   "deepseek" -> DeepSeek 官方 API（https://api.deepseek.com）
LLM_PROVIDER = "deepseek"

# 讯飞 MaaS DeepSeek
XFYUN_LLM_API_BASE = "https://maas-api.cn-huabei-1.xf-yun.com/v2/chat/completions"
XFYUN_LLM_API_KEY = "6dd430477e49e5429833066648cb0983"
XFYUN_LLM_API_SECRET = "NjE4OWE1NWY5ZWQ0MWJlOTUzNGJjNzYw"
XFYUN_LLM_MODEL = "xopdsv32in"

# DeepSeek 官方 API（OpenAI 兼容格式，Bearer Token 鉴权）
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = "sk-17dad74e619a45bebd5a1cb52dc7a0c1"  # 请填写申请的 sk- 开头的 API Key
# DeepSeek V4 模型变体（"flash" / "pro"）
DEEPSEEK_MODEL_VARIANT = "flash"
DEEPSEEK_MODEL_MAP = {
    "flash": "deepseek-v4-flash",
    "pro": "deepseek-v4-pro",
}

STREAM_TIMEOUT_MS = 300000

# 默认输入/输出
DEFAULT_INPUT = "D:/Test/Basic/医疗.csv" # 本地文件地址
DEFAULT_OUTPUT = "D:/Test/Basic/医疗_output.csv" # 输出 CSV 地址
DEFAULT_RETRIEVAL_CONCURRENCY = 1
DEFAULT_LLM_CONCURRENCY = 1
OUTPUT_FIELDNAMES = [
    "query",
    "answer",
    "chunks",
    "retrieval_ms",
    "retrieval_sid",
]

SYSTEM_PROMPT_TEMPLATE = """你是一个专业的知识问答助手，负责理解用户真实需求，基于所提供的参考资料准确回答用户问题。
**目标**：让用户快速获得所需信息，减少误解、追问与人工介入成本。

## 核心原则
1. **忠于原文**
   - 仅使用参考资料中明确写出的内容，严禁编造、推测或补全隐含信息
   - 优先使用资料原文作答，保证要点完整；关键结论、举措和细则尽量保留原文表述
   - 不同主体、条件、范围的信息禁止混淆；仅当答案分散在多处时，才进行必要归纳
   - 不得将仅适用于特定对象/场景的内容套用到未覆盖的情况

2. **精准溯源**
   - 仅使用参考资料中实际存在的编号（id 字段），**严禁编造或引用不存在的编号**
   - 在事实性陈述句末标注 [^num]，**禁止在回复末尾统一列出**
   - 相邻句或列表项引用同一资料，**不要重复标注，仅保留最后一处**
        - 例：✅ `A。B。C [^2]`　　❌ `A [^2]。B [^2]。C [^2]`
   - 多条资料支撑同一陈述时并列标注 [^num1][^num2]
   - 来源编号必须与召回内容严格对应，禁止错配

3. **范围匹配**
   - 精准匹配用户指定的所有条件（时间/地点/人员范围/场景）
   - 严格围绕问题核心关注点回答，必须忠于问题，不扩展无关背景或泛化内容
   
4. **要点完整**
   - 覆盖参考资料中与问题直接相关的核心结论、条件、范围、限制、举措和必要补充信息
   - 当参考资料连续表达多个相关要点时，应完整识别，不得只截取单个结论或局部信息作答
   - 若参考资料只覆盖部分要点，应基于已有内容回答，不得补全资料未提供的细节

5. **逻辑清晰**
   - **开头给结论**：先用一两句直接回应问题，再展开支撑要点，不要从背景或资料片段开始
   - **按问题类型组织正文**：
     - 并列型（多个独立要点）：分条列出
     - 流程型（如何做、操作步骤）：按步骤编号，必要时补充路径或前置条件，不得编造未说明的细节
     - 因果型（为什么、影响）：按"原因 → 结果"或"条件 → 影响"组织
 
### 表达要求
- 客观、准确、中立，不情绪化、不说教
- 优先使用列表、步骤、表格等结构化形式
- 避免模糊词（大概/可能/看情况）和暗示性承诺（通常都会通过/一般没问题）
- 涉及 URL 或访问地址时，统一使用 Markdown 自动链接格式 `<URL>`
- 不要提及参考资料存在（禁止出现"根据资料"等措辞），禁止输出提示语或相关解释性文字

## 边界处理
- 资料冲突时优先级：发布时间更新 > 标注"以本文件为准" > 专项细则 > 总则
- 召回内容不支撑结论时，禁止生成无依据答案
"""

USER_PROMPT_TEMPLATE = """\
请基于以下参考资料，筛选出与用户问题最相关的知识点，简明清晰地直接回答用户问题。

## 相关片段
如下数组中 "id" 为参考资料序号用于溯源，"context" 为资料内容。
溯源标注 [^num] 仅限使用”id”值，正文中出现的其他数字不是溯源编号。

{knowledge}

## 用户问题

{query}
"""


def build_signed_url(
    api_base: str,
    api_key: str,
    api_secret: str,
    method: str = "GET",
    use_websocket_scheme: bool = False,
) -> str:
    """生成带鉴权参数的 URL。"""
    parsed = urlparse(api_base)
    host, path = parsed.hostname, parsed.path
    date = formatdate(usegmt=True)
    sign = f"host: {host}\ndate: {date}\n{method.upper()} {path} HTTP/1.1"
    signature = hmac.new(api_secret.encode(), sign.encode(), hashlib.sha256).digest()
    sig_b64 = base64.b64encode(signature).decode()
    auth = (
        f'api_key="{api_key}", algorithm="hmac-sha256", '
        f'headers="host date request-line", signature="{sig_b64}"'
    )
    auth_b64 = base64.b64encode(auth.encode()).decode()
    qs = urlencode({"authorization": auth_b64, "host": host, "date": date})
    scheme = parsed.scheme
    if use_websocket_scheme:
        if scheme == "https":
            scheme = "wss"
        elif scheme == "http":
            scheme = "ws"
    return urlunparse((scheme, parsed.netloc, parsed.path, "", qs, ""))


def build_llm_references(retrieval_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    references: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(retrieval_chunks, start=1):
        references.append(
            {
                # 模型侧编号与输出 chunks 保持一致，统一从 1 开始。
                "id": chunk.get("id", idx),
                "context": chunk.get("context") or "",
                "documentName": chunk.get("documentName") or "",
            }
        )
    return references


def build_llm_user_prompt(query_text: str, retrieval_chunks: List[dict]) -> str:
    references = build_llm_references(retrieval_chunks)
    return USER_PROMPT_TEMPLATE.format(
        knowledge=json.dumps(references, ensure_ascii=False, indent=2),
        query=query_text,
        chunk_count=len(references),
    )


def build_ssl_context(url: str) -> Optional[ssl.SSLContext]:
    if urlparse(url).scheme not in {"https", "wss"}:
        return None
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def extract_reference_content(reference_item: Any) -> str:
    if isinstance(reference_item, dict):
        content = reference_item.get("content")
        if content not in (None, ""):
            return str(content)
        return json.dumps(reference_item, ensure_ascii=False)
    if reference_item in (None, ""):
        return ""
    return str(reference_item)


def resolve_context_references(context_text: str, references: Dict[str, Any]) -> str:
    if not context_text or not references:
        return context_text

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in references:
            return match.group(0)
        replacement = extract_reference_content(references.get(key))
        return replacement if replacement else match.group(0)

    return re.sub(r"<(unused\d+)>", replace, context_text)


def extract_response_sid(data: Dict[str, Any]) -> str:
    candidates = (
        data.get("sid"),
        (data.get("header") or {}).get("sid"),
        (data.get("message") or {}).get("sid"),
        (data.get("data") or {}).get("sid"),
    )
    for candidate in candidates:
        if candidate is not None and candidate != "":
            return str(candidate)
    return ""


async def post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers=headers,
    )
    ssl_context = build_ssl_context(url)

    def do_request() -> dict:
        with urllib.request.urlopen(
            req,
            timeout=STREAM_TIMEOUT_MS / 1000,
            context=ssl_context,
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))

    return await asyncio.to_thread(do_request)


async def ask_retrieval(query_text: str) -> Dict[str, Any]:
    url = build_signed_url(
        RAG_API_BASE,
        RAG_API_KEY,
        RAG_API_SECRET,
        method="POST",
        use_websocket_scheme=False,
    )
    payload = {
        "query": query_text,
        "topN": RAG_TOP_N,
        "topK": RAG_TOP_K,
        "retrievalMethod": RAG_RETRIEVAL_METHOD,
        "reRankMethod": RAG_RERANK_METHOD,
        "match": {
            "groups": [RAG_GROUP_ID],
        },
        "repoSources": [
            {"repoId": RAG_REPO_ID},
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "X-Consumer-Username": RAG_APPID,
    }
    try:
        data = await post_json(url, payload, headers)
    except Exception as e:
        raise RuntimeError(f"检索接口调用失败: {e}")

    retrieval_sid = extract_response_sid(data)
    code = data.get("message", {}).get("code")
    if code and code != 0:
        raise RuntimeError(f"检索接口调用失败: code={code}, raw={data}")

    retrieval_chunks: List[dict] = []
    for idx, item in enumerate(data.get("data", {}).get("results") or [], start=1):
        doc_info: Dict[str, str] = item.get("docInfo") or {}
        item_references: Dict[str, Any] = item.get("references") or {}
        context_text = item.get("context") or item.get("content") or ""
        retrieval_chunks.append(
            {
                "id": idx,
                "score": item.get("score"),
                "context": resolve_context_references(context_text, item_references),
                "documentId": doc_info.get("documentId") or item.get("docId"),
                "documentName": doc_info.get("documentName"),
            }
        )
    return {
        "chunks": retrieval_chunks,
        "sid": retrieval_sid,
    }


def resolve_llm_endpoint() -> Dict[str, Any]:
    """根据 LLM_PROVIDER 返回当前要使用的接口 URL、模型名和请求头。"""
    if LLM_PROVIDER == "xfyun":
        return {
            "url": XFYUN_LLM_API_BASE,
            "model": XFYUN_LLM_MODEL,
            "headers": {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {XFYUN_LLM_API_KEY}:{XFYUN_LLM_API_SECRET}",
            },
        }
    if LLM_PROVIDER == "deepseek":
        if DEEPSEEK_MODEL_VARIANT not in DEEPSEEK_MODEL_MAP:
            raise RuntimeError(
                f"未知的 DEEPSEEK_MODEL_VARIANT: {DEEPSEEK_MODEL_VARIANT}, "
                f"可选: {list(DEEPSEEK_MODEL_MAP.keys())}"
            )
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY 未配置，请填写官方控制台申请的 sk- 开头 Key")
        return {
            "url": DEEPSEEK_API_BASE,
            "model": DEEPSEEK_MODEL_MAP[DEEPSEEK_MODEL_VARIANT],
            "headers": {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            },
        }
    raise RuntimeError(f"未知的 LLM_PROVIDER: {LLM_PROVIDER}")


async def ask_llm(query_text: str, retrieval_chunks: List[dict]) -> str:
    user_prompt = build_llm_user_prompt(query_text, retrieval_chunks)
    endpoint = resolve_llm_endpoint()
    payload = {
        "model": endpoint["model"],
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_TEMPLATE.format(today=time.strftime("%Y-%m-%d")),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "temperature": 0,
        "max_tokens": 4096,
    }
    try:
        data = await post_json(endpoint["url"], payload, endpoint["headers"])
    except Exception as e:
        raise RuntimeError(f"大模型调用失败: {e}")

    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    if not content:
        raise RuntimeError(f"大模型调用失败: 空响应, raw={data}")
    return content


async def retrieval_worker(
    jobs: "asyncio.Queue[Optional[Dict[str, Any]]]",
    llm_jobs: "asyncio.Queue[Optional[Dict[str, Any]]]",
) -> None:
    while True:
        job = await jobs.get()
        if job is None:
            jobs.task_done()
            return

        retrieval_chunks: List[dict] = []
        retrieval_sid = ""
        retrieval_error: Optional[str] = None
        retrieval_ms = 0
        retrieval_started_at = time.perf_counter()
        try:
            retrieval_result = await ask_retrieval(job["query"])
            retrieval_chunks = retrieval_result["chunks"]
            retrieval_sid = retrieval_result["sid"]
        except Exception as e:
            retrieval_error = str(e)
        finally:
            retrieval_ms = round((time.perf_counter() - retrieval_started_at) * 1000)
        if retrieval_error is None and not retrieval_chunks:
            retrieval_error = "检索无召回结果"

        await llm_jobs.put(
            {
                "idx": job["idx"],
                "query": job["query"],
                "retrieval_chunks": retrieval_chunks,
                "retrieval_error": retrieval_error,
                "retrieval_ms": retrieval_ms,
                "retrieval_sid": retrieval_sid,
            }
        )
        jobs.task_done()


async def llm_worker(
    llm_jobs: "asyncio.Queue[Optional[Dict[str, Any]]]",
    results: "asyncio.Queue[Dict[str, Any]]",
) -> None:
    while True:
        job = await llm_jobs.get()
        if job is None:
            llm_jobs.task_done()
            return

        answer = job["retrieval_error"] or ""
        if not job["retrieval_error"]:
            try:
                answer = await ask_llm(job["query"], job["retrieval_chunks"])
            except Exception as e:
                answer = str(e)

        await results.put(
            {
                "idx": job["idx"],
                "query": job["query"],
                "answer": answer,
                "chunks": json.dumps(job["retrieval_chunks"], ensure_ascii=False),
                "retrieval_ms": job["retrieval_ms"],
                "retrieval_sid": job["retrieval_sid"],
            }
        )
        llm_jobs.task_done()


async def write_results(
    results: "asyncio.Queue[Dict[str, Any]]",
    outp: str,
    total: int,
) -> None:
    pending: Dict[int, Dict[str, Any]] = {}
    next_idx = 0
    written = 0

    with open(outp, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        f.flush()

        while written < total:
            row = await results.get()
            pending[row["idx"]] = row

            while next_idx in pending:
                current = pending.pop(next_idx)
                writer.writerow(
                    {
                        "query": current["query"],
                        "answer": current["answer"],
                        "chunks": current["chunks"],
                        "retrieval_ms": current["retrieval_ms"],
                        "retrieval_sid": current["retrieval_sid"],
                    }
                )
                f.flush()
                written += 1
                next_idx += 1
                print(f"completed: {written}/{total}")

            results.task_done()


async def run_batch(
    inp: str,
    outp: str,
    limit: Optional[int] = None,
    retrieval_concurrency: int = DEFAULT_RETRIEVAL_CONCURRENCY,
    llm_concurrency: int = DEFAULT_LLM_CONCURRENCY,
) -> None:
    reader = None
    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with open(inp, newline="", encoding=encoding, errors="strict") as f:
                candidate = csv.DictReader(f)
                if "query" not in (candidate.fieldnames or []):
                    raise ValueError("输入 CSV 需要包含列名 query")
                source_rows = list(candidate)
            reader = source_rows
            break
        except Exception as e:
            last_error = e
    if reader is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("读取输入文件失败，且未捕获到具体异常")

    source_rows = reader[:limit] if limit is not None else reader
    total = len(source_rows)
    if total == 0:
        with open(outp, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
            writer.writeheader()
        return

    retrieval_jobs: "asyncio.Queue[Optional[Dict[str, Any]]]" = asyncio.Queue()
    llm_jobs: "asyncio.Queue[Optional[Dict[str, Any]]]" = asyncio.Queue()
    results: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()

    retrieval_workers = [
        asyncio.create_task(retrieval_worker(retrieval_jobs, llm_jobs))
        for _ in range(retrieval_concurrency)
    ]
    llm_workers = [
        asyncio.create_task(llm_worker(llm_jobs, results))
        for _ in range(llm_concurrency)
    ]
    writer_task = asyncio.create_task(write_results(results, outp, total))

    for idx, row in enumerate(source_rows):
        print(f"queued: {idx + 1}/{total} {row['query']}")
        await retrieval_jobs.put({"idx": idx, "query": row["query"]})

    for _ in range(retrieval_concurrency):
        await retrieval_jobs.put(None)

    await retrieval_jobs.join()
    await asyncio.gather(*retrieval_workers)

    for _ in range(llm_concurrency):
        await llm_jobs.put(None)

    await llm_jobs.join()
    await asyncio.gather(*llm_workers)

    await results.join()
    await writer_task


async def run_single_query(query_text: str, outp: Optional[str] = None) -> Dict[str, Any]:
    retrieval_chunks: List[dict] = []
    retrieval_sid = ""
    retrieval_error: Optional[str] = None

    retrieval_started_at = time.perf_counter()
    try:
        retrieval_result = await ask_retrieval(query_text)
        retrieval_chunks = retrieval_result["chunks"]
        retrieval_sid = retrieval_result["sid"]
    except Exception as e:
        retrieval_error = str(e)
    retrieval_ms = round((time.perf_counter() - retrieval_started_at) * 1000)

    if retrieval_error is None and not retrieval_chunks:
        retrieval_error = "检索无召回结果"

    answer = retrieval_error or ""
    if not retrieval_error:
        try:
            answer = await ask_llm(query_text, retrieval_chunks)
        except Exception as e:
            answer = str(e)

    row = {
        "query": query_text,
        "answer": answer,
        "chunks": json.dumps(retrieval_chunks, ensure_ascii=False),
        "retrieval_ms": retrieval_ms,
        "retrieval_sid": retrieval_sid,
    }

    if outp:
        with open(outp, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
            writer.writeheader()
            writer.writerow(row)

    return row


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("output_csv", nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 条记录")
    parser.add_argument(
        "--retrieval-concurrency",
        type=int,
        default=DEFAULT_RETRIEVAL_CONCURRENCY,
        help="检索阶段并发数",
    )
    parser.add_argument(
        "--llm-concurrency",
        type=int,
        default=DEFAULT_LLM_CONCURRENCY,
        help="大模型阶段并发数",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="直接传入单条 query；传入后走单条模式，不读取 input_csv",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["xfyun", "deepseek"],
        default=None,
        help="覆盖 LLM_PROVIDER：xfyun=讯飞 MaaS，deepseek=DeepSeek 官方",
    )
    parser.add_argument(
        "--deepseek-variant",
        choices=list(DEEPSEEK_MODEL_MAP.keys()),
        default=None,
        help="DeepSeek 模型变体：flash 或 pro",
    )
    args = parser.parse_args()

    if args.llm_provider:
        global LLM_PROVIDER
        LLM_PROVIDER = args.llm_provider
    if args.deepseek_variant:
        global DEEPSEEK_MODEL_VARIANT
        DEEPSEEK_MODEL_VARIANT = args.deepseek_variant

    if args.query:
        outp = None
        if args.output_csv != DEFAULT_OUTPUT:
            output_ext = os.path.splitext(args.output_csv)[1].lower()
            if output_ext and output_ext != ".csv":
                parser.error("当前脚本输出格式为 CSV，请将输出文件扩展名设为 .csv")
            outp = args.output_csv

        row = asyncio.run(run_single_query(args.query, outp))
        print(row["answer"])
        print(f"\n===== 检索耗时: {row['retrieval_ms']}ms  SID: {row['retrieval_sid']} =====")
        chunks = json.loads(row["chunks"])
        for c in chunks:
            print(f"  [{c['id']}] score={c['score']}  doc={c.get('documentName','')}")
        return

    output_ext = os.path.splitext(args.output_csv)[1].lower()
    if output_ext and output_ext != ".csv":
        parser.error("当前脚本输出格式为 CSV，请将输出文件扩展名设为 .csv")

    asyncio.run(
        run_batch(
            args.input_csv,
            args.output_csv,
            args.limit,
            args.retrieval_concurrency,
            args.llm_concurrency,
        )
    )


if __name__ == "__main__":
    main()
