#!/usr/bin/env python3
"""
Knowledge Eval DM - 便捷驱动脚本
简化评估任务的调用，自动解析 Bitable 链接，提供友好的命令行接口
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent.parent
SCRIPT_PATH = SKILL_DIR / "scripts" / "knowledge_eval.py"

def parse_bitable_url(url):
    """解析飞书 Bitable 链接，提取 app_token 和 table_id"""
    # 支持格式: https://xxx.feishu.cn/base/<app_token>?table=<table_id>
    match = re.search(r'/base/([^/\?]+).*[?&]table=([^&]+)', url)
    if match:
        return match.group(1), match.group(2)
    return None, None

def check_env_vars(required_vars):
    """检查必需的环境变量"""
    missing = []
    for var in required_vars:
        if not os.environ.get(var):
            missing.append(var)
    return missing

def main():
    parser = argparse.ArgumentParser(
        description='Knowledge Eval DM - 双模型评估驱动脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 完整评估（从 Bitable 链接）
  python driver.py https://xxx.feishu.cn/base/xxx?table=xxx

  # 完整评估（直接提供 token）
  python driver.py --app-token bascnxxx --table-id tblxxx

  # 增量评估（只跑缺失的）
  python driver.py <URL> --incremental

  # 只重新生成报告
  python driver.py <URL> --wiki-space-id 7xxx --report-only

  # 只重新生成一致性报告
  python driver.py <URL> --wiki-space-id 7xxx --consistency-only
""")

    parser.add_argument('bitable_url', nargs='?',
                        help='飞书 Bitable 链接（可选，如果提供 --app-token）')
    parser.add_argument('--app-token', help='飞书应用 token')
    parser.add_argument('--table-id', help='飞书表格 ID')
    parser.add_argument('--wiki-space-id', help='飞书知识库空间 ID（生成报告时需要）')
    parser.add_argument('--max-workers', type=int, default=4, help='最大并发数（默认 4）')
    parser.add_argument('--batch-size', type=int, default=20, help='批量写回大小（默认 20）')
    parser.add_argument('--models', help='选择评估模型，逗号分隔（默认 kimi,deepseek）')

    # 模式选项
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--incremental', action='store_true',
                            help='增量模式：只评估缺失的记录')
    mode_group.add_argument('--report-only', action='store_true',
                            help='只重新生成两份模型评估质量报告')
    mode_group.add_argument('--consistency-only', action='store_true',
                            help='只重新生成一致性对比分析报告')

    # 其他选项
    parser.add_argument('--skip-ensure-fields', action='store_true',
                        help='跳过自动建字段')
    parser.add_argument('--skip-consistency', action='store_true',
                        help='跳过生成一致性对比报告')
    parser.add_argument('--reset-progress', action='store_true',
                        help='清除断点续跑的进度文件')
    parser.add_argument('--verbose', action='store_true',
                        help='显示详细日志')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅显示将要执行的命令，不实际运行')

    args = parser.parse_args()

    # 解析 Bitable 链接
    app_token = args.app_token
    table_id = args.table_id

    if args.bitable_url:
        parsed_app, parsed_table = parse_bitable_url(args.bitable_url)
        if parsed_app and parsed_table:
            app_token = app_token or parsed_app
            table_id = table_id or parsed_table
            print(f"[OK] Parsed Bitable URL: app_token={app_token}, table_id={table_id}")
        else:
            print("[ERROR] Cannot parse Bitable URL format", file=sys.stderr)
            print("  Expected: https://xxx.feishu.cn/base/<app_token>?table=<table_id>", file=sys.stderr)
            return 1

    # 检查必需参数
    if not app_token or not table_id:
        print("[ERROR] Must provide Bitable URL or both --app-token and --table-id", file=sys.stderr)
        parser.print_help()
        return 1

    # 检查环境变量
    required_vars = ['FEISHU_APP_ID', 'FEISHU_APP_SECRET', 'MAAS_EVAL_API_KEY']
    if not (args.report_only or args.consistency_only or args.skip_consistency):
        required_vars.append('ANTHROPIC_AUTH_TOKEN')

    missing_vars = check_env_vars(required_vars)
    if missing_vars:
        print(f"[ERROR] Missing required environment variables: {', '.join(missing_vars)}", file=sys.stderr)
        print("\nPlease set these environment variables:", file=sys.stderr)
        for var in missing_vars:
            print(f"  $env:{var} = 'your-value'", file=sys.stderr)
        return 1

    # 构建命令
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        '--app-token', app_token,
        '--table-id', table_id,
        '--max-workers', str(args.max_workers),
        '--batch-size', str(args.batch_size),
    ]

    if args.wiki_space_id:
        cmd.extend(['--wiki-space-id', args.wiki_space_id])

    if args.models:
        cmd.extend(['--models', args.models])

    if args.incremental:
        cmd.append('--only-missing')

    if args.report_only:
        cmd.append('--report-only')

    if args.consistency_only:
        cmd.append('--consistency-only')

    if args.skip_ensure_fields:
        cmd.append('--skip-ensure-fields')

    if args.skip_consistency:
        cmd.append('--skip-consistency')

    if args.reset_progress:
        cmd.append('--reset-progress')

    if args.verbose:
        cmd.append('--verbose')

    # 显示将要执行的命令
    print("\n" + "="*60)
    if args.incremental:
        print("模式: 增量评估（只跑缺失记录）")
    elif args.report_only:
        print("模式: 重新生成模型评估质量报告")
    elif args.consistency_only:
        print("模式: 重新生成一致性对比分析报告")
    else:
        print("模式: 完整评估（双模型并行）")
    print("="*60)
    print(f"命令: {' '.join(cmd)}\n")

    if args.dry_run:
        print("(--dry-run 模式，不实际执行)")
        return 0

    # 执行命令
    try:
        result = subprocess.run(cmd, cwd=SKILL_DIR)
        return result.returncode
    except KeyboardInterrupt:
        print("\n\n✗ 用户中断", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\n✗ 执行失败: {e}", file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())
