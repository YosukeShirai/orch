# orch

Goal-driven task orchestrator powered by Claude Code.

ゴールを与えると、Claude Code がタスクを自動分解し、依存関係を考慮しながら並列実行します。

## Install

```bash
pip install -e ".[dev]"
```

Requires:
- Python 3.11+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (`claude` command)

## Usage

```bash
# 新規プロジェクト開始（並列度2、チェックポイントあり）
orch run "Webスクレイパーを作成して"

# 並列度を指定
orch run "3つの独立したスクリプトを作成して" -c 3

# 自動モード（承認・チェックポイントなし）
orch run "テストを書いて" --auto -c 4

# 中断したプロジェクトを再開
orch resume

# プロジェクト状態の確認
orch status
orch status <project-id>
```

### Options

| Option | Short | Default | Description |
|---|---|---|---|
| `--auto` | `-a` | `false` | 承認・チェックポイントをスキップ |
| `--concurrency` | `-c` | `2` | 最大並列タスク数 |

## How it works

```
Goal → DAG生成 → [承認] → 並列実行 → 完了
                              ↑
                         チェックポイント
                        (pause/resume可)
```

1. **DAG生成**: Claude Code がゴールを3-8個のタスクに分解し、依存関係グラフ(DAG)を構築
2. **承認**: タスクプランを確認し、承認/再生成/キャンセルを選択
3. **並列実行**: 依存関係が解決済みのタスクを最大N個同時に実行（バッチモデル）
4. **チェックポイント**: 各バッチ完了後に続行/一時停止/中止を選択（supervisedモード）
5. **状態管理**: SQLiteに全状態を永続化。中断後も `orch resume` で再開可能

## Architecture

```
src/orch/
├── cli.py                 # Typer CLI
├── agents/
│   ├── base.py            # BaseAgent, AgentResult, enums
│   └── claude_code.py     # Claude Code subprocess agent
├── core/
│   ├── dag.py             # NetworkX DAG engine
│   ├── executor.py        # Orchestrator (sequential/parallel)
│   ├── scheduler.py       # Task scheduler
│   ├── state.py           # SQLite state manager
│   └── monitor.py         # Event logging
└── planner/
    └── dag_generator.py   # Goal → DAG conversion
```

## Development

```bash
# テスト実行
python -m pytest tests/ -v

# 並列実行テストのみ
python -m pytest tests/test_executor.py -v -k "parallel"
```

## License

MIT
