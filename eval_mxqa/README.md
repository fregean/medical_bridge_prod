# MedXpertQA Evaluation Framework

医療特化型QAベンチマーク「MedXpertQA」の評価フレームワークです。OpenAI API、vLLM、Ollamaに対応しています。

## 概要

このフレームワークは以下の機能を提供します：
- MedXpertQA/HLEデータセットでの推論実行
- LLM as Judgeによる自動評価
- マルチモーダル(画像+テキスト)および文字のみの問題に対応
- 複数の推論プロバイダー(OpenAI、vLLM、Ollama)のサポート

## 環境構築

### 基本セットアップ

```bash
# Conda環境作成
conda create -n eval_mxqa python=3.12 -y
conda activate eval_mxqa

# 依存パッケージインストール
pip install -r requirements.txt
```

### 環境変数設定

`.env`ファイルを作成し、必要なAPIキーを設定してください：

```bash
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here  # vLLM使用時のみ
```

## 設定ファイル

`conf/config.yaml`で評価の詳細設定を行います。

### 主要パラメータ

| パラメータ | 型 | 説明 |
|----------|------|------|
| `dataset` | string | 評価データセット (デフォルト: `TsinghuaC3I/MedXpertQA`) |
| `provider` | string | 推論プロバイダー (`openai`, `vllm`, `ollama`) |
| `base_url` | string | vLLM/Ollama使用時のエンドポイントURL (OpenAI APIの場合は`null`) |
| `model` | string | 使用モデル名 |
| `max_completion_tokens` | int | 最大出力トークン数 |
| `reasoning` | boolean | 推論モード有効化 (o3/Qwen3等) |
| `num_workers` | int | 並列リクエスト数 (OpenAI: ~30, vLLM: ~2500推奨) |
| `max_samples` | int | 処理する問題数 (先頭からN問) |
| `question_range` | list | 処理する問題の範囲 `[start, end]` |
| `judge` | string | 評価用モデル (`o3-mini-2025-01-31`推奨) |

### 設定例

#### OpenAI API使用時
```yaml
dataset: TsinghuaC3I/MedXpertQA
provider: openai
base_url: null
model: o3-mini
max_completion_tokens: 38000
reasoning: true
num_workers: 30
question_range: [0, 100]  # 最初の100問のみ
judge: o3-mini-2025-01-31
```

#### vLLM使用時
```yaml
dataset: TsinghuaC3I/MedXpertQA
provider: vllm
base_url: http://localhost:8000/v1
model: Qwen/Qwen3-32B
max_completion_tokens: 128000
reasoning: true
num_workers: 2500
max_samples: 2500
judge: o3-mini-2025-01-31
```

## 実行方法

### 1. 推論実行

```bash
python predict.py
```

推論結果は`predictions/`ディレクトリに保存されます。
- ファイル名形式: `{dataset_name}_{model_name}.json`
- 既存の結果がある場合、未実施の問題のみ処理されます

### 2. 評価実行

```bash
python judge.py
```

評価結果は以下に保存されます：
- `judged/`: 判定結果の詳細
- `leaderboard/`: 集計結果とサマリー
  - `results.jsonl`: 問題ごとの評価結果
  - `results.csv`: CSV形式の評価結果
  - `summary.json`: 評価サマリー

## vLLMサーバーセットアップ

### シングルノード実行

```bash
# vLLMサーバー起動 (8GPU例)
vllm serve Qwen/Qwen3-32B \
  --tensor-parallel-size 8 \
  --enable-reasoning \
  --reasoning-parser qwen3 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.95 &

# ヘルスチェック
until curl -s http://127.0.0.1:8000/health >/dev/null; do
  echo "vLLM starting..."
  sleep 5
done
echo "vLLM Ready"

# 推論実行
python predict.py
```

### マルチノード実行 (Ray Cluster)

1. Ray Clusterの起動

```bash
# jobs/ray_cluster.shを編集してpartition/nodelist設定
bash jobs/ray_cluster.sh
```

2. Headノードに接続してvLLM起動

```bash
# 出力されたheadノードにSSH接続
ssh <head_node>

# モジュール読み込み
module load cuda/12.6 miniconda/24.7.1-py312
conda activate eval_mxqa

# vLLM起動 (Ray Clusterを自動検出)
vllm serve <model_name> \
  --tensor-parallel-size <total_gpus> \
  ...
```

3. 推論実行

```bash
# config.yamlのbase_url/num_workersを調整
python predict.py
python judge.py
```

## SLURMバッチジョブ例

```bash
#!/bin/bash
#SBATCH --job-name=eval_mxqa
#SBATCH --partition=GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# 環境セットアップ
module load cuda/12.6 miniconda/24.7.1-py312
conda activate eval_mxqa

# Hugging Face設定
export HF_TOKEN="your_token"
export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
mkdir -p "$HF_HOME"

# vLLMサーバー起動
vllm serve Qwen/Qwen3-32B \
  --tensor-parallel-size 8 \
  --enable-reasoning \
  --reasoning-parser qwen3 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.95 &
vllm_pid=$!

# ヘルスチェック
until curl -s http://127.0.0.1:8000/health >/dev/null; do
  sleep 5
done

# 推論と評価
python predict.py
python judge.py

# クリーンアップ
kill $vllm_pid
wait
```

## 対応モデル

### 動作確認済み
- **OpenAI**: o3-mini, o1-mini, GPT-4o
- **vLLM対応モデル**:
  - Qwen/Qwen3-8B
  - Qwen/Qwen3-32B
  - その他vLLM対応モデル全般

### 推論モード対応
- OpenAI o3シリーズ: `reasoning: true`で推論思考プロセス利用
- Qwen3シリーズ: `--enable-reasoning --reasoning-parser qwen3`で推論モード

## データセット

### MedXpertQA
- 医療専門QAベンチマーク
- Text-onlyとMultimodal(画像付き)問題を含む
- 約2,400-2,500問

### Humanity's Last Exam (HLE)
- 一般知識ベンチマーク
- HLEデータセットにも対応

## トラブルシューティング

### 推論が途中で止まる場合
- 複数回実行してください。既に処理済みの問題はスキップされます
- `num_workers`を減らして並列度を下げてみてください

### メモリ不足エラー
- vLLMの`--gpu-memory-utilization`を0.9以下に設定
- `max_completion_tokens`を減らす
- `--max-model-len`を調整

### API制限エラー
- `num_workers`を減らす (OpenAI APIの場合は30以下推奨)
- レート制限に応じて調整

## コスト見積もり

### LLM as Judge (o3-mini使用時)
- 全問題評価(2,500問):
  - 入力: ~250K tokens
  - 出力: ~20K tokens
- 他のモデルではトークン数が異なる可能性があります

## ディレクトリ構造

```
eval_mxqa/
├── conf/
│   └── config.yaml          # 設定ファイル
├── hle_benchmark/           # コアロジック
│   ├── openai_predictions.py
│   ├── vllm_predictions.py
│   ├── ollama_predictions.py
│   └── run_judge_results.py
├── jobs/
│   └── ray_cluster.sh       # マルチノード用スクリプト
├── predictions/             # 推論結果
├── judged/                  # 評価結果
├── leaderboard/             # リーダーボード
├── outputs/                 # 実行ログ
├── predict.py               # 推論実行スクリプト
├── judge.py                 # 評価実行スクリプト
└── requirements.txt         # 依存パッケージ
```

## ライセンス

MIT License (詳細はソースコード参照)

## 謝辞

- 元のHLE評価コード: centerforaisafety
- MedXpertQAデータセット: TsinghuaC3I
