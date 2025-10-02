# Medical Bridge Production

医療LLM評価・開発プロジェクト

## プロジェクト概要

このリポジトリは医療分野に特化したLLM（大規模言語モデル）の評価および開発を目的としたプロジェクトです。

## ディレクトリ構成

```
medical_bridge_prod/
├── eval_mxqa/           # MedXpertQA評価フレームワーク
│   ├── README.md        # 詳細なドキュメント
│   ├── predict.py       # 推論実行スクリプト
│   ├── judge.py         # 評価実行スクリプト
│   └── conf/            # 設定ファイル
└── README.md            # このファイル
```

## 主要コンポーネント

### eval_mxqa
MedXpertQAベンチマークを使用した医療LLMの評価フレームワーク。

**主な機能:**
- OpenAI API、vLLM、Ollama対応
- LLM as Judge評価パイプライン
- カスタマイズ可能なプロンプト設定
- マルチモーダル（画像+テキスト）対応

詳細は [eval_mxqa/README.md](./eval_mxqa/README.md) を参照してください。

## クイックスタート

### 評価の実行

```bash
# eval_mxqaディレクトリに移動
cd eval_mxqa

# 環境構築
pip install -r requirements.txt

# 設定ファイル編集
vim conf/config.yaml

# 推論実行
python predict.py

# 評価実行
python judge.py
```

## 開発環境

- Python 3.12+
- CUDA 12.6+ (GPU使用時)
- vLLM 0.9.2+

## ライセンス

MIT License

## 関連リンク

- MedXpertQAデータセット: [TsinghuaC3I/MedXpertQA](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA)
- プロジェクトリポジトリ: [github.com/fregean/medical_bridge_prod](https://github.com/fregean/medical_bridge_prod)
