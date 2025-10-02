#!/bin/bash
#
# サーバーから推論結果をダウンロードするスクリプト。
# 実行前に、ローカルの既存の結果をタイムスタンプ付きでバックアップする。
#

set -euo pipefail
# --- 設定 ---
REMOTE_USER="P10U029"
REMOTE_HOST="10.255.255.101"
SSH_KEY_PATH="$HOME/.ssh/id_ed25519"

# サーバー上の結果ディレクトリのパス（フォルダごとダウンロード）
REMOTE_PREDICTIONS_PATH="~/medical_bridge_prod/eval_mxqa/predictions"
REMOTE_LEADERBOARD_PATH="~/medical_bridge_prod/eval_mxqa/leaderboard"
REMOTE_JUDGED_PATH="~/medical_bridge_prod/eval_mxqa/judged"

# --- メイン処理 ---
echo "準備中：バックアップと一時ディレクトリを作成します..."

# 永続的なバックアップ先
BACKUP_ROOT_DIR="./backup/"
BACKUP_LEAD_DIR="${BACKUP_ROOT_DIR}leaderboard/"
mkdir -p "$BACKUP_LEAD_DIR"

# ダウンロード用の一時ディレクトリを準備（実行のたびにクリーンな状態にする）
LOCAL_TEMP_DIR="./eval_mxqa_results_latest/"
rm -rf "$LOCAL_TEMP_DIR"
mkdir -p "$LOCAL_TEMP_DIR"

echo ""
echo "サーバーから最新の結果を一時ディレクトリに同期します..."

# サーバーから一時ディレクトリへ全結果をダウンロード
RSYNC_OPTS="-avz --update"
rsync $RSYNC_OPTS -e "ssh -i $SSH_KEY_PATH" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PREDICTIONS_PATH}" "${LOCAL_TEMP_DIR}"
rsync $RSYNC_OPTS -e "ssh -i $SSH_KEY_PATH" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_LEADERBOARD_PATH}" "${LOCAL_TEMP_DIR}"
rsync $RSYNC_OPTS -e "ssh -i $SSH_KEY_PATH" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_JUDGED_PATH}" "${LOCAL_TEMP_DIR}"


echo ""
echo "結果をバックアップディレクトリに整理します..."

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 1. predictions と judged をタイムスタンプ付きフォルダにバックアップ
if [ -d "${LOCAL_TEMP_DIR}predictions" ] || [ -d "${LOCAL_TEMP_DIR}judged" ]; then
    VERSIONED_BACKUP_DIR="${BACKUP_ROOT_DIR}results_${TIMESTAMP}/"
    echo "  -> predictionsとjudgedを '${VERSIONED_BACKUP_DIR}' に保存します。"
    mkdir -p "$VERSIONED_BACKUP_DIR"
    
    # 存在するディレクトリのみを移動
    [ -d "${LOCAL_TEMP_DIR}predictions" ] && mv "${LOCAL_TEMP_DIR}predictions" "$VERSIONED_BACKUP_DIR"
    [ -d "${LOCAL_TEMP_DIR}judged" ] && mv "${LOCAL_TEMP_DIR}judged" "$VERSIONED_BACKUP_DIR"
fi

# 2. leaderboard の中身を蓄積用のフォルダにコピー（-n: 既に存在する場合は上書きしない）
if [ -d "${LOCAL_TEMP_DIR}leaderboard" ]; then
    echo "  -> backup/leaderboard/ に新しい結果を追加（マージ）します。"
    rsync -av --ignore-existing "${LOCAL_TEMP_DIR}leaderboard/" "$BACKUP_LEAD_DIR"
fi

# 一時ディレクトリをクリーンアップ
rm -rf "$LOCAL_TEMP_DIR"

echo ""
echo "処理が完了しました。"
echo "全てのバックアップは backup/ ディレクトリにあります。"