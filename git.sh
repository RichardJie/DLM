#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "✗ 脚本失败（行：$LINENO）"; exit 1' ERR

REPO_URL="https://github.com/RichardJie/DLM"
DEFAULT_BRANCH="main"
COMMIT_MSG="${1:-update}"

# 基本配置（仅作用于当前仓库）
git config user.name "RichardJie"
git config user.email "tracy_wufz@163.com"

# 确保存在 origin，并指向 REPO_URL
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi

# 提交本地更改（如果有）
git add -A
if git diff --cached --quiet; then
  echo "没有需要提交的更改"
else
  git commit -m "$COMMIT_MSG"
  echo "更改已提交"
fi

# 当前分支
current_branch="$(git rev-parse --abbrev-ref HEAD)"
echo "当前分支: $current_branch"
echo "同步远端 $DEFAULT_BRANCH ..."

# 先抓取远端
git fetch origin "$DEFAULT_BRANCH"

# 如果本地落后远端，先 rebase 到远端分支上
# 统计：左边=本地独有提交数，右边=远端独有提交数
read ahead behind < <(git rev-list --left-right --count HEAD...origin/"$DEFAULT_BRANCH" | awk '{print $1, $2}')
if [[ "${behind:-0}" -gt 0 ]]; then
  echo "检测到远端有 ${behind} 个你本地没有的提交，执行 rebase ..."
  git pull --rebase origin "$DEFAULT_BRANCH" || {
    echo "✗ rebase 发生冲突或失败，请解决后重试（必要时：git rebase --abort）"
    exit 1
  }
fi

echo "推送到远程 $DEFAULT_BRANCH 分支 ..."
if git push --set-upstream origin "$current_branch:$DEFAULT_BRANCH"; then
  echo "✓ 推送成功"
else
  echo "✗ 推送失败（可能还在落后或被保护分支限制）"
  echo "如需覆盖远端（危险）：git push --force-with-lease origin \"$current_branch:$DEFAULT_BRANCH\""
  exit 1
fi
