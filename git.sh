#!/bin/bash

# 设置用户名和邮箱
git config user.name "RichardJie"
git config user.email "tracy_wufz@163.com"

# 添加所有更改
git add .

# 提交更改（如果有更改的话）
if git diff --cached --quiet; then
    echo "没有需要提交的更改"
else
    git commit -m "update"
    echo "更改已提交"
fi

# 获取当前分支名称
current_branch=$(git rev-parse --abbrev-ref HEAD)

echo "当前分支: $current_branch"
echo "推送到远程 main 分支..."

# 推送当前分支到远程的 main 分支
git push https://github.com/RichardJie/DLM "$current_branch":main

echo "✓ 推送成功"