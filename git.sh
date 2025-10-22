#!/bin/bash

# 设置用户名和邮箱
git config user.name "RichardJie"
git config user.email "tracy_wufz@163.com"

# 添加所有更改
git add .

# 提交更改
git commit -m "update"

# 获取当前分支名称
current_branch=$(git rev-parse --abbrev-ref HEAD)

# 推送当前分支到远程的 main 分支
git push https://github.com/RichardJie/DLM "$current_branch":main

