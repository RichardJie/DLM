#!/bin/bash

#====================================#
# 🧙 神之自动 Git 推送脚本（无需参数）
# 适配服务器+临时GitHub身份
# 创建并推送 dpo 分支
#====================================#

#=========================#
#     ✅ 可自定义参数区    #
#=========================#

# Git 用户信息
GIT_USER_NAME="RichardJie"
GIT_USER_EMAIL="tracy_wufz@163.com"

# 使用的分支名
BRANCH_NAME="dpo"

# commit 信息
COMMIT_MESSAGE="update: push dpo branch to GitHub"

# 临时 SSH key 的绝对路径
SSH_KEY_PATH="/hpc2ssd/JH_DATA/spooler/yuxuanzhao/lijungang/wujie/.ssh/id_ed25519_tempgithub"

# GitHub 仓库信息
REPO_NAME="RichardJie/DLM"
GITHUB_HOST_ALIAS="github-tempgit"

#=========================#
#    ⚙️ SSH 配置           #
#=========================#

SSH_CONFIG="$HOME/.ssh/config"

# 如果 SSH config 中未包含目标 host，则添加
if ! grep -q "$GITHUB_HOST_ALIAS" "$SSH_CONFIG" 2>/dev/null; then
    echo "🔧 配置 SSH Host: $GITHUB_HOST_ALIAS 到 $SSH_CONFIG"
    mkdir -p ~/.ssh
    cat <<EOF >> $SSH_CONFIG

Host $GITHUB_HOST_ALIAS
    HostName github.com
    User git
    IdentityFile $SSH_KEY_PATH
    IdentitiesOnly yes
EOF
else
    echo "✅ SSH config 已存在 host $GITHUB_HOST_ALIAS，跳过配置"
fi

#=========================#
#    ⚙️ Git 配置          #
#=========================#

echo "🔧 设置 Git 用户信息..."
git config user.name "$GIT_USER_NAME"
git config user.email "$GIT_USER_EMAIL"

#=========================#
#    🚀 Git 操作流程      #
#=========================#

# 检查远程 origin 是否设置
if ! git remote | grep origin > /dev/null; then
    echo "🔧 添加远程 origin 为 $GITHUB_HOST_ALIAS:$REPO_NAME.git"
    git remote add origin $GITHUB_HOST_ALIAS:$REPO_NAME.git
fi

echo "🔄 切换到 main 并同步最新"
git checkout main 2>/dev/null || git checkout -b main
git pull origin main 2>/dev/null

echo "🌱 创建并切换到分支：$BRANCH_NAME"
git checkout -b $BRANCH_NAME 2>/dev/null || git checkout $BRANCH_NAME

echo "➕ 添加文件..."
git add .

echo "📝 提交代码：$COMMIT_MESSAGE"
git commit -m "$COMMIT_MESSAGE"

echo "🚀 推送分支到远程..."
git push --set-upstream origin $BRANCH_NAME

echo -e "\n✅ 分支 $BRANCH_NAME 已成功推送到远程仓库 $REPO_NAME 🚀"
