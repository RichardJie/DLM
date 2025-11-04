#!/usr/bin/env bash
set -e

git config user.name "RichardJie"
git config user.email "tracy_wufz@163.com"

git add -A
git commit -m "updata" || echo "没有更改"
git push