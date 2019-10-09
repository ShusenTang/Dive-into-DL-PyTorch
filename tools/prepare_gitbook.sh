#!/usr/bin/env bash

gbdir=".gitbook"
summary="${gbdir}/SUMMARY.md"
readme="${gbdir}/README.md"

mkdir -p ${gbdir}

echo '根据项目README.md自动生成gitbook项目所需 SUMMARY.md 文件 ......'
cat README.md \
  | awk '/^## 目录/ {print "# SUMMARY \n"} /^### / {hasmd=index($0, "md"); if (hasmd > 0) {print "* "substr($0, 5)} else print "* ["$2 $3"]()"} /^\[/ {print $0} /\.\.\./ {print "   * "$0}' \
  | sed 's/https:\/\/github.com\/ShusenTang\/Dive-into-DL-PyTorch\/blob\/master\/docs\///g' \
  | sed 's/^\[/   \* \[/g' \
  > ${summary}

echo '根据项目README.md自动生成gitbook项目所需 README.md 文件 ......'
cat README.md | awk '!/^### / && !/^\[/ && !/目录/ && !/更新/ {print $0}' > ${readme}

cd ${gbdir}

ln -fs ../docs/chapter* .
ln -fs ../img .

gitbook serve .
#if [[ ! -L  ]]