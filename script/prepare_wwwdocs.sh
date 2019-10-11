#!/usr/bin/env bash
set -e

docs=".wwwdocs"
echo "本脚本将自动创建 ${docs} 目录"
echo '初始化项目依赖时将使用 docsify 工具自动生成本地文档web访问文档'
echo '本web文档为绿色创建，不会对现有项目产生副作用！不会产生产生git需要提交文件！'
echo '请放心食用 :))'

mkdir -p ${docs}

echo '根据项目README.md自动生成目录文件 ......'
cat README.md \
  | awk '/^## 目录/ {print "* [前言]()"} /^### / {hasmd=index($0, "md"); if (hasmd > 0) {print "* "substr($0, 5)} else print "* ["$2 $3"]()"} /^\[/ {print $0} /\.\.\./ {print "   * "$0}' \
  | sed 's/https:\/\/github.com\/ShusenTang\/Dive-into-DL-PyTorch\/blob\/master\/docs\///g' \
  | sed 's/^\[/   \* \[/g' \
  > ${docs}/_sidebar.md

echo '根据项目根目录下README.md以及docs/README.md合并生成项目所需${docs}导航 ......'
sredme=`cat docs/README.md`
cat README.md | awk -v sredme="${sredme}" '!/^### / && !/^\[/ && !/更新/ {print $0} /^## 目录/ {print sredme}' | sed 's/## 目录/## 说明/g' > ${docs}/README.md

echo '生成 docsify 所需入口文件......'
touch ${docs}/.nojekyll
cat > ${docs}/index.html << EOF
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Document</title>
  <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1" />
  <meta name="description" content="Description">
  <meta name="viewport" content="width=device-width, user-scalable=no, initial-scale=1.0, maximum-scale=1.0, minimum-scale=1.0">
  <link rel="stylesheet" href="//unpkg.com/docsify/lib/themes/vue.css">
</head>
<body>
  <div id="app"></div>
  <script>
    window.\$docsify = {
      loadSidebar: true,
      auto2top: true,
      alias: {
        '/.*/_sidebar.md': '/_sidebar.md'
      },
      markdown: {
        smartypants: true,
        renderer: {
          em: function(text) {
            return text;
          },
          codespan: function(text) {
            var reg = RegExp(/textrm/);
            if(text.match(reg)){
              return text
            }
            return '<code>' + text + '</code>';
          }
        }
      },
      plugins: [
        function(hook, vm) {
          hook.doneEach(function () {
            window.MathJax.Hub.Queue(["Typeset", MathJax.Hub, document.getElementById('app')]);
          })
        }
      ],
      externalLinkTarget: '_target',
      name: '《动手学深度学习（PyTorch版）》',
      repo: 'https://github.com/ShusenTang/Dive-into-DL-PyTorch'
    }
  </script>
  <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      extensions: ["tex2jax.js"],
      jax: ["input/TeX", "output/HTML-CSS"],
      tex2jax: {
        inlineMath: [ ['\$','\$'], ["\\(","\\)"] ],
        displayMath: [ ['\$$','\$$'], ["\\[","\\]"] ],
        processEscapes: true,
        skipTags: ["script", "noscript", "style", "textarea", "pre", "code", "a"]
      },
      "HTML-CSS": { fonts: ["TeX"] }
    });
  </script>
  <script src="//unpkg.com/docsify/lib/docsify.min.js"></script>
  <script src="//unpkg.com/docsify/lib/plugins/zoom-image.js"></script>
  <script src="//unpkg.com/docsify-copy-code"></script>
  <script src="//unpkg.com/prismjs/components/prism-bash.js"></script>
  <script src="//unpkg.com/prismjs/components/prism-python.js"></script>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js"></script>
</body>
</html>
EOF

echo '为各章节markdown文件以及图片建立软连接 ......'
cd ${docs}
ln -fs ../docs/chapter* .
ln -fs ../img .

port_used=`lsof -nP -iTCP -sTCP:LISTEN | grep 3000 | wc -l`
if [[ ${port_used} -gt 0 ]]; then
  echo '【警告】当前3000端口已被占用，请停止进程后再运行此脚本！'
  exit 1
fi

echo '启动web server，稍后请在浏览器中打开：http://localhost:3000 ，即可访问 ......'
if command -v docsify > /dev/null; then
  docsify serve .
else
  #echo 'docsify-cli 没有安装，建议使用：npm i docsify-cli -g'
  python -m SimpleHTTPServer 3000
fi