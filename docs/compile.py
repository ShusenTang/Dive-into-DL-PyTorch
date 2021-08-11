import os
import re
import glob
from utils import get_content_files, downgrade_heading, create_empty_page, pure_markdown


def init():
    os.system("rm -rf build")
    os.system("mkdir -p build/docs")
    os.system("mkdir -p build/target")
    os.system("cp -r img tex -t build")
    os.system("cp _sidebar.md -t build")
    for file in get_content_files():
        print("Found", file)
        os.system("cp {} build/docs".format(file)) # move all markdown files to build/docs
    os.chdir("build")
    print("Inited.", end="\n\n")


def create_chapter():
    files = []
    with open("_sidebar.md", "r") as f:
        for line in f.readlines():
            # 提取页面文件
            res = re.findall(r"\]\((.+)\)$", line)

            if len(res) > 0:  # 存在对应 markdown 文件
                filename = "docs/{}".format(re.sub(r"^(.+)\/", "", res[0]))
                if line[0] == " ":
                    # 将子章节内的标题统一降一级
                    downgrade_heading(filename)

            else:  # 不存在对应 markdown 文件
                # 提取页面标题
                title = re.findall(r"\*\s?\[?(.+?)\]?\(?\)?$", line)[0].replace("\\", "")
                filename = "docs/{}.md".format(title.replace(" ", ""))
                
                # this is a work around for not including this chapter
                if title == "简介":
                    continue
                
                if line[0] == " ":
                    # 针对子章节生成提示
                    create_empty_page(filename, title, is_chapter=False)
                else:
                    # 针对章节生成标题
                    create_empty_page(filename, title, is_chapter=True)
            files.append(filename)
    return files


def format():
    print("Formating docs...")
    for filename in glob.glob("docs/*.md"):
        pure_markdown(filename)
    os.system("prettier --write docs")


def build(files):
    print("Building...")
    title = r"《动手学深度学习》PyTorch 实现"
    author = r"原书作者：阿斯顿・张、李沐、扎卡里C.立顿、\\ 亚历山大J.斯莫拉、以及其他社区贡献者 \thanks{GitHub 地址: https://github.com/d2l-ai/d2l-zh}"
    date = r"\today"

    os.system("pandoc {} -o target/release.pdf \
    --from=markdown+link_attributes+footnotes+blank_before_header \
    --to=pdf \
    --toc \
    --toc-depth=2 \
    --resource-path=./docs \
    --listings \
    --pdf-engine=xelatex \
    --template=tex/custom_template.tex \
    -V mathspec \
    -V graphicx \
    -V colorlinks \
    -V title='{}' \
    -V author='{}' \
    -V date='{}' \
    --include-before-body=tex/cover.tex".format(" ".join(files), title, author, date)
    )
    
    print("Done.", end="\n\n")


def main():
    init()
    files = create_chapter()
    format()
    build(files)

if __name__ == "__main__":
    main()
