from pathlib import Path
import os
import glob
work_dir = Path.cwd()

export_pdf_dir = work_dir / 'pdf'
print(export_pdf_dir)
if not export_pdf_dir.exists():
    export_pdf_dir.mkdir()

for md_file in list(sorted(glob.glob('./*/*.md'))):
    print(md_file)
    md_file_name = md_file
    zhanjie=md_file_name.split("/")[-2]
    print(zhanjie)
    pdf_file_name = md_file_name.replace('.md', '.pdf')
    pdf_file = export_pdf_dir/pdf_file_name
    os.makedirs(str(export_pdf_dir/zhanjie),exist_ok=True)
    print(pdf_file)
    cmd = "pandoc '{}' -o '{}' -s --highlight-style pygments  --latex-engine=xelatex -V mainfont='PingFang SC' --template=template.tex".format(md_file, pdf_file)
    os.system(cmd)