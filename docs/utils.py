import re


def get_content_files(sidebar_filename="_sidebar.md"):
    """
    parse the sidebar file and return a list of all content files
    """

    with open(sidebar_filename, "r") as f:
        files = re.findall(r"\]\((.+)\)$", f.read(), re.M)
        return files


def downgrade_heading(filename):
    """
    downgrade all headings by 1 level
    """

    print("Downgrading headings in {}".format(filename))
    with open(filename, "r") as f:
        lines = f.readlines()

    with open(filename, "w") as f:
        is_codeblock = False
        for line in lines:
            if line.startswith("```"):
                is_codeblock = not is_codeblock
            elif line.startswith("#") and not is_codeblock:
                line = "#" + line
            f.write(line)


def create_empty_page(filename, title, is_chapter):
    """
    create an empty page with the given title
    """
    
    print("Create new page for {}".format(filename))
    with open(filename, "w") as f:
        if is_chapter:
            f.write(f"# {title}\n")
        else:
            f.write(f"## {title}\n\n此章节仍未编写完成，敬请期待。\n")


def pure_markdown(filename):
    """
    convert all markdown to pure markdown
    """

    with open(filename, "r") as f:
        lines = f.readlines()

    with open(filename, "w") as f:
        i = 0
        while i < len(lines):
            window = lines[i : i + 4]
            if window[0].startswith("<div"):
                src = re.findall(r"src=\"(.*?)\"", window[1])
                src = "" if src == [] else src[0]
                width = re.findall(r"width=\"(.*?)\"", window[1])
                width = "80%" if width == [] else "{}%".format(min(int(width[0]) * 100 // 750, 100))
                cap = re.findall(r"<div.*?>(.+?)</div>", window[-1]) + re.findall(
                    r"<center.*?>(.+?)</center>", window[-1]
                )
                cap = "" if cap == [] else cap[0].replace("\n", "").replace("\t", "").replace(" ", "")
                f.write("\n![{}]({}){{ width={} }}\n\n".format(cap, src, width))
                if cap == "":
                    i += 3
                else:
                    i += 4
            else:
                f.write(lines[i])
                i += 1
