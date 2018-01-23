# this script creates github markdown files from my "custom" python docstrings
# messy code ahead :)


import sys


def get_doc_blocks(text):
    blocks = []
    in_block = False
    for i, line in enumerate(text):
        if line.strip().startswith('"""') and in_block == False:
            blocks.append({"kind": text[i-1].strip().split()[0], "definition": text[i-1].strip()[:-1]})
            block_start = i
            in_block = True
        elif line.strip().startswith('"""') and in_block == True:
            blocks[-1]["content"] = text[block_start: i]
            in_block = False
    return blocks


def escape_underscores(string):
    pos_underscores = []
    for i, char in enumerate(string):
        if char in ["_", ">"]:
            pos_underscores.append(i)
    for i, pos in enumerate(pos_underscores):
        string = string[:(i+pos)] + "\\" + string[(i+pos):]
    return string


def class2md(block):
    md = "# Class " + block["definition"].split()[1] + " - Documentation\n\n"

    for i, line in enumerate(block["content"]):
        if line.strip() == "":
            block["content"][i] = '\n\n'
        elif line.strip().startswith("'"):
            block["content"][i] = ' ``' + line.strip()[1:] + '``  \n'
        elif line.strip().startswith("#"):
            block["content"][i] = ' ' + line.strip()[1:] + '  \n'
        else:
            block["content"][i] = line.strip()

    md += " ".join(line for line in block["content"][1:]) + "\n\n"
    return escape_underscores(md)


def methods_list2md(blocks):
    md = "## Methods - Overview\n\n| name | description |\n|:-|:-|\n"
    for block in blocks:
        name = block["definition"].split(" ")[1].split("(")[0]
        description = block["content"][0].strip().split('"""')[1].strip()
        md += "| " + name + " | " + description + " |\n"
    return escape_underscores(md)


def method2md(block):
    name = escape_underscores(block["definition"].split(" ")[1].split("(")[0])
    md = "## " + name + "\n\n" + "``` python\n" + block["definition"] + "\n```\n"
    description = block["content"][0].strip().split('"""')[1].strip()
    pos_params = -1
    only_returns = False
    for i, line in enumerate(block["content"]):
        if line.strip() == "Parameters":
            pos_params = i
            break
        if line.strip() == 'Returns':
            pos_params = i
            only_returns = True
            break
    if pos_params == -1:
        description += " ".join(block["content"][1:]) + "\n\n"
        md += escape_underscores(description)
        return md
    for i, line in enumerate(block["content"][1:pos_params]):
        if line.strip() == "":
            block["content"][i+1] = '\n\n'
        elif line.strip().startswith("'"):
            block["content"][i+1] = ' ' + line.strip()[1:] + '  \n'
        else:
            block["content"][i+1] = line.strip()
    description += " " + " ".join(block["content"][1:pos_params]) + "\n\n"
    md += escape_underscores(description)
    pos_params += 2
    had_param = False
    while pos_params < len(block["content"]):
        tmp = block["content"][pos_params].split(":")
        if len(tmp) == 1 or only_returns == True:
            if tmp[0].strip() == "Returns" or only_returns == True:
                if only_returns == True: pos_params -= 2
                md += "\n| returns | type | description |\n|:-|:-|:-|\n"
                tmp2 = block["content"][pos_params+2].split(":")
                param2 = tmp2[0].strip()
                param2_type = tmp2[1].strip()
                param2_descr = block["content"][pos_params+3].strip()
                md += "| " + param2 + " | " + param2_type + " | " + param2_descr + " |\n"
                return md
        if not had_param:
            md += "| parameter | type | description |\n|:-|:-|:-|\n"
            had_param = True
        param = tmp[0].strip()
        param_type = tmp[1].strip()
        param_descr = block["content"][pos_params+1].strip()
        md += "| " + param + " | " + param_type + " | " + param_descr + " |\n"
        pos_params += 3
    return md


def doc2md(files):
    for file_name in files:
        with open(file_name, "rt") as handle:
            text = handle.read().split('\n')

        blocks = get_doc_blocks(text)
        md = ""
        if blocks[0]["kind"] == "class":
            md += class2md(blocks[0])
            md += methods_list2md(blocks[1:])
            for block in blocks[1:]:
                md += method2md(block)
        elif blocks[0]["kind"] == "def":
            md += "# File " + file_name.split("/")[-1] + " - Documentation\n\n"
            md += methods_list2md(blocks)
            for block in blocks:
                md += method2md(block)
        
        with open(file_name.split('/')[-1][:-2]+'md', 'wt') as handle:
            handle.write(md)
        print(file_name.split('/')[-1][:-2]+'md' + " created.")


def main():
    files = ["../pysster/Data.py",
             "../pysster/Grid_Search.py",
             "../pysster/Model.py",
             "../pysster/Motif.py",
             "../pysster/utils.py"]
    doc2md(files)


if __name__ == "__main__":
    main()