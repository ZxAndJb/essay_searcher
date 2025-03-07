import re

def filter_content(text):
    # 过滤掉以 '![]' 开头的行
    # ^ 表示行开头，\Q 和 \E 用来转义 '[]' 中的特殊字符
    text = re.sub(r'\n\n!\[\][^\n]*', '', text)

    # 过滤掉被 <html> 标签包裹的内容
    # 匹配从 <html> 到 </html> 的所有内容，包括多行内容
    text = re.sub(r'<html>[\s\S]*?</html>', '', text)
    return text.strip()

def load_md(file_path, is_omit_ref_apx=True):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    sections = re.split(r'\n(?=#+ )', content)
    meta_information = {}
    title2content = {}
    for idx, sec in enumerate(sections):

        position = sec.find('\n\n')
        if position != -1:
            header = sec[:position]
        else:
            match = re.search(r'^(#{1,6})[\s\S]*?(?=\n)', sec, re.MULTILINE)
            header = match.group()

        if idx == 0:
            meta_information['title'] = header
        else:
            filtered_content = filter_content(sec[position:].strip())
            if len(filtered_content) != 0:
                title2content[header] = filtered_content

        if is_omit_ref_apx:
            if 'conclusion' in header.lower():
                break
    return {"meta_information": meta_information, "content": title2content}

def result_paser(retri_res, retain_fileds):
    if len(retri_res) == 1:
        retri_res = retri_res[0]
    res = []
    for data in retri_res:
        d_dict = {}
        for field in retain_fileds:
            d_dict[field] = data['entity'][field]
        res.append(d_dict)
    return res

def parse_dict_to_string(docs):
    result = ""
    for idx, doc in enumerate(docs):
        result +=f"文档信息{idx+1} \n"
        for key, value in doc.items():
            if key == "paper_name":
                result += f"论文标题： {value}\n"
            elif key == "section":
                result += f"论文子标题： {value}\n"
            elif key == "content":
                result += f"内容： {value}\n"
    return result

def parse_search_dict_to_string(docs):
    result = ""
    for idx, doc in enumerate(docs):
        for key, value in doc.items():
            if key =="url":
                result += f"相关链接{idx + 1}: {value}\n"
            if key == "content":
                result += f"简介： {value}\n"
    return result