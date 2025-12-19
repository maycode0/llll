import re
import codecs
import json

with open('datasets/yelp/yelp_100_300_50k_pro.json', 'r', encoding='utf-8') as f:
    datas = json.load(f)
def comprehensive_clean(text):
    # 1. 处理 Unicode 转义字符 (如 \\u00e8 -> è)
    # 使用 codecs.decode 可以一次性处理文中所有的 \\uXXXX
    try:
        # 先处理可能的编码冲突，将其转为正常的 unicode
        text = codecs.decode(text, 'unicode-escape')
    except Exception:
        pass 

    # 2. 处理多余的引号 \\\"\" -> "
    # 替换顺序：先长后短
    text = text.replace('\\\"\"', '"').replace('\"\"', '"')

    # 3. 处理字面量换行符 \\n
    # 替换为一个小空格，避免单词粘连
    text = text.replace('\\n', ' ').replace('\n', ' ')

    # 4. 清理多余的空格 (将连续的多个空格合并为一个)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
'''
for data in datas:
    data['text'] = comprehensive_clean(data['text']).lower()
with open('datasets/yelp/yelp_100_300_50k_pro.json', 'w', encoding='utf-8') as f:
    json.dump(datas, f, ensure_ascii=False, indent=4)
'''
print(datas[15]['text'])
