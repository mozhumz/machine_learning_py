import os
import re
from string import punctuation
# 标点符号集合
punc = punctuation + r'··.,;\\《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：'
dicts={i:'' for i in punc}
# 根据文件路径（相对或绝对）创建目录
def mkdirs(file):
    dir=os.path.dirname(os.path.abspath(file))
    print(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
'''
去除标点，用空格替换
'''
def trim_with_space(text):
    if text is None:
        return ''
    return re.sub(r"[{}]+".format(punc)," ",text)

'''
去除标点，用''替换
'''
def trim_with_out_space(text):
    if text is None:
        return ''
    return text.translate(str.maketrans(dicts))