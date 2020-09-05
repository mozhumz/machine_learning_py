import re
from string import punctuation
from common import common_util
"""
去除中英文标点符号
"""
text = " Hello, world! 这，是：我;第!一个程序\?()（）<>《》【】 "
punc2=r"""~`!#$%^&*()_+-=|\'\\;":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·"""
text=text+punc2
print(common_util.trim_with_space(text))
# punctuation =punctuation+ r"""~`!#$%^&*()_+-=|\'\\;":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·"""
s =text
punc = punctuation + r'··.,;\\《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：'
dicts={i:'' for i in punc}
punc_table=str.maketrans(dicts)
new_s=s.translate(punc_table)
print(new_s)

text= re.sub(r"[{}]+".format(punc)," ",text)
print(text)