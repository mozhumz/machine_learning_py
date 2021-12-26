import htmlUtil
import re
from lxml import etree

def soup_parse(res,html):
    soup=htmlUtil.parseHtml(html)
    brief_intro = soup.find('div', attrs={'class': 'lemma-summary'})
    if brief_intro:
        res['brief_intro'] = brief_intro
    card = soup.find('div', attrs={'class': 'basic-info J-basic-info cmn-clearfix'})
    card_dict = {}
    card_lr = card.findChildren(recursive=False)
    if card_lr:
        for dl in card_lr:
            dl_children = dl.findChildren(recursive=False)
            if dl_children:
                dl_name = None
                dl_cnt = 1
                for dl_child in dl_children:
                    if dl_cnt == 1:
                        dl_name = trim(dl_child.text)
                        dl_cnt += 1
                    else:
                        card_dict[dl_name] = trim(dl_child.text)
                        dl_cnt = 1
    print(card_dict)
    if card_dict:
        res['card'] = card_dict

def trim(text:str):
    return text.replace('\n',' ').replace('\xa0','').replace('\u3000','')

def lxml_parse():
    global html
    html = etree.HTML(html)
    # html=etree.parse(html,etree.HTMLParser())
    # result=html.xpath('//*')  #//代表获取子孙节点，*代表获取所有
    card = html.xpath('//div[@class="basic-info J-basic-info cmn-clearfix"]/dl')
    for s in card:
        s_text = s.xpath('//text()')
    print(card)


if __name__ == '__main__':
    req_url='https://baike.baidu.com/item/%E5%8D%8E%E4%B8%BA%E6%8A%80%E6%9C%AF%E6%9C%89%E9%99%90%E5%85%AC%E5%8F%B8'
    html=htmlUtil.getReq(url=req_url)
    res={}
    soup_parse(res,html)
    # lxml_parse()
    # import re
    #
    # c = re.sub('<[^<]+?>', '', ''.join(html)).replace('\n', '').strip()
    # print(c)