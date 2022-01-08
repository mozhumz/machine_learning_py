import htmlUtil
from lxml import etree


def soup_parse(html):
    res = {}
    soup = htmlUtil.parseHtml(html)
    # 简介
    brief_intro = soup.find('div', attrs={'class': 'lemma-summary'})
    if brief_intro:
        res['brief_intro'] = trim(brief_intro.text)

    # 名片属性
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

    # 解析目录
    catalog_dict = dict()
    catalog = soup.find('div', attrs={'class': 'catalog-list'})
    catalog_ol_list = catalog.findChildren(recursive=False)
    if catalog_ol_list:
        pre_level = 0
        pre_level1_key = ''
        for ol in catalog_ol_list:
            li_list = ol.findChildren(recursive=False)
            if li_list:
                for li in li_list:
                    li_cls = li.attrs['class'][0]
                    span_list = li.findChildren(recursive=False)
                    if span_list:
                        li_text = trim(span_list[-1].text)
                        if 'level1'.__eq__(li_cls):
                            pre_level1_key = li_text
                            catalog_dict[pre_level1_key] = []
                        elif 'level2'.__eq__(li_cls):
                            v_list = catalog_dict.get(pre_level1_key)
                            v_list.append(li_text)

    print(catalog_dict)
    if catalog_dict:
        res['catalog_dict'] = catalog_dict

    # 解析目录对应的内容
    content_dict = dict()
    cata_content = soup.find('div', attrs={'class': 'main-content J-content'})
    if cata_content:
        div_content_list = cata_content.findChildren(recursive=False)
        pre_key = ''
        if div_content_list:
            for div_content in div_content_list:
                div_class = get_one_html_class(div_content)

                if div_class:

                    # 标题
                    if 'para-title'.__eq__(div_class):
                        try:
                            title = div_content.select('.title-text')
                            pre_key = title[-1].contents[-1]
                        except Exception as e:
                            print(e)
                            pre_key = None

                    # 内容
                    if 'para'.__eq__(div_class):
                        content_dict[pre_key] = trim(div_content.text)

    print(content_dict)
    if content_dict:
        res['content_dict'] = content_dict

    return res


def get_one_html_class(html):
    if html:
        cls = html.attrs.get('class')
        if cls:
            return cls[0]
        return None
    return None


def trim(text: str):
    return text.replace('\n', ' ').replace('\xa0', '').replace('\u3000', '')


def lxml_parse():
    global html
    html = etree.HTML(html)
    # html=etree.parse(html,etree.HTMLParser())
    # result=html.xpath('//*')  #//代表获取子孙节点，*代表获取所有
    card = html.xpath('//div[@class="basic-info J-basic-info cmn-clearfix"]/dl')
    for s in card:
        s_text = s.xpath('//text()')
    print(card)


def parse_baike(req_url):
    try:
        html = htmlUtil.getReq(url=req_url)
        return soup_parse(html)
    except Exception as e:
        print(e)
        html = htmlUtil.getReq(url=req_url)
        return soup_parse(html)


def parse_keyword(keyword):
    keyword = htmlUtil.encodeurl(keyword)
    req_url = 'http://baike.baidu.com/item/%s' % keyword
    return parse_baike(req_url)


if __name__ == '__main__':
    keyword = 'java'
    print('res:%s' % parse_keyword(keyword))
    # lxml_parse()
    # import re
    #
    # c = re.sub('<[^<]+?>', '', ''.join(html)).replace('\n', '').strip()
    # print(c)
