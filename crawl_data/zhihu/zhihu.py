import hashlib
import execjs
from common import http_util,MyLogger
from concurrent.futures import ThreadPoolExecutor

import socket


log = MyLogger.Logger('all.log',level='info')
socket.setdefaulttimeout(30)
'''
avatar_url: "https://pic2.zhimg.com/50/8658418bc_720w.jpg?source=54b3c3a5"
excerpt: "学科该词有以下两种含义：①相对独立的知识体系。人类所有的知识划分为五大门类：自然科学，农业科学，医药科学，工程与技术科学，人文与社会科学。②我国高等学校本科教育专业设置的学科分类，我国高等教育划分为13个学科门类：哲学、经济学、法学、教育学、文学、历史学、理学、工学、农学、医学、军事学、管理学、艺术学。"
id: "19618774"
introduction: "学科该词有以下两种含义：①相对独立的知识体系。人类所有的知识划分为五大门类：自然科学，农业科学，医药科学，工程与技术科学，人文与社会科学。②我国高等学校本科教育专业设置的学科分类，我国高等教育划分为13个学科门类：哲学、经济学、法学、教育学、文学、历史学、理学、工学、农学、医学、军事学、管理学、艺术学。"
is_black: false
is_super_topic_vote: true
is_vote: false
name: "学科"
type: "topic"
url: "http://www.zhihu.com/api/v3/topics/19618774"
'''
with open('g_encrypt.js', 'r') as f:
    ctx1 = execjs.compile(f.read(), cwd='F:\\idea_workspace\\github\\python-course-master\\node_modules')
def get_signature(id,limit,offset,cookie):
    f = '101_3_2.0+/api/v3/topics/%s/children?limit=%s&offset=%s+"%s"'%(id,limit,offset,http_util.parse_cookie(cookie).get('d_c0'))
    # f = '101_3_2.0+/api/v3/topics/%s/children+"%s"'%(id,http_util.parse_cookie(cookie).get('d_c0'))
    fmd5 = hashlib.new('md5', f.encode()).hexdigest()
    log.log(fmd5)

    encrypt_str = "2.0_%s" % ctx1.call('b', fmd5)
    log.log(encrypt_str)
    return encrypt_str






from faker import Faker
import requests
import json
import time
from lxml import etree
import threading
from threading import Thread
from pymongo import MongoClient
# from hyper.contrib import HTTP20Adapter

threadPool = ThreadPoolExecutor(max_workers=100)
rlock = threading.RLock()

def get_children(id,limit,offset,cookie):
    api_url='https://www.zhihu.com/api/v3/topics/%s/children?limit=%s&offset=%s'
    url=api_url%(id,limit,offset)
    headers = {
        'user-agent': Faker().user_agent(),
        # 'user-agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36',
        'cookie':cookie,
        'x-zse-93':'101_3_2.0',
        'x-zse-96':get_signature(id,limit,offset,cookie),
        'referer':'https://www.zhihu.com/topic/19776749/hot',
        'sec-ch-ua':'"Google Chrome";v="95", "Chromium";v="95", ";Not A Brand";v="99"',
        'sec-ch-ua-mobile':'?0',
        'sec-ch-ua-platform':'"Windows"',
        'sec-fetch-dest':'empty',
        'sec-fetch-mode':'cors',
        'sec-fetch-site':'same-origin',
        'x-ab-param':'top_test_4_liguangyi=1;pf_adjust=0;zr_expslotpaid=1;se_ffzx_jushen1=0;tp_dingyue_video=0;qap_question_visitor= 0;qap_question_author=0;tp_zrec=0;tp_topic_style=0;tp_contents=2;pf_noti_entry_num=0;zr_slotpaidexp=1',
        'x-ab-pb':'CtYBhAJDAKED4wR9AtgCVwQUBYsF4wUZBbULCgQzBOkEEgXMAowEfwUYBlYMMQYBC6IDNwWABZsLCwQ0BI0EBwxQA9oEQwVkBOAEUQUPC3UEKQVqAbIFQAY3DHQB8wPgCxEFOQZHADIDcgNFBJ4FTwNXA6ADMwX0C7QK3AsBBioGnwLXAtEE7ApgCzAGOwK5AlYFiQy0AEAB+AOMBRwGzwsbANYEFQVSBdgFUgv0A7EF1wu3Aw4F5AppAcIFFgamBOcFwQQKBkEG9gJCBD8AVQU/BjQMjAIyBRJrABgAAQAAAAAAAAADAAAAAAAAAAABAAAAAAACAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAQAAAAsAAAAAAAAACwAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=',
        'x-requested-with':'fetch'
    }
    # sessions=requests.session()
    # sessions.mount(url, HTTP20Adapter())
    # response=sessions.get(url,headers=headers)
    res= {}
    try:
        response=requests.get(url=url,headers=headers)
        response.encoding = 'utf-8'
        res=response.text
        log.log(res)
        response.close()
        # time.sleep(1)
        if not res:
            return []

        res=json.loads(res).get('data')
        ls=set_parent_id(id, res)
        # 异步写入Mongo
        threadPool.submit(batch_insert_to_mongo,list=res)
        del res
        return ls
    except Exception as e:
        log.log("Exception=%s"%e)
        return []



def get_all_children(id,limit,offset,cookie):
    # for child in child_list:
    #     child['parentId']=id
    #     log.log(child)
    log.log("get_all_children-id=%s"%id)
    res=[]
    child_list=get_children(id,limit,offset,cookie)
    while child_list and len(child_list)>0:
        res.extend(child_list)
        offset+=limit
        child_list=get_children(id,limit,offset,cookie)
    if res and len(res)>0:
        log.log('get_all_children-first:'+str(res[0]))
        log.log('get_all_children-last:'+str(res[-1]))
    return res


def set_parent_id(id, res):
    ls=[]
    for c in res:
        c['parentId'] = str(id)
        ls.append(c['id'])
    return ls


def batch_insert_to_mongo(list):
    for one in list:
        with rlock:
            insert_mongo(one)

def load_children(id,limit,offset,cookie):
    log.log('load_children:id=%s'%id)
    child_list=get_all_children(id,limit,offset,cookie)
    # if len(child_list)==0:
    #     return
    while len(child_list)>0:
        relist=[]
        # batch_insert_to_mongo(child_list)
        for child in child_list:
            if(str(child)=='19776751'):
                continue
            # load_children(id=child,limit=limit,offset=offset,cookie=cookie)
            # 获取某节点的下一级子列表
            node_list=get_all_children(child,limit,offset,cookie)
            if len(node_list)>0:
                relist.extend(node_list)
        del child_list
        child_list=relist
        # task=threadPool.submit(load_children,id=child['id'],limit=limit,offset=offset,cookie=cookie)

    return


def insert_mongo(data):
    client = MongoClient('mongodb://mozhu:123456@localhost:27017/admin')
    db = client.hyj.zhihu_topic5
    if db.find_one({'id':data['id'],'parentId':data['parentId']}):
    # if db.find_one({'id':data['id']}):
        log.log('已存在')
    else:
        db.insert_one(data)
        log.log('插入成功')
    return

def process(id,limit,offset,cookie):

    threadlist = []


    # for i in range(number):
    #     t = Thread(target=all_topic_urls, args=(cut_lists[i],))
    #     t.start()
    #     threadlist.append(t)
    # for thd in threadlist:
    #     thd.join()


def start(id,limit, offset, cookie):

    child_list = get_all_children(id, limit, offset, cookie)
    # if len(child_list)==0:
    #     pass
    # batch_insert_to_mongo(child_list)
    threadlist = []
    relist = []
    for child in child_list:
        if (str(child) == '19776751'):
            continue
        node_list = get_all_children(id, limit, offset, cookie)
        if len(node_list) > 0:
            relist.extend(node_list)
    log.log('relist:len=%s,data=%s' % (len(relist), relist))
    for id in relist:
        load_children(id, limit, offset, cookie)
        t = Thread(target=load_children, args=(id, limit, offset, cookie))
        t.start()
        threadlist.append(t)
    return threadlist
        # task=threadPool.submit(load_children,id=child['id'],limit=limit,offset=offset,cookie=cookie)


if __name__ == '__main__':
    cookie='_zap=7c0c3042-fc21-40e8-ba21-43e86519000a; d_c0="AAARWzrPxhOPTskGcpEuVAsXuVwn2cynUow=|1632539393"; _9755xjdesxxd_=32; YD00517437729195%3AWM_TID=vzdptJ53pPlEAUBEEUM%2F8UOknqvlS0XD; _xsrf=146c7345-3f23-4693-ae0a-87f19399b85c; Hm_lvt_98beee57fd2ef70ccdd5ca52b9740c49=1636642498; Hm_lpvt_98beee57fd2ef70ccdd5ca52b9740c49=1636642498; captcha_session_v2="2|1:0|10:1636642499|18:captcha_session_v2|88:WnlramU4ZGlLRFd4Z2tHVmU3MXNtaE9ZTGNEK25WdS9ERzhoUkNSNkdkaGlOMExVTUROMEgzV1Z6U1Nxc2cveg==|0abacb7e93feff9bf8593008e1df7c8a92896607691f62031eeb0a123b69927f"; gdxidpyhxdE=c0DnZgtBbQxInClVjABCWPl1%5CG1krUPI%2Fr135s0I%2Bf1hKnNkae1zT8TUyimS6LqVjgcyOtaoNmQo8oOdul1IDmnkuZOu5wg%2BugqANWWpMaiZ4H6n5V9YbX%2FQc9d%2BCZHYrzOZBGhVjc0X0eMyiNTcb9Z81VirbyWhViNbjZ%2BxQ4ADISkW%3A1636643402352; YD00517437729195%3AWM_NI=FicaJGoVxfplYJa1HEiJYQuyIwotvFtc4d8Vhrcx%2FiUnLRNRxAtPcKRnzAMcLYm3HlbWgzVRZ2eFKXsMuPzTtRS3P6YPVXCVI4OPAWw2Vp3VDspKwA7DOShXkjyRKivaVVY%3D; YD00517437729195%3AWM_NIKE=9ca17ae2e6ffcda170e2e6eed1ee4b86edb9a3d941b6a88bb7c14b969f8f84ae3fb59cbbadaa63a7b2a9b8b82af0fea7c3b92a86b7ad82c94dad8ab887f3689b96afd2cb4fab9697b2f8738ba9a195f54eb8bdfc85bb5eacaead99ae6397b284b3cd5abc9fbcb1e24192b2b6a2d0699087a3d4cb68a289a99bd55a9aeeaba7ca54b193bbb1f960b88e9dd4d33e9b909ab9c25e8bec81b1c45db5ee9a91b25e88adc0acf64bf59c81b4eb54a9b6b7b3ca63b6bf9ab7dc37e2a3; KLBRSID=0a401b23e8a71b70de2f4b37f5b4e379|1636642512|1636642496'
    limit=10
    offset=0
    # 根话题id
    # id='19776749'
    # 「形而上」话题
    # id='19778298'
    # load_children(id,limit,offset,cookie)
    relist=[]
    # for id in ['19778317','19580349','19550912','19778298']:
    for id in ['19778317']:
        threadlist=start(id,limit, offset, cookie)
        relist.extend(threadlist)

    for thd in relist:
        thd.join()
    log.log('done')








