from faker import Faker
import requests
import json
import time
from lxml import etree
import threading
from threading import Thread
from pymongo import MongoClient
from hyper.contrib import HTTP20Adapter
import hashlib

topic_api_url = 'https://www.zhihu.com/node/TopicsPlazzaListV2'

def insert_mongo(data):
    client = MongoClient('mongodb://mozhu:123456@localhost:27017/admin')
    db = client.hyj.Zhihu
    if db.find_one(data):
        print('已存在')
    else:
        db.insert_one(data)
        print('插入成功')


def seplist(start_urls, cut_number):
    cut_list = []
    for i in range(cut_number):
        cut_list.append([])
    for i in range(len(start_urls)):
        cut_list[i % cut_number].append(start_urls[i])
    return cut_list


def all_topic_id(url):
    headers = {
        'user-agent': Faker().user_agent()
    }
    response = requests.get(url, headers=headers)
    response.encoding = 'utf-8'
    html = etree.HTML(response.text)
    ids = []
    for each in html.xpath('//li[@class="zm-topic-cat-item"]'):
        topic_id = each.xpath('@data-id')[0]
        ids.append(topic_id)
    return ids

def get_parent_data():
    headers = {
        'User-Agent': Faker().user_agent()
    }
    url = 'https://www.zhihu.com/topics'

    response = requests.get(url, headers=headers)
    res = response.text
    html = etree.HTML(res)
    ul = html.xpath("//ul[@class='zm-topic-cat-main clearfix']/li");
    parent_topic = {}
    for li in ul:
        title = li.xpath('./a/text()')[0];
        topic_id = li.xpath('./@data-id')[0];
        parent_topic[topic_id] = title

    return parent_topic

def all_topic_urls(lists):
    for item in lists:

        offset = 0
        while True:
            lists = get_topic_children_node(item)
            if not lists:
                break
            for one_topic in lists:
                htm = etree.HTML(one_topic)
                one_topic_url = htm.xpath('//a[1]/@href')[0]
                topic_name = htm.xpath('//strong/text()')[0]
                link = 'https://www.zhihu.com' + one_topic_url
                topic_dict = {'topic_name': topic_name,
                              'topic_link': link}
                with rlock:
                    insert_mongo(data=topic_dict)
            offset += 20


def get_topic_children_node(item,  offset=0):
    headers = {
        'user-agent': Faker().user_agent()
    }
    data = {'method': 'next',
            "params": json.dumps({"topic_id": int(item), "offset": offset, "hash_id": ""})}
    res = requests.post(topic_api_url, data=data, headers=headers)
    lists = json.loads(res.text)['msg']
    return lists

def get_children2(id):
    res={}
    api_url='https://www.zhihu.com/api/v3/topics/%s/children'
    url=api_url%id
    headers = {
        'user-agent': Faker().user_agent(),
        'cookie':'_zap=da65d76c-fbcb-4c76-b731-1bbff7fb04ec; d_c0="AJCaOogwaxGPTjuq8M-UoT-tMeRndcdBNWM=|1592031146"; _ga=GA1.2.2020371219.1592031151; _xsrf=FC5ygIfy9Id687u5zuQ2rhDZD26rAfS2; __snaker__id=YqPDWFm68rU5FPKm; _9755xjdesxxd_=32; YD00517437729195%3AWM_TID=n6EMpKJWJsFFQERRAUMqyz%2FmzH62puZU; gdxidpyhxdE=bCfVQYiVxphcSI1VuWp82ViBonGYx%2B94%2Bc59UNOInO1gyEZtm5VejBMyUUSuk7whM5XUv7ESISl6dpSEd2hv%5CsoERW3E4Xm%2FQ2dLV9iEvwJP1ITidZXEe%5CYTyX67LjVowQlI4k%5CLs9fh%2Bvzxgh9yfSgEoSAkGJmjPz89r6V3cmuOKjIQ%3A1633443302320; YD00517437729195%3AWM_NI=Nv46%2F8vUXvDaBROGu%2BpcxIzpu%2B6QFUlVro4L7PuPWimUEgml5RJ%2BMJKE48N0qX6J6IbAwUaTzO4zsqYix5x8oRkI2VVLuYXeMxss5Ic5I2RMrDr7%2FKvASeb0ILD2gQyheG8%3D; YD00517437729195%3AWM_NIKE=9ca17ae2e6ffcda170e2e6eed1aa34b6b782d4b5748d868fa7c14f929b8b84f87c85868b91c453a1b09fd6cf2af0fea7c3b92a8eaafcd7ae34b68b9897c76b828e81a8f821b4bdb8aaf97cb7f5fab2c55fb08abe90b748f590bbb1d663af8b8b88f068f78e86ccd042899bfcd9cd3cb388faabb543a29fb9b6e460ba91fc9bf27b9bb7b8d6fb34ad8ea5aab15e9694bbbadb6f8fa8bd94b843a1eb87d3d147b29cbed8ae41b69f9d86e13c8b948cd3e65bbbb8aba8bb37e2a3; z_c0="2|1:0|10:1633442413|4:z_c0|92:Mi4xT3pVSUJBQUFBQUFBa0pvNmlEQnJFU1lBQUFCZ0FsVk5iYWhKWWdBZDg4OG1RT2t6Q1NNX1N4TU5zVUtFcHd1Q0F3|5fcf3e2ca5ef5c2ab208793aa1a0c1fcf44564aff46faea55c815141169af26e"; tst=r; q_c1=be1793119d154b20846b05d3dad2418f|1636173236000|1636173236000; __utmc=51854390; __utmz=51854390.1636173236.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); __utmv=51854390.100--|2=registration_date=20170201=1^3=entry_date=20170201=1; __utma=51854390.2020371219.1592031151.1636173610.1636173610.3; NOT_UNREGISTER_WAITING=1; SESSIONID=iszLu5J1gez7KZsprLPb2iITVrMF816iqGlupSPDSg8; osd=UlERB0lPDXS9rgFeektGL_h1_NJvL3MO99k9GUsqYQOK41kkB_Id9NilBVN86h0J8dUE-V7GVZKSJxm_B7nMpLc=; JOID=UVoXBklMBnK8rgJVfEpGLPNz_dJsJHUP99o2H0oqYgiM4lknDPQc9NuuA1J86RYP8NUH8ljHVZGZIRi_BLLKpbc=; Hm_lvt_98beee57fd2ef70ccdd5ca52b9740c49=1636173082,1636173200,1636181886; Hm_lpvt_98beee57fd2ef70ccdd5ca52b9740c49=1636182000; KLBRSID=d1f07ca9b929274b65d830a00cbd719a|1636182030|1636173080',
        'x-zse-93':'101_3_2.0',
        'x-zse-96':'2.0_a7Y0Nve8nqOfoRY0fTYyo0Uqo_FYcXNBB0F8Q7uq6XxY',
    }
    sessions=requests.session()
    sessions.mount(url, HTTP20Adapter())
    response=sessions.get(url,headers=headers)
    # response=requests.get(url=url,headers=headers)
    response.encoding = 'utf-8'
    print(response.text)

    return res

def process():
    global rlock
    rlock = threading.RLock()
    start_url = 'https://www.zhihu.com/topics'
    id_lists = all_topic_id(start_url)
    number = int(len(id_lists) / 2)
    cut_lists = seplist(id_lists, number)
    threadlist = []
    for i in range(number):
        t = Thread(target=all_topic_urls, args=(cut_lists[i],))
        t.start()
        threadlist.append(t)
    for thd in threadlist:
        thd.join()

def md5(content):
    hl=hashlib.md5()
    hl.update(content.encode(encoding='utf-8'))
    return hl.hexdigest()

if __name__ == '__main__':
    s = time.time()
    # process()
    # 19776749 19612637
    # get_children2(19612637)
    api_url='https://www.zhihu.com/api/v3/topics/%s/children'
    url=api_url% 19612637
    print(md5(url+'AJCaOogwaxGPTjuq8M-UoT-tMeRndcdBNWM=|1592031146'))
    # lists = get_topic_children_node(item=19554825)
    # print(lists)
    # get_parent_data()
    print('抓取全部话题用时：', time.time() - s)
