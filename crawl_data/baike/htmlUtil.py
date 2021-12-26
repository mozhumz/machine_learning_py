import requests
from bs4 import BeautifulSoup
import urllib.parse
from faker import Faker
headers={
'User-Agent':Faker().user_agent()

}

def encodeurl(url):
    return urllib.parse.quote(url)

def getReq(url):
    r=requests.get(url=url,headers=headers)
    r.encoding = r.apparent_encoding
    return r.text

# def getReq(url,cookies):
#     r=requests.get(url=url,headers=headers,cookies=cookies)
#     r.encoding = r.apparent_encoding
#     return r.text

def postReq(url,data):
    r = requests.post(url=url,data=data)
    r.encoding = r.apparent_encoding
    return r.text

def parseHtml(html):
    soup=BeautifulSoup(html,'lxml')
    return soup

def getSoup(url):
    html=getReq(url)
    return parseHtml(html)

# soup=getSoup(url='http://www.shicimingju.com/chaxun/list/3025.html')
ck={
    'PHPSESSID':'bm7goiro2pohs6jpillgkqi2f3',
    'kw':'%u738B%u7EF4',
    # 'Hm_lvt_4c1638db937a6ad4a0e6a8bdfa32146f':'1519184040',
    # 'Hm_lpvt_4c1638db937a6ad4a0e6a8bdfa32146f':'1519193029',
    # 'AJSTAT_ok_pages':'28',
    # 'AJSTAT_ok_times':'2',
    # '__tins__3854170':'%7B%22sid%22%3A%201519192480299%2C%20%22vd%22%3A%2010%2C%20%22expires%22%3A%201519194829484%7D',
    # '__51cke__':'',
    # '__51laig__':'30',
    # 'top_is_guding':'no',
    # 'shiju':'',
    # 'cate':'all',
    'font-weight':'%u4E2D'


}


