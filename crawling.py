from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import openpyxl

html = urlopen('https://search.naver.com/search.naver?where=news&query=%EC%96%91%ED%8C%8C%20%EB%A7%88%EB%8A%98&sm=tab_opt&sort=2&photo=0&field=0&reporter_article=&pd=3&ds=1991.01.01&de=1991.12.31&docid=&nso=so%3Ada%2Cp%3Afrom19910101to19911231%2Ca%3Aall&mynews=0&refresh_start=0&related=0')
bsObject = BeautifulSoup(html,'html.parser')

a = bsObject.body.find('div',{"class" : 'group_news'}).find('ul',{"class":'list_news'})

for lis in a.find_all('div',{'class':'news_info'}):
    print(lis.find('span',{'class' : "info"}).text.strip().split('.'))

for lis in a.find_all('a',{'class':'news_tit'}):
    print(lis.get('title'))
    print(lis.get('title'))

for lis in a.find_all('a',{'class':'api_txt_lines dsc_txt_wrap'}):
    print(lis.text)
    print(lis.text)
    # print(lis.get('title'))
import csv
data = [[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18]] # 이차원 리스트
with open('test.csv','w', newline='') as f:
    makewrite = csv.writer(f)
    for lis in a.find_all('div',{'class':'news_info'}):
        makewrite.writerow(lis.find('span',{'class' : "info"}).text.strip().split('.'))
        makewrite.writerow(lis.find('span',{'class' : "info"}).text.strip().split('.'))
    for lis in a.find_all('a',{'class':'news_tit'}):
        a = lis.get('title').strip()
        makewrite.writerow()