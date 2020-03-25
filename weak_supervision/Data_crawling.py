import pandas as pd
import re
from bs4 import BeautifulSoup
import requests
from urllib.request import urlopen
import time

class ULRParser :
    def __init__(self, crp_code:list):
        api_code = 'api_code'  # 개인 api 키를 발급받아서 사용
        start_date = '20100101'
        # end_date = '20191231'
        page_set = '100'
        self._url = f'http://dart.fss.or.kr/api/search.xml?auth={api_code}&crp_cd={crp_code}start_dt={start_date}&bsn_tp=A001&page_set={page_set}'
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit 537.36 (KHTML, like Gecko) Chrome",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"}
        self._doc_url = 'https://dart.fss.or.kr/dsaf001/main.do?rcpNo='

    @staticmethod
    def get_dom(url, header) :
        req = requests.get(url, headers= header)
        html = req.text
        dom = BeautifulSoup(html, 'html.parser')
        return dom

    def _get_df(self):  # 검색된 회사의 사업보고서 내용을 데이터프레임으로 저장

        dom = ULRParser.get_dom(url=self._url, header= self._headers)
        te = dom.findAll('list')
        data = pd.DataFrame()

        for t in te:
            temp = pd.DataFrame(([[t.crp_cls.string, t.crp_nm.string, t.crp_cd.string, t.rpt_nm.string,
                                   t.rcp_no.string, t.flr_nm.string, t.rcp_dt.string, t.rmk.string]]),
                                columns=["crp_cls", "crp_nm", "crp_cd", "rpt_nm", "rcp_no", "flr_nm", "rcp_dt", "rmk"])
            data = pd.concat([data, temp])
        data.reset_index(drop=True, inplace=True)

        return (data)


    def _get_url(self ,data):  # dataframe을 넣었을 때 해당 사업보고서의 주소를 저장
        rcp_ls = []
        for row in data['rcp_no']:
            rcp_ls.append(row)
            urls = []
            for i in rcp_ls:
                urls.append(self._doc_url + i)
        return urls

    def _get_final_url(self, url):  # 사업보고서 주소를 넣으면 '이사의 경영진단 및 분석' 주소 반환
        dom = ULRParser.get_dom(url)
        body = str(dom.find('head'))

        try :
            a = re.search('이사의 경영진단 및 분석의견', body).span()
        except AttributeError as e :
            print(e)
            raise AttributeError

        b = re.search(r'viewDoc(.*);', body[a[0]:]).group()
        ls = b[8:-2].split(',')
        ls = [i[1:-1] for i in ls]
        ls[1] = ls[1][1:]  # 드러움
        ls[2] = ls[2][1:]
        ls[3] = ls[3][1:]
        ls[4] = ls[4][1:]
        ls[5] = ls[5][1:]

        url_final = 'http://dart.fss.or.kr/report/viewer.do?rcpNo=' + ls[0] + '&dcmNo=' + ls[1] + '&eleId=' + ls[
            2] + '&offset=' + ls[3] + '&length=' + ls[4] + '&dtd=dart3.xsd'
        return (url_final)

    def extracting_text(self, url):  # 경영진단 url 넣었을 시 해당 본문 텍스트 반환
        dom = ULRParser.get_dom(url)
        tables = []
        ka = dom.find_all('p')  # table을 제외한 본문 text 부분 찾아서 리스트안에저장
        for k in ka:
            tables.append(k.get_text())
        tables = tables[4:]  # 개요부분 삭제
        table = ''.join(tables)  # 리스트를 string으로 바꾼다
        table = table.replace('\xa0', '')  # 정리
        table = table.replace('\n', '')
        return table

    def main(self):  # 기업의 종목 코드 넣었을 시 근 10개년 종목내용을 저장한 dataframe 반환
        try:
            df = self._get_df()
            urls = self._get_url(df)
            strs = []
            for i in urls:
                strs.append(self._extracting_text(self._get_final_url(i)))
            str_series = pd.Series(strs)
            df['str'] = str_series
            return (df)

        except:
            pass