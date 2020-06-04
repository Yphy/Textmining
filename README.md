# NLP

NLP(natural language processing) 을 빙자한 텍스트 마이닝을 공부하고 있습니다.


## ml_analysis

자연어처리를 공부하면서 필요하다고 생각되는 딥러닝 네트워크들을 공부하여 정리를 하고 있습니다. 
bentrevett님의 pytorch 코드에서 많은 도움을 얻었습니다.   


**[Sentiment analysis](https://github.com/Yphy/NLP/tree/master/ml_analysis/Sentiment%20Analysis)**
 
- simple sentiment analysis
- upgraded sentiment analysis
- faster sentiment analysis

**[Sequence to Sequence, seq2seq](https://github.com/Yphy/NLP/tree/master/ml_analysis/Seq2seq)**

- Sequence to Sequence Learning with Neural Networks
-  Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
- Neural Machine Translation by Jointly Learning to Align and Translate



## MD&A analysis

전자공시 시스템 Dart 에서 한국 코스피 시장에 상장되어있는 기업들의 사업보고서 중 **'이사의 경영진단 및 분석'** 파트에 해당하는 md&a section의 텍스트의 변화를 분석하는 프로젝트 및 논문작성을 진행중입니다.

해당 프로젝트에 대한 미국의 유사연구는 다음과 같습니다.
>_'Similarity_BROWN_TUCKER(2011 JAR)Large-Sample Evidence on Firms Year-over-Year MD_A Modifications'_ 
>_'Similarity_Cohen_Malloy_Nguyen(2019)LAZY PRICES'_   



### Data
다트 사이트에서 제공하는 api를 이용하여 2010~2019년 10년 동안 연속적으로 kospi200에 편입되어있던 기업들의 사업보고서만을 대상으로 하였습니다. 편입퇴출 효과를 제거하기 위해 한번이라도 kospi200에서 제외되거나 사업보고서가 공시되지 않은 기업은 제외하였고 정정보고서나 추가보고서가 같은 해 중복 될 경우 첫 번째 공시날짜를 기준으로 하였습니다. 참고로 크롤링시 다트에서는 1분에 100번 이상의 request가 있을 시 사용자의 네트워크를 24시간 동안 차단합니다.

 - [dart crawling](https://github.com/Yphy/NLP/blob/master/md%26a_analysis/Data_crawling.py)

추후에 기업의 md&a modification score 와 future return 를 함께 분석하기 위해 크롤링한 코스피 기업들의 해당 기간동안의 주가와 수익률을 FinanceDataReader 패키지를 통해 불러왔습니다. 

		pip install -U finance-datareader
 - [kospi200 data](https://github.com/Yphy/NLP/blob/master/md%26a_analysis/1.kospi200%20data.ipynb)
 
 

### Preprocessing

한글은 영어와 달리 교착어이기때문에 단순 띄어쓰기만으로 의미가 구분되지 않습니다. 논문에서는 토큰화에 대한 정확한 설명이 없었기 때문에 형태소분석 후 모든 형태소를 사용하는 방법과 조사나 어미를 제외하는 등의 경우를 고려하고자 합니다.
형태소 분석기는 **konlpy** 와 **Mecab-ko**를 사용하였습니다. 
아래 표의 pos tags를 선택하여 lemmatization을 합니다.

![mecab pos tags](https://user-images.githubusercontent.com/47969237/78253547-86944c80-752f-11ea-9276-34582748e48c.PNG)

 - [tokenization](https://github.com/Yphy/NLP/blob/master/md%26a_analysis/3.Tokenization.ipynb)
 
 

### Measuring modification score

사업보고서의 변화율 척도를 확인하는 방법으로 이전 보고서와 다음 해의 보고서의 text similarity를 계산합니다. 그 방법으로 **자카드 유사도**와 **코사인 유사도** 를 사용합니다. 사업보고서의 filing date인 rcp_dt 를 인덱스 , 기업명을 컬럼으로 하여 데이터프레임형태로 유사도를 저장하였습니다. 추후 다른 측정기법들을 추가하겠습니다.


1. sim_Jaccard = $$|{D_1}^{TF}∩{D_2}^{TF}|\over|{D_1}^{TF}∪{D_2}^{TF}|$$



2. sim_cosine = $${D_1}^{TF}\cdot {D_2}^{TF}\over||{D_1}^{TF}|∪|{D_2}^{TF}||$$




### Create main table

 - [ Similarity_Cohen_Malloy_Nguyen(2019)LAZY PRICES](https://github.com/Yphy/NLP/blob/master/md%26a_analysis/Similarity_Cohen_Malloy_Nguyen(2019)LAZY%20PRICES.ipynb](https://github.com/Yphy/NLP/blob/master/md%26a_analysis/Similarity_Cohen_Malloy_Nguyen(2019)LAZY%20PRICES.ipynb))

- [Similarity_BROWN_TUCKER(2011 JAR)Large-Sample Evidence on Firms Year-over-Year MD_A Modifications](https://github.com/Yphy/NLP/blob/master/md%26a_analysis/2.Similarity_Cohen_Malloy_Nguyen(2019)LAZY%20PRICES.ipynb](https://github.com/Yphy/NLP/blob/master/md%26a_analysis/2.Similarity_Cohen_Malloy_Nguyen(2019)LAZY%20PRICES.ipynb))
