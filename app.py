import requests
import pandas as pd
import streamlit as st
from io import StringIO
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

URL = "https://finance.naver.com/sise/"
HEADERS = {"User-Agent": "Mozilla/5.0"}

st.set_page_config(page_title="네이버 금융 시황 요약", page_icon="📈")
st.title("📈 네이버 금융 시황 요약")
st.write("네이버 금융 시세 페이지를 스크래핑해서 GPT가 시장 흐름을 요약해줍니다.")


def get_naver_finance_tables():
    response = requests.get(URL, headers=HEADERS)
    response.encoding = "euc-kr"

    tables = pd.read_html(StringIO(response.text))

    stock_tables = []
    for table in tables:
        cols = [str(c) for c in table.columns]
        if any("종목명" in c for c in cols):
            table = table.dropna(how="all")
            stock_tables.append(table)

    if not stock_tables:
        raise ValueError("종목 관련 테이블을 찾지 못했습니다.")

    return stock_tables


def make_stock_text(tables, max_rows=30):
    result = []
    for i, df in enumerate(tables, start=1):
        df = df.head(max_rows)
        result.append(f"[테이블 {i}]\n{df.to_string(index=False)}")
    return "\n\n".join(result)


prompt = ChatPromptTemplate.from_template("""
너는 증권사 리서치센터 애널리스트입니다.
아래 네이버 금융 종목 데이터를 바탕으로 시장 흐름을 간단히 요약해 주세요.

요약 기준:
1. 상승/하락이 두드러지는 종목
2. 거래대금 또는 거래량이 큰 종목
3. 업종 또는 테마상 특징
4. 투자자가 유의해야 할 점
5. 과도한 매수·매도 권유는 하지 말 것

데이터:
{stock_data}
""")


@st.cache_resource
def get_chain():
    llm = ChatOpenAI(model="gpt-5.5")
    return prompt | llm | StrOutputParser()


if st.button("시황 요약 보기", type="primary"):
    try:
        with st.spinner("네이버 금융 데이터를 가져오는 중..."):
            tables = get_naver_finance_tables()
            stock_text = make_stock_text(tables)

        with st.spinner("GPT가 시황을 분석하는 중..."):
            chain = get_chain()
            summary = chain.invoke({"stock_data": stock_text})

        st.subheader("🧾 시황 요약")
        st.markdown(summary)

        with st.expander("원본 테이블 보기"):
            for i, df in enumerate(tables, start=1):
                st.write(f"테이블 {i}")
                st.dataframe(df.head(30))
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
