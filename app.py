import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Initialize LLM
llm = OllamaLLM(model="mistral")

# Chain of Thought Prompt
cot_prompt = PromptTemplate.from_template("""
Question: {input}
Context: {context}

Let's think step-by-step:
1. Identify key aspects of the question.
2. Analyze the context and extract meaningful information.
3. Form a coherent and fact-based answer.

Answer:
""")

reasoning_chain = cot_prompt | llm

def scrape_bing_news(query: str) -> str:
    url = f"https://www.bing.com/news/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")

        
        articles = soup.select("div.news-card") or soup.select("div.t_s") or soup.select("div.news-card-newsitem")
        if not articles:
            return ""

        news_list = []
        for art in articles[:5]:
            title_tag = art.select_one("a.title") or art.select_one("a")
            desc_tag = art.select_one("div.snippet") or art.select_one("div.sn_snip")
            title = title_tag.get_text(strip=True) if title_tag else "No Title"
            desc = desc_tag.get_text(strip=True) if desc_tag else "No Description"
            news_list.append(f"{title} - {desc}")

        return "\n".join(news_list)
    except Exception:
        return ""

def scrape_bbc_world_news() -> str:
    url = "https://www.bbc.com/news/world"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")

        articles = soup.select("a.gs-c-promo-heading")
        if not articles:
            return ""

        news_list = []
        for art in articles[:5]:
            title = art.get_text(strip=True)
            link = art['href']
            full_link = f"https://www.bbc.com{link}" if link.startswith("/") else link
            news_list.append(f"{title} - {full_link}")

        return "\n".join(news_list)
    except Exception:
        return ""

# UI
st.title("Live News Search with Chain of Thought Reasoning")

user_query = st.text_input("Enter your search query:")

if user_query:
    with st.spinner("Fetching and analyzing news..."):
        
        context = scrape_bing_news(user_query)

        
        if not context:
            context = scrape_bbc_world_news()

        if not context:
            context = "No relevant news found."

        cot_thinking = cot_prompt.format(input=user_query, context=context)
        reasoned_answer = reasoning_chain.invoke({"input": user_query, "context": context})

    st.subheader("Top News Context")
    st.text(context)

    st.subheader("Chain of Thought Reasoning")
    st.markdown(f"<pre>{cot_thinking}</pre>", unsafe_allow_html=True)

    st.subheader("CoT Reasoned Answer")
    st.write(reasoned_answer)
