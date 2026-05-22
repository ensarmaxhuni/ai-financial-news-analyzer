import os
import re
from datetime import datetime
from typing import Dict, List, Tuple
from urllib.parse import quote_plus

import feedparser
import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


st.set_page_config(
    page_title="AI Financial News Analyzer",
    page_icon="📈",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .hero-card {
        padding: 2rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(37,99,235,0.25), rgba(124,58,237,0.25));
        border: 1px solid rgba(148,163,184,0.25);
        margin-bottom: 1.5rem;
    }
    .news-card {
        padding: 1.1rem;
        border-radius: 16px;
        background: rgba(15, 23, 42, 0.07);
        border: 1px solid rgba(148,163,184,0.25);
        margin-bottom: 1rem;
    }
    .positive { color: #16a34a; font-weight: 700; }
    .negative { color: #dc2626; font-weight: 700; }
    .neutral { color: #d97706; font-weight: 700; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

FINANCE_FEEDS = {
    "Yahoo Finance": "https://finance.yahoo.com/rss/topstories",
    "MarketWatch Top Stories": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "CNBC Finance": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "Investing.com News": "https://www.investing.com/rss/news.rss",
}

DEFAULT_KEYWORDS = [
    "inflation", "interest rates", "Federal Reserve", "earnings", "stocks",
    "ETF", "crypto", "AI", "recession", "GDP", "oil", "unemployment",
    "Treasury yields", "Nvidia", "Apple", "Microsoft", "Tesla"
]


def clean_html(raw_text: str) -> str:
    if not raw_text:
        return ""
    clean = re.sub("<.*?>", " ", raw_text)
    return re.sub(r"\s+", " ", clean).strip()


@st.cache_data(ttl=900)
def fetch_rss_feed(feed_url: str, limit: int = 25) -> List[Dict]:
    parsed = feedparser.parse(feed_url)
    articles = []
    for entry in parsed.entries[:limit]:
        title = clean_html(entry.get("title", "Untitled"))
        summary = clean_html(entry.get("summary", ""))
        articles.append({
            "source": parsed.feed.get("title", "Unknown source"),
            "title": title,
            "summary": summary,
            "link": entry.get("link", ""),
            "published": entry.get("published", entry.get("updated", "Unknown date")),
            "text": f"{title}. {summary}",
        })
    return articles


@st.cache_data(ttl=900)
def fetch_query_news(query: str, limit: int = 20) -> List[Dict]:
    encoded_query = quote_plus(f"{query} finance OR stock market OR economy")
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    parsed = feedparser.parse(url)
    articles = []
    for entry in parsed.entries[:limit]:
        title = clean_html(entry.get("title", "Untitled"))
        summary = clean_html(entry.get("summary", ""))
        articles.append({
            "source": "Google News",
            "title": title,
            "summary": summary,
            "link": entry.get("link", ""),
            "published": entry.get("published", entry.get("updated", "Unknown date")),
            "text": f"{title}. {summary}",
        })
    return articles


def score_sentiment(text: str) -> Tuple[float, str]:
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.15:
        return score, "Bullish / Positive"
    if score <= -0.15:
        return score, "Bearish / Negative"
    return score, "Neutral / Mixed"


def extract_keywords(text: str, keywords: List[str]) -> List[str]:
    text_lower = text.lower()
    return [kw for kw in keywords if kw.lower() in text_lower]


def build_dataframe(articles: List[Dict], keywords: List[str]) -> pd.DataFrame:
    rows = []
    for article in articles:
        score, label = score_sentiment(article["text"])
        hits = extract_keywords(article["text"], keywords)
        rows.append({
            "Published": article["published"],
            "Source": article["source"],
            "Title": article["title"],
            "Summary": article["summary"],
            "Sentiment Score": round(score, 3),
            "Sentiment": label,
            "Keywords": ", ".join(hits) if hits else "None",
            "URL": article["link"],
        })
    return pd.DataFrame(rows)


def get_openai_summary(df: pd.DataFrame, topic: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return ""

    compact_news = []
    for _, row in df.head(10).iterrows():
        compact_news.append(
            f"- Title: {row['Title']}\n"
            f"  Sentiment: {row['Sentiment']}\n"
            f"  Keywords: {row['Keywords']}\n"
            f"  Summary: {row['Summary'][:350]}"
        )

    prompt = f"""
You are a financial market intelligence analyst.

Topic: {topic}

Analyze the following financial news items and produce a concise professional market intelligence brief.

Include:
1. Executive summary
2. Key market themes
3. Bullish factors
4. Bearish risks
5. Business/investor implications
6. Watchlist for the next 24-72 hours

News:
{chr(10).join(compact_news)}
"""
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a careful financial news analyst. Do not provide financial advice."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as exc:
        return f"OpenAI summary could not be generated: {exc}"


def create_basic_market_brief(df: pd.DataFrame, topic: str) -> str:
    if df.empty:
        return "No articles were found for this query."

    total = len(df)
    bullish = len(df[df["Sentiment"] == "Bullish / Positive"])
    bearish = len(df[df["Sentiment"] == "Bearish / Negative"])
    neutral = len(df[df["Sentiment"] == "Neutral / Mixed"])
    avg_score = df["Sentiment Score"].mean()

    top_keywords = (
        df["Keywords"].str.split(", ").explode()
        .replace("None", pd.NA).dropna().value_counts().head(8).index.tolist()
    )

    if avg_score > 0.15:
        tone = "overall positive/bullish"
    elif avg_score < -0.15:
        tone = "overall negative/bearish"
    else:
        tone = "mixed or neutral"

    return f"""
### Market Intelligence Brief: {topic}

Based on **{total} financial news articles**, the current news tone appears **{tone}**.

**Sentiment breakdown**
- Bullish / Positive: {bullish}
- Bearish / Negative: {bearish}
- Neutral / Mixed: {neutral}

**Average sentiment score:** {avg_score:.3f}

**Key themes detected:** {", ".join(top_keywords) if top_keywords else "No major predefined themes detected."}

**Business interpretation:**  
The current news environment shows a mix of market signals. Positive sentiment may indicate stronger confidence, better earnings expectations, or favorable macroeconomic developments. Negative sentiment may indicate risks such as inflation pressure, interest-rate uncertainty, weak growth, geopolitical concerns, or sector volatility.

**Important note:**  
This tool is for educational and market research purposes only. It does not provide investment advice.
"""


def make_download_report(df: pd.DataFrame, brief: str, topic: str) -> str:
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    article_lines = []
    for _, row in df.iterrows():
        article_lines.append(
            f"### {row['Title']}\n"
            f"- Source: {row['Source']}\n"
            f"- Published: {row['Published']}\n"
            f"- Sentiment: {row['Sentiment']} ({row['Sentiment Score']})\n"
            f"- Keywords: {row['Keywords']}\n"
            f"- URL: {row['URL']}\n"
        )
    return f"# AI Financial News Analyzer Report\n\nGenerated: {report_time}\nTopic: {topic}\n\n---\n\n{brief}\n\n---\n\n# Article Analysis\n\n{chr(10).join(article_lines)}"


st.markdown("""
<div class="hero-card">
    <h1>📈 AI Financial News Analyzer</h1>
    <p>
        Analyze financial news, detect market sentiment, extract key themes,
        and generate an executive-style market intelligence brief.
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Analyzer Settings")
    topic = st.text_input(
        "Topic, ticker, ETF, company, or macro theme",
        value="AI stocks",
        help="Examples: Nvidia, S&P 500, Bitcoin, inflation, oil prices, Tesla, ETFs"
    )
    source_mode = st.radio("News source mode", ["Topic search", "Finance RSS feeds"], index=0)
    article_limit = st.slider("Number of articles", min_value=5, max_value=40, value=15, step=5)
    custom_keywords = st.text_area("Custom keywords to track", value=", ".join(DEFAULT_KEYWORDS))
    use_openai = st.checkbox("Use OpenAI executive summary if OPENAI_API_KEY is configured", value=True)
    st.caption("Educational and research purposes only. Not financial advice.")

keywords = [kw.strip() for kw in custom_keywords.split(",") if kw.strip()]

if st.button("Analyze Financial News", type="primary", use_container_width=True):
    with st.spinner("Fetching and analyzing financial news..."):
        if source_mode == "Topic search":
            articles = fetch_query_news(topic, article_limit)
        else:
            articles = []
            per_feed_limit = max(3, article_limit // max(1, len(FINANCE_FEEDS)))
            for _, feed_url in FINANCE_FEEDS.items():
                articles.extend(fetch_rss_feed(feed_url, per_feed_limit))
            articles = articles[:article_limit]

        df = build_dataframe(articles, keywords)

    if df.empty:
        st.warning("No articles found. Try another topic or source mode.")
        st.stop()

    avg_sentiment = df["Sentiment Score"].mean()
    bullish_count = len(df[df["Sentiment"] == "Bullish / Positive"])
    bearish_count = len(df[df["Sentiment"] == "Bearish / Negative"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Articles Analyzed", len(df))
    col2.metric("Avg Sentiment", f"{avg_sentiment:.3f}")
    col3.metric("Bullish", bullish_count)
    col4.metric("Bearish", bearish_count)

    st.subheader("AI Market Intelligence Brief")
    openai_brief = get_openai_summary(df, topic) if use_openai else ""
    brief = openai_brief if openai_brief else create_basic_market_brief(df, topic)
    st.markdown(brief)

    st.subheader("Sentiment Distribution")
    sentiment_counts = df["Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    st.bar_chart(sentiment_counts.set_index("Sentiment"))

    st.subheader("Analyzed Articles")
    for _, row in df.iterrows():
        sentiment_class = "neutral"
        if row["Sentiment"] == "Bullish / Positive":
            sentiment_class = "positive"
        elif row["Sentiment"] == "Bearish / Negative":
            sentiment_class = "negative"

        st.markdown(
            f"""
            <div class="news-card">
                <h4>{row['Title']}</h4>
                <p><strong>Source:</strong> {row['Source']} | <strong>Published:</strong> {row['Published']}</p>
                <p><strong>Sentiment:</strong> <span class="{sentiment_class}">{row['Sentiment']}</span> ({row['Sentiment Score']})</p>
                <p><strong>Keywords:</strong> {row['Keywords']}</p>
                <p>{row['Summary']}</p>
                <a href="{row['URL']}" target="_blank">Read original article</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    report = make_download_report(df, brief, topic)
    st.download_button(
        label="Download Market Intelligence Report",
        data=report,
        file_name=f"financial_news_report_{topic.replace(' ', '_').lower()}.md",
        mime="text/markdown",
        use_container_width=True,
    )
else:
    st.info("Enter a topic in the sidebar and click **Analyze Financial News** to begin.")
    st.markdown("""
    ### Suggested topics to test:
    - Nvidia
    - S&P 500
    - Bitcoin
    - Inflation
    - Federal Reserve
    - AI stocks
    - Tesla earnings
    - Oil prices
    """)
