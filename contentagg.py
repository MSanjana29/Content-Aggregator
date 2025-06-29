import streamlit as st
import feedparser
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from newspaper import Article
import urllib.parse
import wikipedia

# --- Clean HTML Summary ---
def clean_summary(raw_html):
    soup = BeautifulSoup(raw_html, 'html.parser')
    return soup.get_text()

# --- TF-IDF-based Summarization ---
def tfidf_summarize(text, num_sentences=2):
    sentences = text.split('. ')
    if len(sentences) <= num_sentences:
        return text.strip()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    scores = tfidf_matrix.sum(axis=1).flatten().tolist()[0]
    ranked_sentences = sorted(((score, sentence) for sentence, score in zip(sentences, scores)), reverse=True)
    summary = ". ".join([sentence for _, sentence in ranked_sentences[:num_sentences]])
    return summary.strip()

# --- Softmax Normalization ---
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# --- Extract Full Article Text ---
def get_full_article_text(url):
    article = Article(url)
    try:
        article.download()
        article.parse()
        return article.text
    except:
        return ""

# --- Fetch RSS News and Summarize ---
def fetch_rss_news(query):
    query = query.replace(" ", "+")
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)

    if not feed.entries:
        return []

    articles = []
    relevance_scores = []

    for entry in feed.entries[:10]:
        title = entry.title
        raw_link = entry.link
        published = entry.published if 'published' in entry else 'N/A'
        source = entry.source.title if 'source' in entry else 'Google News'
        raw_summary = entry.summary if 'summary' in entry else ''

        # Fix redirect link
        parsed_url = urllib.parse.urlparse(raw_link)
        real_url = urllib.parse.parse_qs(parsed_url.query).get('url')
        link = real_url[0] if real_url else raw_link

        # Fetch article content
        full_text = get_full_article_text(link)
        if full_text and len(full_text.split('. ')) > 2:
            summarized_text = tfidf_summarize(full_text, num_sentences=2)
        else:
            clean_text = clean_summary(raw_summary)
            summarized_text = tfidf_summarize(clean_text, num_sentences=2)

        relevance_score = len(set(query.lower().split()) & set(title.lower().split()))
        relevance_scores.append(relevance_score)

        articles.append({
            "title": title,
            "link": link,
            "published": published,
            "source": source,
            "relevance_score": relevance_score,
            "summary": summarized_text
        })

    normalized_scores = softmax(np.array(relevance_scores))
    for i, article in enumerate(articles):
        article['relevance_score'] = normalized_scores[i]

    return sorted(articles, key=lambda x: x['relevance_score'], reverse=True)

# --- Get Wikipedia Summary ---
def get_article_theory(title):
    try:
        # Fetch a short theory/overview related to the article's title from Wikipedia
        return wikipedia.summary(title, sentences=2)
    except wikipedia.exceptions.DisambiguationError as e:
        # Handle disambiguation errors if there are multiple possible matches
        return wikipedia.summary(e.options[0], sentences=2)
    except wikipedia.exceptions.HTTPTimeoutError:
        return "🔍 Could not fetch the theory due to a timeout."
    except:
        return "🔍 No general description found for this topic."

# --- Streamlit Setup ---
st.set_page_config(page_title="RSS Content Aggregator", page_icon="🔗", layout="wide")

# --- Login Page ---
def login():
    st.title("🔐 CONTENT AGGREGATION (RSS ONLY)")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if username == "admin" and password == "1234":
                st.session_state.logged_in = True
                st.success("✅ Login successful!")
            else:
                st.error("❌ Invalid credentials.")

# --- Main App ---
def main_app():
    st.title("🔗 RSS Content Aggregator")
    st.write("Enter a topic to fetch the latest news articles and a short theory about it.")

    query = st.text_input("🔍 Enter your topic:")
    if query:
        with st.spinner("Getting general info and fetching news..."):
            articles = fetch_rss_news(query)

        if articles:
            st.subheader("📰 Latest Articles")

            for idx, article in enumerate(articles):
                st.markdown(f"### 🔗 [{article['title']}]({article['link']})")
                st.caption(f"Source: {article['source']} | Published: {article['published']}")
                st.write(f"*Relevance Score:* {article['relevance_score']:.4f}")
                
                # Get theory (small description) for each article
                article_theory = get_article_theory(article['title'])
                
                st.subheader(f"📚 Theory about {article['title']}")
                st.info(article_theory)

                st.text_area(
                    "🔹 Summary:",
                    article['summary'],
                    height=150,
                    disabled=True,
                    key=f"summary_{idx}"
                )
                st.markdown("---")

            st.subheader("📊 Relevance Score Chart")
            df = pd.DataFrame(articles)
            st.bar_chart(df[["title", "relevance_score"]].set_index("title"))
        else:
            st.warning("⚠ No articles found for the given topic. Please try a different query.")

# --- Auth Routing ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
else:
    main_app()
