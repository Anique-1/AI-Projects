import streamlit as st
import requests
import toml
from datetime import datetime

# Streamlit page configuration
st.set_page_config(
    page_title="AI News Dashboard",
    page_icon="📰",
    layout="wide"
)

# Function to fetch news using SerpAPI
def get_news(api_key, query, num_results=10):
    url = "https://serpapi.com/search"
    params = {
        "q": f"{query} AI news",
        "tbm": "nws",
        "num": num_results,
        "api_key": api_key
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error fetching news: {e}")
        return None

# Load API key from secrets.toml
try:
    
    serp_api_key = "ENETR YOUR SERP API KEY HERE"
except FileNotFoundError:
    serp_api_key = None
    st.error("Please configure your SERP API key in .streamlit/secrets.toml")

# Streamlit app
def main():
    st.title("📰 AI News Dashboard")
    st.markdown("Enter a keyword to fetch the latest AI-related news articles.")

    # Input form
    with st.form(key="news_form"):
        query = st.text_input("Enter a keyword (e.g., machine learning, generative AI):", value="AI")
        submit_button = st.form_submit_button(label="Fetch News")

    if submit_button and query and serp_api_key:
        with st.spinner("Fetching news articles..."):
            news_data = get_news(serp_api_key, query)
        
        if news_data and "news_results" in news_data:
            st.success(f"Found {len(news_data['news_results'])} articles for '{query}'")
            
            # Display news articles
            for article in news_data["news_results"]:
                col1, col2 = st.columns([1, 3])
                
                # Thumbnail (if available)
                with col1:
                    if article.get("thumbnail"):
                        st.image(article["thumbnail"], width=100)
                    else:
                        st.image("https://via.placeholder.com/100", width=100)
                
                # Article details
                with col2:
                    st.subheader(article["title"])
                    st.write(f"**Source:** {article['source']}")
                    st.write(f"**Date:** {article['date']}")
                    st.write(article["snippet"])
                    st.markdown(f"[Read more]({article['link']})")
                st.markdown("---")
        else:
            st.warning("No news articles found or an error occurred. Try a different keyword.")

if __name__ == "__main__":
    main()