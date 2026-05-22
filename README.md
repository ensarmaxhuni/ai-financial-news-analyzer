# AI Financial News Analyzer

A professional AI-powered financial news intelligence app built with **Python**, **Streamlit**, **RSS feeds**, **sentiment analysis**, and optional **OpenAI executive summaries**.

## Live Demo

Coming soon after deployment.

## Project Overview

The AI Financial News Analyzer helps users monitor financial news, analyze sentiment, identify key market themes, and generate a professional market intelligence brief.

This project is designed as part of an AI portfolio focused on:

- Generative AI
- Financial analytics
- Business intelligence
- NLP
- Market sentiment
- AI-powered decision support

## Key Features

- Search financial news by topic, ticker, company, ETF, or macroeconomic theme
- Fetch live news from Google News RSS and financial RSS sources
- Analyze article sentiment using VADER NLP
- Classify articles as bullish, bearish, or neutral
- Extract tracked financial keywords
- Generate market intelligence briefs
- Optional OpenAI executive summary if an API key is configured
- Download a markdown market intelligence report
- Streamlit UI ready for deployment

## Example Topics

- Nvidia
- AI stocks
- S&P 500
- Bitcoin
- Inflation
- Federal Reserve
- Tesla earnings
- Oil prices
- ETFs

## Tech Stack

- Python
- Streamlit
- Pandas
- Feedparser
- VADER Sentiment
- OpenAI API optional
- RSS news feeds

## Project Structure

```text
ai-financial-news-analyzer/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── LICENSE
└── sample_report.md
```

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/ensarmaxhuni/ai-financial-news-analyzer.git
cd ai-financial-news-analyzer
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

## Optional OpenAI Setup

The app works without an OpenAI API key by using a rule-based market intelligence summary.

To enable AI-generated executive summaries:

1. Create a `.env` file locally.
2. Add:

```text
OPENAI_API_KEY=your_api_key_here
```

For Streamlit Cloud deployment, add the key in app settings under secrets.

## Important Disclaimer

This application is for educational, portfolio, and market research purposes only.  
It does **not** provide investment advice, trading recommendations, or financial planning guidance.

## Portfolio Purpose

This project demonstrates:

- Financial news analysis
- NLP sentiment classification
- AI-assisted market intelligence
- Streamlit app development
- RSS data ingestion
- Business-focused AI product design

## Author

**Ensar Maxhuni**  
AI & Business Intelligence Portfolio  
GitHub: https://github.com/ensarmaxhuni  
Portfolio: https://ensarmaxhuni.github.io
