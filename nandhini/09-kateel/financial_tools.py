import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from duckduckgo_search import DDGS

# ==============================================================================
# 1. CORE MARKET DATA TOOLS
# ==============================================================================

@tool
def get_stock_prices(ticker: str):
    """
    Retrieves the current stock price and recent history for a given ticker symbol (e.g., AAPL, MSFT, INTC).
    Returns the last 5 days of Close Price and Volume.
    """
    try:
        stock = yf.Ticker(ticker)
        # Get last 5 days history to show trend
        hist = stock.history(period="5d")
        if hist.empty:
            return f"No stock data found for {ticker}. Check the ticker symbol."
        
        # Format specifically for LLM readability
        current_price = hist['Close'].iloc[-1]
        return f"Current Price: {current_price:.2f}\n\nRecent History:\n{hist[['Close', 'Volume']].to_string()}"
    except Exception as e:
        return f"Error fetching stock prices: {e}"

@tool
def get_financial_statements(ticker: str):
    """
    Retrieves key financial ratios, balance sheet metrics, and cash flow data.
    Useful for fundamental analysis (P/E, Debt/Equity, Revenue Growth).
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract key fundamental data with fallbacks
        data = {
            "Market Cap": info.get("marketCap", "N/A"),
            "Trailing PE": info.get("trailingPE", "N/A"),
            "Forward PE": info.get("forwardPE", "N/A"),
            "Price/Book": info.get("priceToBook", "N/A"),
            "Debt/Equity": info.get("debtToEquity", "N/A"),
            "Revenue Growth": info.get("revenueGrowth", "N/A"),
            "Profit Margins": info.get("profitMargins", "N/A"),
            "Free Cashflow": info.get("freeCashflow", "N/A"),
            "Total Revenue": info.get("totalRevenue", "N/A"),
            "Net Income": info.get("netIncomeToCommon", "N/A")
        }
        return str(data)
    except Exception as e:
        return f"Error fetching financials: {e}"

@tool
def get_etf_fund_info(ticker: str):
    """
    Specifically for Mutual Funds and ETFs. 
    Retrieves Expense Ratio, Top Holdings, and Sector Weightings.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 1. Basic Info
        fund_data = {
            "Category": info.get("category", "N/A"),
            "Expense Ratio": info.get("annualReportExpenseRatio", "N/A"),
            "YTD Return": info.get("ytdReturn", "N/A"),
            "Beta": info.get("beta3Year", "N/A"),
            "Total Assets": info.get("totalAssets", "N/A")
        }

        # 2. Top Holdings (Try to fetch if available)
        holdings_str = "Top Holdings data not directly available via API."
        try:
            # yfinance often hides this in complex objects, we stick to basic info or top 10 if keys exist
            if 'holdings' in info:
                holdings_str = str(info['holdings'])
        except:
            pass

        return f"FUND DATA FOR {ticker}:\n{str(fund_data)}\n\n{holdings_str}"
    except Exception as e:
        return f"Error fetching fund data: {e}"

@tool
def get_technical_indicators(ticker: str):
    """
    Calculates technical indicators: SMA_50, SMA_200, and RSI (14-day).
    """
    try:
        stock = yf.Ticker(ticker)
        # Fetch 1 year to ensure we have enough data for 200-day MA
        hist = stock.history(period="1y") 
        if hist.empty:
            return "No history found."
        
        # 1. Moving Averages
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
        
        # 2. RSI Calculation (14-day)
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))

        # Return last 5 days
        return hist.tail(5)[['Close', 'SMA_50', 'SMA_200', 'RSI']].to_string()
    except Exception as e:
        return f"Error calculating indicators: {e}"

# ==============================================================================
# 2. RESEARCH & NEWS TOOLS
# ==============================================================================

@tool
def get_market_news(query: str):
    """
    Searches for the latest financial news and headlines regarding a specific company or market trend.
    """
    try:
        results = DDGS().text(query, max_results=4)
        if not results:
             return "No news found."
        return "\n\n".join([f"Headline: {r['title']}\nSource: {r['href']}\nSummary: {r['body']}" for r in results])
    except Exception as e:
        return f"Error fetching news: {e}"

@tool
def get_social_media_sentiment(ticker: str):
    """
    Searches social media platforms (specifically Reddit) for recent discussions 
    and retail investor sentiment regarding a specific stock ticker.
    """
    try:
        query = f"site:reddit.com {ticker} stock discussion"
        results = DDGS().text(query, max_results=5)
        if not results:
             return "No social media discussions found."
        return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Error fetching social sentiment: {e}"

@tool
def get_web_search(query: str):
    """
    Performs a general web search. 
    USE THIS for: Market size (TAM/SAM), Industry trends, Competitor features, or general fact-checking.
    """
    try:
        results = DDGS().text(query, max_results=4)
        if not results:
             return "No results found."
        return "\n\n".join([f"Title: {r['title']}\nLink: {r['href']}\nSnippet: {r['body']}" for r in results])
    except Exception as e:
        return f"Error performing web search: {e}"

@tool
def search_gov_data(query: str):
    """
    Performs a TARGETED search on U.S. Government websites (.gov).
    USE THIS for: Inflation rates (CPI), Unemployment stats, GDP data, or official SEC filings.
    Example Query: 'US inflation rate historical data', 'Unemployment rate November 2024'
    """
    try:
        # We append site:.gov to ensure we get official sources like data.gov, bls.gov, bea.gov
        targeted_query = f"site:.gov {query}"
        results = DDGS().text(targeted_query, max_results=4)
        if not results:
             return "No official government data found."
        return "\n\n".join([f"Title: {r['title']}\nLink: {r['href']}\nSnippet: {r['body']}" for r in results])
    except Exception as e:
        return f"Error performing gov search: {e}"

@tool
def get_company_profile(company_name: str):
    """
    Searches for a company's business model, revenue streams, and product lines.
    Useful for competitive analysis.
    """
    try:
        query = f"{company_name} business model revenue streams products"
        results = DDGS().text(query, max_results=3)
        if not results:
             return "No profile found."
        return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Error fetching company profile: {e}"

# ==============================================================================
# 3. DEEP DIVE TOOLS (CRITICAL FOR PLAN-EXECUTE)
# ==============================================================================

@tool
def scrape_website(url: str):
    """
    Reads the full text content of a specific website URL.
    USE THIS when you find a relevant link (news, SEC filing, blog) via web search 
    and need to read the details to answer the user's question.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Clean up HTML (remove scripts, styles, navs)
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()    

        text = soup.get_text(separator="\n")
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        # Truncate to avoid context window explosion (8000 chars is approx 2000 tokens)
        return f"CONTENT FROM {url}:\n\n" + text[:8000] + "\n\n...(content truncated)..."
        
    except Exception as e:
        return f"Error scraping {url}: {e}"

# ==============================================================================
# TOOL MAPPING
# ==============================================================================
TOOL_MAP = {
    "get_stock_prices": get_stock_prices,
    "get_financial_statements": get_financial_statements,
    "get_etf_fund_info": get_etf_fund_info,          # <--- NEW
    "get_market_news": get_market_news,
    "get_technical_indicators": get_technical_indicators,
    "get_social_media_sentiment": get_social_media_sentiment,
    "get_web_search": get_web_search,
    "search_gov_data": search_gov_data,              # <--- NEW
    "get_company_profile": get_company_profile,
    "scrape_website": scrape_website
}