import os
import io
import json
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import wikipedia
import wolframalpha
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from duckduckgo_search import DDGS
from langchain_experimental.utilities import PythonREPL
from pypdf import PdfReader as PyPdfReader
from pytrends.request import TrendReq

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
        
        fund_data = {
            "Category": info.get("category", "N/A"),
            "Expense Ratio": info.get("annualReportExpenseRatio", "N/A"),
            "YTD Return": info.get("ytdReturn", "N/A"),
            "Beta": info.get("beta3Year", "N/A"),
            "Total Assets": info.get("totalAssets", "N/A")
        }

        holdings_str = "Top Holdings data not directly available via API."
        try:
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
# 2. FILE & DATA ANALYSIS TOOLS
# ==============================================================================

@tool
def read_csv_file(file_path: str):
    """
    Reads a CSV file from the local disk.
    Returns the first 10 rows and the list of columns.
    Useful for analyzing user-uploaded portfolios or historical data files.
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File not found at {file_path}"
        
        df = pd.read_csv(file_path)
        summary = df.head(10).to_markdown()
        columns = list(df.columns)
        stats = df.describe().to_markdown()
        
        return f"CSV FILE: {file_path}\nCOLUMNS: {columns}\n\nPREVIEW (First 10 rows):\n{summary}\n\nSTATS:\n{stats}"
    except Exception as e:
        return f"Error reading CSV: {e}"

@tool
def read_json_file(file_path: str):
    """
    Reads a JSON file from the local disk.
    Useful for reading configuration files or API responses saved locally.
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File not found at {file_path}"
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert to string and truncate if too large
        data_str = json.dumps(data, indent=2)
        return f"JSON CONTENT:\n{data_str[:5000]}..." if len(data_str) > 5000 else data_str
    except Exception as e:
        return f"Error reading JSON: {e}"

@tool
def python_interpreter(code: str):
    """
    A Python Shell. Use this to execute python commands for complex calculations, 
    projections, correlations, or data analysis that is too hard to do mentally.
    Input: Valid Python code string.
    """
    try:
        # Warning: This executes code locally. Ensure environment is sandboxed in production.
        repl = PythonREPL()
        result = repl.run(code)
        return f"CODE EXECUTION RESULT:\n{result}"
    except Exception as e:
        return f"Error executing code: {e}"

# ==============================================================================
# 3. KNOWLEDGE & FACTS TOOLS
# ==============================================================================

@tool
def get_wikipedia_summary(query: str):
    """
    Searches Wikipedia for a topic and returns a summary.
    USE THIS for: Definitions, History, or general background knowledge on a company or concept.
    """
    try:
        # Limit to 4 sentences
        return wikipedia.summary(query, sentences=4)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Wikipedia Query Ambiguous. Options: {e.options[:5]}"
    except wikipedia.exceptions.PageError:
        return "Page not found on Wikipedia."
    except Exception as e:
        return f"Error fetching Wikipedia: {e}"

@tool
def query_wolfram_alpha(query: str):
    """
    Queries Wolfram Alpha for factual or computational data.
    USE THIS for: Mathematical calculations, Physics facts, Historical economic data (GDP, Inflation), or Unit conversions.
    Requires: WOLFRAM_ALPHA_APPID in environment variables.
    """
    app_id = os.getenv("WOLFRAM_ALPHA_APPID")
    if not app_id:
        return "Error: Wolfram Alpha App ID is missing. Cannot execute query."
    
    try:
        client = wolframalpha.Client(app_id)
        res = client.query(query)
        return next(res.results).text
    except StopIteration:
        return "Wolfram Alpha could not find a direct answer."
    except Exception as e:
        return f"Error querying Wolfram Alpha: {e}"

# ==============================================================================
# 4. RESEARCH & NEWS TOOLS
# ==============================================================================

@tool
def get_market_news(query: str):
    """
    Searches for the latest financial news and headlines.
    """
    try:
        results = DDGS().text(query, max_results=4)
        if not results:
             return "No news found."
        return "\n\n".join([f"Headline: {r['title']}\nSource: {r['href']}\nSummary: {r['body']}" for r in results])
    except Exception as e:
        return f"Error fetching news: {e}"

@tool
def get_web_search(query: str):
    """
    Performs a general web search. 
    USE THIS for: Market size (TAM/SAM), Industry trends, Competitor features.
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
    """
    try:
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
# 5. SOCIAL & PROFESSIONAL TOOLS
# ==============================================================================

@tool
def get_social_media_sentiment(ticker: str):
    """
    Searches Reddit for recent discussions/sentiment regarding a specific stock.
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
def get_twitter_sentiment(ticker_or_topic: str):
    """
    Searches for recent discussions on X (Twitter) regarding a specific topic.
    Useful for: Real-time sentiment, breaking rumors, and crypto trends.
    """
    try:
        query = f"site:twitter.com {ticker_or_topic}"
        results = DDGS().text(query, max_results=5)
        if not results:
             return "No Twitter discussions found."
        return "\n\n".join([f"Tweet Context: {r['title']}\nSnippet: {r['body']}" for r in results])
    except Exception as e:
        return f"Error fetching Twitter sentiment: {e}"

@tool
def check_linkedin_people(name_company: str):
    """
    Searches for a professional's background or a company's key executives on LinkedIn via public search.
    USE THIS for: Checking CEO history, finding CTO technical background, or spotting key hires.
    """
    try:
        query = f"site:linkedin.com/in/ OR site:linkedin.com/company/ {name_company}"
        results = DDGS().text(query, max_results=4)
        if not results:
             return "No LinkedIn profiles found."
        return "\n\n".join([f"Profile: {r['title']}\nSnippet: {r['body']}\nLink: {r['href']}" for r in results])
    except Exception as e:
        return f"Error searching LinkedIn: {e}"

# ==============================================================================
# 6. DEEP DIVE & ANALYTICAL TOOLS
# ==============================================================================

@tool
def scrape_website(url: str):
    """
    Reads the full text content of a specific website URL.
    USE THIS when you find a relevant link (news, SEC filing, blog) via web search.
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

        return f"CONTENT FROM {url}:\n\n" + text[:8000] + "\n\n...(content truncated)..."
        
    except Exception as e:
        return f"Error scraping {url}: {e}"

@tool
def read_online_pdf(url: str):
    """
    Downloads and reads a PDF file from a URL.
    USE THIS when the search results give you a link ending in .pdf.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        with io.BytesIO(response.content) as f:
            reader = PyPdfReader(f)
            text = ""
            for page in reader.pages[:5]: 
                text += page.extract_text() + "\n"
                
        return f"PDF CONTENT FROM {url}:\n\n{text[:8000]}\n...(truncated)..."
    except Exception as e:
        return f"Error reading PDF: {e}"

@tool
def get_google_trends(keyword: str):
    """
    Retrieves Google Trends data (Interest over time) for a keyword.
    Returns the average interest over the last 12 months.
    """
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        kw_list = [keyword]
        pytrends.build_payload(kw_list, cat=0, timeframe='today 12-m', geo='', gprop='')
        
        data = pytrends.interest_over_time()
        
        if data.empty:
            return f"No trend data found for {keyword}"
            
        avg_interest = data[keyword].mean()
        current_interest = data[keyword].iloc[-1]
        
        trend_direction = "Rising" if current_interest > avg_interest else "Falling"
        
        return (f"GOOGLE TRENDS FOR '{keyword}' (Last 12 Months):\n"
                f"Trend Direction: {trend_direction}\n"
                f"Current Interest Score: {current_interest}\n"
                f"Average Interest Score: {avg_interest:.1f}")
    except Exception as e:
        return f"Error fetching Google Trends: {e} (Note: Google sometimes rate limits this tool)"

# ==============================================================================
# TOOL MAPPING
# ==============================================================================
TOOL_MAP = {
    "get_stock_prices": get_stock_prices,
    "get_financial_statements": get_financial_statements,
    "get_etf_fund_info": get_etf_fund_info,
    "get_market_news": get_market_news,
    "get_technical_indicators": get_technical_indicators,
    "get_social_media_sentiment": get_social_media_sentiment,
    "get_twitter_sentiment": get_twitter_sentiment,
    "check_linkedin_people": check_linkedin_people,
    "get_web_search": get_web_search,
    "search_gov_data": search_gov_data,
    "get_company_profile": get_company_profile,
    "scrape_website": scrape_website,
    "read_online_pdf": read_online_pdf,
    "python_interpreter": python_interpreter,
    "get_google_trends": get_google_trends,
    "read_csv_file": read_csv_file,
    "read_json_file": read_json_file,
    "get_wikipedia_summary": get_wikipedia_summary,
    "query_wolfram_alpha": query_wolfram_alpha
}