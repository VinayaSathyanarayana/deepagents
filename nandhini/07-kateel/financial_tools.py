import yfinance as yf
from langchain_core.tools import tool
from duckduckgo_search import DDGS

@tool
def get_stock_prices(ticker: str):
    """Retrieves the current stock price and recent history for a given ticker symbol (e.g., AAPL, MSFT)."""
    try:
        stock = yf.Ticker(ticker)
        # Get last 5 days history to show trend
        hist = stock.history(period="5d")
        if hist.empty:
            return "No data found."
        return hist[['Close', 'Volume']].to_string()
    except Exception as e:
        return f"Error fetching stock prices: {e}"

@tool
def get_financial_statements(ticker: str):
    """Retrieves key financial ratios, balance sheet, and income statement data for fundamental analysis."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        # Extract key fundamental data
        data = {
            "Market Cap": info.get("marketCap"),
            "Trailing PE": info.get("trailingPE"),
            "Forward PE": info.get("forwardPE"),
            "Price/Book": info.get("priceToBook"),
            "Debt/Equity": info.get("debtToEquity"),
            "Revenue Growth": info.get("revenueGrowth"),
            "Profit Margins": info.get("profitMargins")
        }
        return str(data)
    except Exception as e:
        return f"Error fetching financials: {e}"

@tool
def get_market_news(query: str):
    """Searches for the latest financial news and headlines regarding a specific company or market trend."""
    try:
        results = DDGS().text(query, max_results=3)
        return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Error fetching news: {e}"

@tool
def get_technical_indicators(ticker: str):
    """Calculates basic technical indicators (SMA) based on historical data."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        return hist.tail(5)[['Close', 'SMA_50']].to_string()
    except Exception as e:
        return f"Error calculating indicators: {e}"

# Master Dictionary to map YAML strings to actual functions
TOOL_MAP = {
    "get_stock_prices": get_stock_prices,
    "get_financial_statements": get_financial_statements,
    "get_market_news": get_market_news,
    "get_technical_indicators": get_technical_indicators
}