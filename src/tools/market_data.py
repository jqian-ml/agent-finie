"""Market data tools using yfinance."""

import yfinance as yf
import pandas as pd
from datetime import datetime
from langchain_core.tools import tool
import time

# Add retry logic for rate limiting
def retry_with_backoff(func, max_retries=3, initial_delay=1):
    """Retry function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = initial_delay * (2 ** attempt)
            time.sleep(delay)
    return None


@tool
def get_stock_price(ticker: str, period: str = "1mo") -> str:
    """
    Get stock price data for a given ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        period: Time period - valid values: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max
    
    Returns:
        Formatted string with price data and key metrics
    """
    try:
        # Add delay to avoid rate limiting
        time.sleep(0.5)
        
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        stock.get_earnings_history
        
        if hist.empty:
            return f"No price data found for {ticker}. The ticker may be invalid or delisted."
        
        # Return complete historical data
        result = f"Stock Price Data for {ticker} (Period: {period}):\n\n{hist.to_string()}"
        
        return result
        
    except Exception as e:
        return f"Error fetching data for {ticker}: {str(e)}\nThis could be due to rate limiting or invalid ticker. Try again in a moment."


@tool
def get_fundamental_metrics(ticker: str) -> str:
    """
    Get fundamental metrics for a stock.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
    
    Returns:
        Formatted string with fundamental metrics
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        result = f"""Fundamental Metrics for {ticker}: {info}"""
        return result
        
    except Exception as e:
        return f"Error fetching fundamentals for {ticker}: {str(e)}"

@tool
def get_company_news(ticker: str, days_back: int = 7) -> str:
    """
    Get recent news for a company (from yfinance).
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        days_back: How many days to look back for news (default: 7 days)
    
    Returns:
        Formatted string with recent news headlines
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if not news:
            return f"No recent news found for {ticker}"
        
        output = f"Recent News for {ticker}:\n\n"
        
        for i, article in enumerate(news, 1):
            title = article.get('title', 'No title')
            summary = article.get('summary', 'No summary')
            pubdate = article.get('pubDate', None)
            link = article.get('clickThroughUrl', {}).get('url', 'No link')
            
            output += f"{i}. {title}\n"
            output += f"   Summary: {summary}\n"
            output += f"   Date: {pubdate}\n"
            output += f"   Link: {link}\n\n"
        
        return output
        
    except Exception as e:
        return f"Error fetching news for {ticker}: {str(e)}"


@tool
def get_earnings_data(ticker: str) -> str:
    """
    Get earnings data including quarterly results, earnings dates, and surprises.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
    
    Returns:
        Formatted string with earnings data
    """
    try:
        stock = yf.Ticker(ticker)
        
        output = f"Earnings Data for {ticker}:\n\n"
        
        # Get earnings dates
        try:
            earnings_dates = stock.earnings_dates
            if earnings_dates is not None and not earnings_dates.empty:
                output += "Recent Earnings Dates:\n"
                # Get last 4 quarters
                for i, (date, row) in enumerate(earnings_dates.head(4).iterrows()):
                    output += f"  {date.strftime('%Y-%m-%d')}: "
                    eps_estimate = row.get('EPS Estimate', 'N/A')
                    eps_actual = row.get('Reported EPS', 'N/A')
                    if eps_estimate != 'N/A' and eps_actual != 'N/A':
                        surprise = ((eps_actual - eps_estimate) / eps_estimate) * 100 if eps_estimate != 0 else 0
                        output += f"EPS ${eps_actual:.2f} vs Est ${eps_estimate:.2f} (Surprise: {surprise:+.1f}%)\n"
                    else:
                        output += f"EPS {eps_actual}\n"
                output += "\n"
        except Exception as e:
            output += f"Could not fetch earnings dates: {str(e)}\n\n"
        
        # Get quarterly earnings (using income_stmt to avoid deprecation warning)
        try:
            # Use quarterly_income_stmt instead of deprecated quarterly_earnings
            income_stmt = stock.quarterly_income_stmt
            if income_stmt is not None and not income_stmt.empty:
                output += "Quarterly Net Income (Recent 4 quarters):\n"
                # Get Net Income row if available
                if 'Net Income' in income_stmt.index:
                    net_income = income_stmt.loc['Net Income']
                    for i, (date, value) in enumerate(net_income.head(4).items()):
                        if pd.notna(value):
                            output += f"  Q{i+1} ({date.strftime('%Y-%m-%d')}): ${value/1e9:.2f}B\n"
                        else:
                            output += f"  Q{i+1}: N/A\n"
                    output += "\n"
        except Exception as e:
            output += f"Could not fetch quarterly income: {str(e)}\n\n"
        
        # Get earnings history
        try:
            earnings_history = stock.earnings_history
            if earnings_history is not None and not earnings_history.empty:
                output += "Earnings Surprises (Last 4 reports):\n"
                for i, (idx, row) in enumerate(earnings_history.head(4).iterrows()):
                    quarter = row.get('Quarter', 'N/A')
                    eps_estimate = row.get('epsEstimate', 'N/A')
                    eps_actual = row.get('epsActual', 'N/A')
                    surprise_pct = row.get('surprisePercent', 'N/A')
                    output += f"  {quarter}: ${eps_actual:.2f} vs ${eps_estimate:.2f} est. "
                    output += f"({surprise_pct*100:+.1f}% surprise)\n" if surprise_pct != 'N/A' else "\n"
                output += "\n"
        except Exception as e:
            output += f"Could not fetch earnings history: {str(e)}\n\n"
        
        # Get next earnings date
        try:
            info = stock.info
            next_earnings = info.get('earningsTimestamp', None)
            if next_earnings:
                next_date = datetime.fromtimestamp(next_earnings)
                output += f"Next Earnings Date: {next_date.strftime('%Y-%m-%d')}\n"
        except:
            pass
        
        return output if len(output) > 50 else f"No earnings data available for {ticker}"
        
    except Exception as e:
        return f"Error fetching earnings data for {ticker}: {str(e)}"


# Export tools list for easy access
MARKET_DATA_TOOLS = [
    get_stock_price,
    get_fundamental_metrics,
    get_company_news,
    get_earnings_data,
]
