import os
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from crewai import Agent, Task, Crew
from langchain.llms import Groq
import time

# Set up Groq LLM
# You'll need to set your Groq API key in your environment variables
# or you can set it directly (not recommended for production)
# os.environ["GROQ_API_KEY"] = "your-api-key"
groq_llm = Groq(model="llama3-8b-8192")  # Using Llama 3 model through Groq

# Sample user credentials (in a real app, use a secure database)
USERS = {
    "user1": "password1",
    "user2": "password2"
}

def search_stock_by_name(stock_name):
    """Search for a stock ticker symbol by company name."""
    try:
        # This is a simple approach - in a real app, you might use a more comprehensive search
        tickers = yf.Tickers(stock_name)
        if hasattr(tickers, 'tickers') and len(tickers.tickers) > 0:
            for ticker_symbol in tickers.tickers:
                ticker = yf.Ticker(ticker_symbol)
                info = ticker.info
                if 'longName' in info and stock_name.lower() in info['longName'].lower():
                    return ticker_symbol
        
        # If the above approach fails, try using yfinance's search function
        search_result = yf.Ticker(stock_name)
        if hasattr(search_result, 'info') and 'symbol' in search_result.info:
            return search_result.info['symbol']
        
        return None
    except Exception as e:
        st.error(f"Error searching for stock: {str(e)}")
        return None

# Helper functions for yfinance data analysis
def yf_technical_analysis(stock_identifier):
    """Perform technical analysis on a stock using yfinance."""
    try:
        # Identify if we have a name or symbol
        stock_symbol = stock_identifier
        if not stock_identifier.isupper() or len(stock_identifier.split()) > 1:
            found_symbol = search_stock_by_name(stock_identifier)
            if found_symbol:
                stock_symbol = found_symbol
            else:
                return f"Could not find a stock symbol for '{stock_identifier}'"
        
        # Get stock data
        stock = yf.Ticker(stock_symbol)
        hist = stock.history(period="1y")
        
        if hist.empty:
            return f"No historical data available for {stock_symbol}"

        # Calculate moving averages
        hist['MA50'] = hist['Close'].rolling(window=50).mean()
        hist['MA200'] = hist['Close'].rolling(window=200).mean()

        # RSI calculation (simplified)
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))

        return {
            "symbol": stock_symbol,
            "name": stock.info.get('longName', stock_symbol),
            "price_data": hist.tail(30).to_dict(),
            "current_price": hist['Close'].iloc[-1],
            "ma50": hist['MA50'].iloc[-1],
            "ma200": hist['MA200'].iloc[-1],
            "rsi": hist['RSI'].iloc[-1],
            "ma_signal": "bullish" if hist['MA50'].iloc[-1] > hist['MA200'].iloc[-1] else "bearish",
            "volume": hist['Volume'].iloc[-1]
        }
    except Exception as e:
        return f"Error in technical analysis: {str(e)}"

def yf_fundamental_analysis(stock_identifier):
    """Perform fundamental analysis on a stock using yfinance."""
    try:
        # Identify if we have a name or symbol
        stock_symbol = stock_identifier
        if not stock_identifier.isupper() or len(stock_identifier.split()) > 1:
            found_symbol = search_stock_by_name(stock_identifier)
            if found_symbol:
                stock_symbol = found_symbol
            else:
                return f"Could not find a stock symbol for '{stock_identifier}'"
        
        # Get stock info
        stock = yf.Ticker(stock_symbol)
        info = stock.info

        # Extract key metrics
        metrics = {
            "symbol": stock_symbol,
            "name": info.get('longName', stock_symbol),
            "sector": info.get('sector', 'N/A'),
            "industry": info.get('industry', 'N/A'),
            "market_cap": info.get('marketCap', 'N/A'),
            "pe_ratio": info.get('trailingPE', 'N/A'),
            "forward_pe": info.get('forwardPE', 'N/A'),
            "dividend_yield": info.get('dividendYield', 'N/A'),
            "eps": info.get('trailingEps', 'N/A'),
            "beta": info.get('beta', 'N/A'),
            "52_week_high": info.get('fiftyTwoWeekHigh', 'N/A'),
            "52_week_low": info.get('fiftyTwoWeekLow', 'N/A')
        }

        # Get financial statements summary (if available)
        try:
            balance_sheet = stock.balance_sheet.iloc[:, 0].to_dict() if not stock.balance_sheet.empty else {}
            income_stmt = stock.income_stmt.iloc[:, 0].to_dict() if not stock.income_stmt.empty else {}

            financials = {
                "total_assets": balance_sheet.get('Total Assets', 'N/A'),
                "total_debt": balance_sheet.get('Total Debt', 'N/A'),
                "total_revenue": income_stmt.get('Total Revenue', 'N/A'),
                "gross_profit": income_stmt.get('Gross Profit', 'N/A'),
                "net_income": income_stmt.get('Net Income', 'N/A')
            }
            metrics.update(financials)
        except:
            pass

        return metrics
    except Exception as e:
        return f"Error in fundamental analysis: {str(e)}"

def competitor_analysis(stock_identifier):
    """Analyze competitors of a given stock."""
    try:
        # Identify if we have a name or symbol
        stock_symbol = stock_identifier
        if not stock_identifier.isupper() or len(stock_identifier.split()) > 1:
            found_symbol = search_stock_by_name(stock_identifier)
            if found_symbol:
                stock_symbol = found_symbol
            else:
                return f"Could not find a stock symbol for '{stock_identifier}'"
        
        stock = yf.Ticker(stock_symbol)
        info = stock.info

        # Get sector and industry
        sector = info.get('sector', '')
        industry = info.get('industry', '')

        # Find real competitors instead of using hardcoded lists
        competitors = []
        
        # If we have sector and industry, try to find companies in the same industry
        if sector and industry:
            # This is a simplified approach - in a real app, you'd use a more comprehensive database
            try:
                # Get peers from Yahoo Finance if available
                if 'companyOfficers' in info:  # This is a hack to make sure we have a real company
                    peers = stock.get_recommendations()
                    if not peers.empty and len(peers) > 0:
                        symbols = peers.index.unique().tolist()
                        competitors = [s for s in symbols if s != stock_symbol][:3]
            except:
                pass
        
        # If we couldn't find competitors, use market cap to find similar sized companies
        if not competitors and 'marketCap' in info and info['marketCap']:
            try:
                market_cap = info['marketCap']
                # In a real app, you'd query a database for companies with similar market cap
                # For now, just use some well-known companies
                if market_cap > 1000000000000:  # > $1T
                    candidates = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
                elif market_cap > 100000000000:  # > $100B
                    candidates = ['TSLA', 'JPM', 'V', 'PG', 'JNJ', 'WMT']
                else:
                    candidates = ['AMD', 'UBER', 'SPOT', 'SQ', 'TWTR']
                
                competitors = [s for s in candidates if s != stock_symbol][:3]
            except:
                pass
        
        # If still no competitors, use some defaults based on sector
        if not competitors:
            if sector == 'Technology':
                competitors = ['AAPL', 'MSFT', 'GOOGL']
            elif sector == 'Financial Services':
                competitors = ['JPM', 'BAC', 'GS']
            elif sector == 'Healthcare':
                competitors = ['JNJ', 'PFE', 'UNH']
            else:
                competitors = ['AAPL', 'JPM', 'JNJ']
            
            competitors = [s for s in competitors if s != stock_symbol][:3]

        competitor_data = {}
        for symbol in competitors:
            comp = yf.Ticker(symbol)
            comp_info = comp.info
            try:
                competitor_data[symbol] = {
                    'name': comp_info.get('longName', symbol),
                    'price': comp_info.get('currentPrice', comp_info.get('regularMarketPrice', 'N/A')),
                    'market_cap': comp_info.get('marketCap', 'N/A'),
                    'pe_ratio': comp_info.get('trailingPE', 'N/A')
                }
            except:
                competitor_data[symbol] = {'name': symbol, 'error': 'Data unavailable'}

        return {
            'symbol': stock_symbol,
            'name': info.get('longName', stock_symbol),
            'sector': sector,
            'industry': industry,
            'competitors': competitor_data
        }
    except Exception as e:
        return f"Error in competitor analysis: {str(e)}"

def risk_assessment(stock_identifier):
    """Assess the risk level of a stock."""
    try:
        # Identify if we have a name or symbol
        stock_symbol = stock_identifier
        if not stock_identifier.isupper() or len(stock_identifier.split()) > 1:
            found_symbol = search_stock_by_name(stock_identifier)
            if found_symbol:
                stock_symbol = found_symbol
            else:
                return f"Could not find a stock symbol for '{stock_identifier}'"
        
        stock = yf.Ticker(stock_symbol)
        hist = stock.history(period="1y")
        
        if hist.empty:
            return f"No historical data available for {stock_symbol}"

        # Calculate volatility
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5)  # Annualized volatility

        # Get beta
        info = stock.info
        beta = info.get('beta', None)

        # Calculate drawdown
        peak = hist['Close'].cummax()
        drawdown = (hist['Close'] - peak) / peak
        max_drawdown = drawdown.min()

        return {
            'symbol': stock_symbol,
            'name': info.get('longName', stock_symbol),
            'volatility': volatility,
            'beta': beta,
            'max_drawdown': max_drawdown,
            'risk_level': 'High' if volatility > 0.3 or (beta and beta > 1.5) else
                        'Medium' if volatility > 0.15 or (beta and beta > 1) else 'Low',
            'drawdown_history': drawdown.tail(30).to_dict()
        }
    except Exception as e:
        return f"Error in risk assessment: {str(e)}"

def sentiment_analysis(stock_identifier):
    """
    Placeholder for sentiment analysis.
    In a real app, this would connect to news APIs and social media data.
    """
    try:
        # Identify if we have a name or symbol
        stock_symbol = stock_identifier
        if not stock_identifier.isupper() or len(stock_identifier.split()) > 1:
            found_symbol = search_stock_by_name(stock_identifier)
            if found_symbol:
                stock_symbol = found_symbol
            else:
                return f"Could not find a stock symbol for '{stock_identifier}'"
        
        stock = yf.Ticker(stock_symbol)
        info = stock.info
        
        # In a real application, you'd integrate with a news API and sentiment analysis service
        # For now, we'll generate some placeholder data
        
        import random
        sentiment_options = ['positive', 'neutral', 'negative', 'mixed']
        weights = [0.4, 0.3, 0.2, 0.1]  # Weighted towards positive
        sentiment = random.choices(sentiment_options, weights=weights)[0]
        
        # Score based on sentiment
        base_score = {
            'positive': random.uniform(0.65, 0.9),
            'neutral': random.uniform(0.45, 0.65),
            'negative': random.uniform(0.2, 0.45),
            'mixed': random.uniform(0.4, 0.6)
        }[sentiment]
        
        # Adjust based on recent performance
        try:
            recent_performance = stock.history(period="1m")['Close'].pct_change().mean() * 100
            if recent_performance > 5:  # If stock is up >5% in last month
                base_score = min(0.9, base_score * 1.1)
            elif recent_performance < -5:  # If stock is down >5% in last month
                base_score = max(0.1, base_score * 0.9)
        except:
            pass
            
        return {
            'symbol': stock_symbol,
            'name': info.get('longName', stock_symbol),
            'sentiment': sentiment,
            'score': round(base_score, 2),
            'sources': 'news (60%), social media (40%)',
            'note': 'This is simulated sentiment data for demonstration purposes'
        }
    except Exception as e:
        return f"Error in sentiment analysis: {str(e)}"

def analyze_stock(stock_identifier):
    """Create and run a CrewAI workflow to analyze a stock."""
    # Define Agents
    researcher = Agent(
        role='Stock Market Researcher',
        goal='Gather and analyze comprehensive data about the stock',
        backstory="You're an experienced stock market researcher with a keen eye for detail and a talent for uncovering hidden trends.",
        tools=[yf_technical_analysis, yf_fundamental_analysis, competitor_analysis],
        llm=groq_llm,
        verbose=True
    )

    analyst = Agent(
        role='Financial Analyst',
        goal='Analyze the gathered data and provide investment insights',
        backstory="You're a seasoned financial analyst known for your accurate predictions and ability to synthesize complex information.",
        tools=[yf_technical_analysis, yf_fundamental_analysis, risk_assessment],
        llm=groq_llm,
        verbose=True
    )

    sentiment_analyst = Agent(
        role='Sentiment Analyst',
        goal='Analyze market sentiment and its potential impact on the stock',
        backstory="You're an expert in behavioral finance and sentiment analysis, capable of gauging market emotions and their effects on stock performance.",
        tools=[sentiment_analysis],
        llm=groq_llm,
        verbose=True
    )

    strategist = Agent(
        role='Investment Strategist',
        goal='Develop a comprehensive investment strategy based on all available data',
        backstory="You're a renowned investment strategist known for creating tailored investment plans that balance risk and reward.",
        tools=[],
        llm=groq_llm,
        verbose=True
    )

    # Define Tasks
    research_task = Task(
        description=f"Research {stock_identifier} using advanced technical and fundamental analysis tools. Provide a comprehensive summary of key metrics, including chart patterns, financial ratios, and competitor analysis.",
        agent=researcher
    )

    sentiment_task = Task(
        description=f"Analyze the market sentiment for {stock_identifier} using news and social media data. Evaluate how current sentiment might affect the stock's performance.",
        agent=sentiment_analyst
    )

    analysis_task = Task(
        description=f"Synthesize the research data and sentiment analysis for {stock_identifier}. Conduct a thorough risk assessment and provide a detailed analysis of the stock's potential.",
        agent=analyst,
        dependencies=[research_task, sentiment_task]
    )

    strategy_task = Task(
        description=f"Based on all the gathered information about {stock_identifier}, develop a comprehensive investment strategy. Consider various scenarios and provide actionable recommendations for different investor profiles.",
        agent=strategist,
        dependencies=[analysis_task]
    )

    # Create Crew
    crew = Crew(
        agents=[researcher, analyst, sentiment_analyst, strategist],
        tasks=[research_task, sentiment_task, analysis_task, strategy_task],
        verbose=2
    )

    # Execute Crew workflow
    result = crew.kickoff()
    return result

# Streamlit UI
def main():
    st.set_page_config(page_title="Financial Stock Analysis Agent", layout="wide")

    # Session state initialization
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = ""
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'stock_input' not in st.session_state:
        st.session_state.stock_input = ""
    if 'stock_symbol' not in st.session_state:
        st.session_state.stock_symbol = ""
    if 'analysis_started' not in st.session_state:
        st.session_state.analysis_started = False

    # Application title
    st.title("Financial Stock Analysis Agent")

    # Login Form
    if not st.session_state.authenticated:
        with st.form("login_form"):
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")

            if submit_button:
                if username in USERS and USERS[username] == password:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password")

    # Main application (after login)
    else:
        st.sidebar.success(f"Logged in as {st.session_state.username}")
        st.sidebar.button("Logout", on_click=lambda: logout())

        st.subheader("Stock Analysis")

        # Stock input form
        with st.form("stock_form"):
            stock_input = st.text_input("Enter Stock Name or Symbol (e.g., Apple, AAPL, Microsoft)",
                                        value=st.session_state.stock_input)
            analyze_button = st.form_submit_button("Analyze Stock")

            if analyze_button and stock_input:
                st.session_state.stock_input = stock_input
                st.session_state.analysis_started = True
                st.session_state.analysis_result = None
                
                # Try to get a symbol if a name was entered
                if not stock_input.isupper() or len(stock_input.split()) > 1:
                    symbol = search_stock_by_name(stock_input)
                    if symbol:
                        st.session_state.stock_symbol = symbol
                    else:
                        st.session_state.stock_symbol = stock_input  # Will try again during analysis
                else:
                    st.session_state.stock_symbol = stock_input.upper()
                
                st.experimental_rerun()

        # Show stock data visualization if we have a valid stock symbol
        if st.session_state.stock_symbol:
            try:
                stock_data = yf.Ticker(st.session_state.stock_symbol)
                hist = stock_data.history(period="1y")

                if not hist.empty:
                    company_name = stock_data.info.get('longName', st.session_state.stock_symbol)
                    st.subheader(f"{company_name} ({st.session_state.stock_symbol}) - Price History (1 Year)")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=hist.index,
                        open=hist['Open'],
                        high=hist['High'],
                        low=hist['Low'],
                        close=hist['Close'],
                        name='Price Action'
                    ))
                    fig.update_layout(xaxis_rangeslider_visible=False, height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    # Basic stock info
                    info = stock_data.info
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Current Price", f"${hist['Close'].iloc[-1]:.2f}")
                    
                    with col2:
                        st.metric("Volume", f"{hist['Volume'].iloc[-1]:,}")
                    
                    with col3:
                        if 'marketCap' in info and info['marketCap']:
                            market_cap = info['marketCap'] / 1_000_000_000
                            st.metric("Market Cap", f"${market_cap:.2f}B")
                        else:
                            st.metric("Market Cap", "N/A")
                            
                    with col4:
                        if 'sector' in info and info['sector']:
                            st.metric("Sector", info['sector'])
                        else:
                            st.metric("Sector", "N/A")
            except Exception as e:
                st.warning(f"Could not load chart data: {str(e)}")

        # Run CrewAI analysis
        if st.session_state.analysis_started and not st.session_state.analysis_result:
            with st.spinner(f"Analyzing {st.session_state.stock_input}... This may take a few minutes as agents work together to provide comprehensive analysis."):
                try:
                    # Create a progress indicator
                    progress_bar = st.progress(0)
                    for i in range(100):
                        # Simulate analysis progress
                        time.sleep(0.1)
                        progress_bar.progress(i + 1)

                    result = analyze_stock(st.session_state.stock_input)
                    st.session_state.analysis_result = result
                    st.session_state.analysis_started = False
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
                    st.session_state.analysis_started = False

        # Display analysis results
        if st.session_state.analysis_result:
            st.subheader("Stock Analysis Results")
            st.markdown(st.session_state.analysis_result)

            if st.button("Clear Results"):
                st.session_state.analysis_result = None
                st.experimental_rerun()

def logout():
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.analysis_result = None
    st.session_state.stock_input = ""
    st.session_state.stock_symbol = ""
    st.session_state.analysis_started = False

if __name__ == "__main__":
    main()
