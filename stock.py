import os
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from crewai import Agent, Task, Crew
from langchain.llms import Groq
import time

# Sample user credentials (in a real app, use a secure database)
USERS = {
    "user1": "password1",
    "user2": "password2"
}

# Helper functions for yfinance data analysis
def yf_technical_analysis(stock_symbol):
    """Perform technical analysis on a stock using yfinance."""
    try:
        # Get stock data
        stock = yf.Ticker(stock_symbol)
        hist = stock.history(period="1y")
        
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

def yf_fundamental_analysis(stock_symbol):
    """Perform fundamental analysis on a stock using yfinance."""
    try:
        # Get stock info
        stock = yf.Ticker(stock_symbol)
        info = stock.info
        
        # Extract key metrics
        metrics = {
            "name": info.get('longName', 'N/A'),
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
            balance_sheet = stock.balance_sheet.iloc[:, 0].to_dict()
            income_stmt = stock.income_stmt.iloc[:, 0].to_dict()
            
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

def competitor_analysis(stock_symbol):
    """Analyze competitors of a given stock."""
    try:
        stock = yf.Ticker(stock_symbol)
        info = stock.info
        
        # Get sector and industry
        sector = info.get('sector', '')
        industry = info.get('industry', '')
        
        # Use a simplified approach to find competitors
        # In a real app, you might use a more sophisticated method
        if 'symbol' in info:
            symbols = [info['symbol']]
            if 'quoteType' in info and info['quoteType'] == 'EQUITY':
                # For simplicity, let's just use some major stocks in different sectors
                tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
                finance_stocks = ['JPM', 'BAC', 'GS', 'MS']
                healthcare_stocks = ['JNJ', 'PFE', 'UNH', 'MRK']
                
                if sector == 'Technology':
                    competitors = [s for s in tech_stocks if s != stock_symbol][:3]
                elif sector == 'Financial Services':
                    competitors = [s for s in finance_stocks if s != stock_symbol][:3]
                elif sector == 'Healthcare':
                    competitors = [s for s in healthcare_stocks if s != stock_symbol][:3]
                else:
                    competitors = ['AAPL', 'JPM', 'JNJ'][:3]  # Default competitors
                
                competitor_data = {}
                for symbol in competitors:
                    comp = yf.Ticker(symbol)
                    comp_info = comp.info
                    try:
                        competitor_data[symbol] = {
                            'name': comp_info.get('longName', 'N/A'),
                            'price': comp_info.get('currentPrice', 'N/A'),
                            'market_cap': comp_info.get('marketCap', 'N/A'),
                            'pe_ratio': comp_info.get('trailingPE', 'N/A')
                        }
                    except:
                        competitor_data[symbol] = {'name': symbol, 'error': 'Data unavailable'}
                
                return {
                    'sector': sector,
                    'industry': industry,
                    'competitors': competitor_data
                }
    except Exception as e:
        return f"Error in competitor analysis: {str(e)}"

def risk_assessment(stock_symbol):
    """Assess the risk level of a stock."""
    try:
        stock = yf.Ticker(stock_symbol)
        hist = stock.history(period="1y")
        
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
            'volatility': volatility,
            'beta': beta,
            'max_drawdown': max_drawdown,
            'risk_level': 'High' if volatility > 0.3 or (beta and beta > 1.5) else 
                        'Medium' if volatility > 0.15 or (beta and beta > 1) else 'Low',
            'drawdown_history': drawdown.tail(30).to_dict()
        }
    except Exception as e:
        return f"Error in risk assessment: {str(e)}"

def sentiment_analysis(stock_symbol):
    """
    Mock sentiment analysis function.
    In a real app, this would connect to news APIs and social media data.
    """
    sentiment_data = {
        'AAPL': {'sentiment': 'positive', 'score': 0.75, 'sources': 'news (65%), social media (35%)'},
        'MSFT': {'sentiment': 'positive', 'score': 0.68, 'sources': 'news (70%), social media (30%)'},
        'GOOGL': {'sentiment': 'neutral', 'score': 0.52, 'sources': 'news (55%), social media (45%)'},
        'AMZN': {'sentiment': 'positive', 'score': 0.71, 'sources': 'news (60%), social media (40%)'},
        'META': {'sentiment': 'mixed', 'score': 0.49, 'sources': 'news (50%), social media (50%)'},
        'TSLA': {'sentiment': 'volatile', 'score': 0.60, 'sources': 'news (40%), social media (60%)'},
        'JPM': {'sentiment': 'neutral', 'score': 0.55, 'sources': 'news (80%), social media (20%)'},
        'JNJ': {'sentiment': 'positive', 'score': 0.65, 'sources': 'news (75%), social media (25%)'},
    }
    
    # Return sentiment data if available, otherwise generate mock data
    if stock_symbol in sentiment_data:
        return sentiment_data[stock_symbol]
    else:
        import random
        sentiments = ['positive', 'neutral', 'negative', 'mixed']
        scores = [round(random.uniform(0.3, 0.8), 2) for _ in range(4)]
        return {
            'sentiment': random.choice(sentiments),
            'score': random.uniform(0.3, 0.8),
            'sources': 'news (60%), social media (40%)',
            'note': 'Limited data available for this stock'
        }

def analyze_stock(stock_symbol, llm_instance):
    """Create and run a CrewAI workflow to analyze a stock."""
    # Define Agents
    researcher = Agent(
        role='Stock Market Researcher',
        goal='Gather and analyze comprehensive data about the stock',
        backstory="You're an experienced stock market researcher with a keen eye for detail and a talent for uncovering hidden trends.",
        tools=[yf_technical_analysis, yf_fundamental_analysis, competitor_analysis],
        llm=llm_instance,
        verbose=True
    )
    
    analyst = Agent(
        role='Financial Analyst',
        goal='Analyze the gathered data and provide investment insights',
        backstory="You're a seasoned financial analyst known for your accurate predictions and ability to synthesize complex information.",
        tools=[yf_technical_analysis, yf_fundamental_analysis, risk_assessment],
        llm=llm_instance,
        verbose=True
    )
    
    sentiment_analyst = Agent(
        role='Sentiment Analyst',
        goal='Analyze market sentiment and its potential impact on the stock',
        backstory="You're an expert in behavioral finance and sentiment analysis, capable of gauging market emotions and their effects on stock performance.",
        tools=[sentiment_analysis],
        llm=llm_instance,
        verbose=True
    )
    
    strategist = Agent(
        role='Investment Strategist',
        goal='Develop a comprehensive investment strategy based on all available data',
        backstory="You're a renowned investment strategist known for creating tailored investment plans that balance risk and reward.",
        tools=[],
        llm=llm_instance,
        verbose=True
    )
    
    # Define Tasks
    research_task = Task(
        description=f"Research {stock_symbol} using advanced technical and fundamental analysis tools. Provide a comprehensive summary of key metrics, including chart patterns, financial ratios, and competitor analysis.",
        agent=researcher
    )
    
    sentiment_task = Task(
        description=f"Analyze the market sentiment for {stock_symbol} using news and social media data. Evaluate how current sentiment might affect the stock's performance.",
        agent=sentiment_analyst
    )
    
    analysis_task = Task(
        description=f"Synthesize the research data and sentiment analysis for {stock_symbol}. Conduct a thorough risk assessment and provide a detailed analysis of the stock's potential.",
        agent=analyst,
        dependencies=[research_task, sentiment_task]
    )
    
    strategy_task = Task(
        description=f"Based on all the gathered information about {stock_symbol}, develop a comprehensive investment strategy. Consider various scenarios and provide actionable recommendations for different investor profiles.",
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
    if 'stock_symbol' not in st.session_state:
        st.session_state.stock_symbol = ""
    if 'analysis_started' not in st.session_state:
        st.session_state.analysis_started = False
    if 'groq_api_key' not in st.session_state:
        st.session_state.groq_api_key = ""
    if 'groq_model' not in st.session_state:
        st.session_state.groq_model = "llama3-70b-8192"
        
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
        
        # Groq API key input
        with st.sidebar.expander("Groq API Settings", expanded=True):
            groq_api_key = st.text_input(
                "Enter Groq API Key", 
                value=st.session_state.groq_api_key,
                type="password"
            )
            groq_model = st.selectbox(
                "Select Groq Model",
                ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"],
                index=0
            )
            
            if st.button("Save API Settings"):
                st.session_state.groq_api_key = groq_api_key
                st.session_state.groq_model = groq_model
                if groq_api_key:
                    st.success("API settings saved!")
                else:
                    st.warning("Please enter your Groq API key")
        
        st.subheader("Stock Analysis")
        
        # Stock input form
        with st.form("stock_form"):
            stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT)", 
                                        value=st.session_state.stock_symbol)
            analyze_button = st.form_submit_button("Analyze Stock")
            
            if analyze_button and stock_symbol:
                if not st.session_state.groq_api_key:
                    st.error("Please enter your Groq API key in the sidebar first!")
                else:
                    st.session_state.stock_symbol = stock_symbol.upper()
                    st.session_state.analysis_started = True
                    st.session_state.analysis_result = None
                    st.experimental_rerun()
        
        # Show stock data visualization
        if st.session_state.stock_symbol:
            try:
                stock_data = yf.Ticker(st.session_state.stock_symbol)
                hist = stock_data.history(period="1y")
                
                if not hist.empty:
                    st.subheader(f"{st.session_state.stock_symbol} - Price History (1 Year)")
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
                    col1, col2, col3 = st.columns(3)
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
            except Exception as e:
                st.error(f"Error loading stock data: {str(e)}")
        
        # Run CrewAI analysis
        if st.session_state.analysis_started and not st.session_state.analysis_result:
            with st.spinner(f"Analyzing {st.session_state.stock_symbol}... This may take a few minutes as agents work together to provide comprehensive analysis."):
                try:
                    # Check if Groq API key is set
                    if not st.session_state.groq_api_key:
                        st.error("Please enter your Groq API key in the sidebar!")
                        st.session_state.analysis_started = False
                    else:
                        # Initialize Groq LLM
                        os.environ["GROQ_API_KEY"] = st.session_state.groq_api_key
                        llm_instance = Groq(model_name=st.session_state.groq_model)
                        
                        # Create a progress indicator
                        progress_bar = st.progress(0)
                        for i in range(100):
                            # Simulate analysis progress
                            time.sleep(0.1)
                            progress_bar.progress(i + 1)
                            
                        result = analyze_stock(st.session_state.stock_symbol, llm_instance)
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
    st.session_state.stock_symbol = ""
    st.session_state.analysis_started = False
    st.session_state.groq_api_key = ""

if __name__ == "__main__":
    main()
