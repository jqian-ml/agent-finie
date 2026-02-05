# Finie - AI Finance Agent

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.60-green.svg)](https://github.com/langchain-ai/langgraph)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-orange.svg)](https://openai.com/)
[![yfinance](https://img.shields.io/badge/yfinance-0.2.66-yellow.svg)](https://github.com/ranaroussi/yfinance)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

An autonomous AI finance analyst that investigates stocks, explains price movements, and makes investment predictions using real-time market data.

## What Finie Does

Finie is a ReAct (Reasoning + Acting) agent that autonomously:
- ✅ Analyzes stock prices and identifies trends
- ✅ Investigates earnings reports and surprises
- ✅ Correlates news events with price movements
- ✅ Evaluates fundamentals (P/E, growth, margins, debt)
- ✅ Makes forward-looking predictions (UP/DOWN/NEUTRAL)
- ✅ Maintains conversation context across multiple questions

**Key Feature**: Instead of dumping data, Finie selects the 2-3 critical metrics that matter and explains causation, not just correlation.

## How It Works

Finie uses a **ReAct (Reasoning + Acting) loop**:

1. **Understand** → Identifies what you're asking
2. **Investigate** → Autonomously calls tools to gather data
   - Checks stock price → Sees -5% drop
   - Checks earnings → Finds earnings miss
   - Checks news → Discovers product delay announcement
3. **Synthesize** → Extracts only critical metrics (2-3 data points)
4. **Predict** → Makes forward-looking call with reasoning
5. **Remember** → Maintains conversation context for follow-ups

Example flow:
```
You: "Should I buy NVDA?"
Finie: [Calls get_stock_price] → [Calls get_earnings_data] → [Calls get_fundamental_metrics] → [Calls get_company_news]
Finie: "Strong buy. P/E 46x with 62% revenue growth. Recent earnings beat by 5%. Prediction: UP 15-20% over 6 months."
```

## Tech Stack

- **LLM**: OpenAI GPT-4o-mini (primary), GPT-4o (complex reasoning)
- **Agent Framework**: LangGraph
- **Data Sources**: yfinance, Alpha Vantage, SEC EDGAR
- **Vector Store**: ChromaDB (ltemperature 0.1 for consistent analysis)
- **Agent Framework**: LangGraph 0.2.60 (ReAct pattern with state management)
- **Data Source**: yfinance 0.2.66 (real-time market data)
- **Tools**: 5 market data tools (price, fundamentals, earnings, news, comparison)
- *Quick Start

### Prerequisites
- Python 3.13+
- OpenAI API key

### Installation

```bash
# 1. Clone and setup
git clone https://github.com/ml-for-fun/agent-finie.git
cd agent-finie
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# 2. Add your OpenAI API key
cp .env.example .env
# Edit .env: OPENAI_API_KEY=sk-...

# 3. Run
python main

## Usage

### Simple Query Example

```python
from src.agents.finie import FinieAgent

# Initialize the agent
agent = FinieAgent()

# Ask a question
response = agent.query("Why did ORCL stock drop after the OpenAI announcement?")
print(response)
```

###Example Conversations

```
You: should I invest in nvidia now

Finie: 
**Conclusion:** Investing in NVIDIA now appears to be a strong opportunity for long-term growth.

**Key Metrics:**
1. Current price $185.61, down 1.3% recently but up 61% over past year
2. P/E of 46x with forward P/E of 24x shows high growth expectations
3. Recent earnings beat by +5.2% ($0.89 vs $0.85 est)

**Causation:** NVIDIA's leadership in AI/gaming with 62.5% revenue growth and strong 
earnings momentum despite recent volatility.

**Prediction:** UP 15-20% over next 6 months as AI demand accelerates and next earnings 
(Feb 25) likely beats expectations.
```


## What's Next

**Potential enhancements**:
- Add comparison with sector averages for context
- Integrate real earnings call transcripts (currently no RAG)
- Add technical analysis indicators (RSI, MACD, moving averages)
- Implement backtesting to validate predictions
- Add portfolio analysis capabilities

## Disclaimer

⚠️ **Important**: Finie is for educational and research purposes only. It does not provide official financial advice. Always consult with a qualified financial advisor before making investment decisions.
