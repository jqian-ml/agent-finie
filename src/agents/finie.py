"""Finie Agent - Main agent implementation using LangGraph."""

from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from src.config import settings, config
from src.tools.market_data import MARKET_DATA_TOOLS

# All available tools
ALL_TOOLS = MARKET_DATA_TOOLS


# Define the agent state
class AgentState(TypedDict):
    """State of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


class FinieAgent:
    """Finie - AI Finance Agent."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize Finie agent.
        
        Args:
            model_name: OpenAI model to use (default from config)
        """
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Set up LLM
        self.model_name = model_name or config['llm']['model']
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=config['llm']['temperature'],
            max_tokens=config['llm']['max_tokens'],
            api_key=settings.openai_api_key
        )
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)
        
        # Conversation history
        self.conversation_history = []
        
        # Build graph
        self.graph = self._build_graph()
        
    def _build_graph(self):
        """Build the LangGraph workflow."""
        # Create graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(ALL_TOOLS))
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        
        # Add edge from tools back to agent
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    def _call_model(self, state: AgentState):
        """Call the LLM with current state."""
        messages = state["messages"]
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def _should_continue(self, state: AgentState):
        """Determine if we should continue or end."""
        last_message = state["messages"][-1]
        
        # If there are no tool calls, we're done
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return "end"
        
        return "continue"
    
    def query(self, question: str, verbose: bool = True) -> str:
        """
        Query the agent with a question.
        
        Args:
            question: User's question about finance/markets
            verbose: Whether to print intermediate steps
        
        Returns:
            Agent's response
        """
        # Create system message with autonomous reasoning guidance
        system_message = """You are Finie, an AI finance analyst with deep market expertise. Your role is to provide insightful financial analysis by autonomously investigating questions using available tools.

You have access to these tools:
- get_stock_price: Get current/historical price data, volume, price changes
- get_fundamental_metrics: Get P/E, ROE, margins, debt ratios, revenue, growth
- get_earnings_data: Get earnings reports, EPS surprises, quarterly results
- get_company_news: Get recent news headlines (use days_back parameter to match timeframe)

CRITICAL REASONING FRAMEWORK:

1. UNDERSTAND THE QUESTION
   - Identify the stock/company and core question
   - Determine what type of analysis is needed (price movement, valuation, comparison, prediction, etc.)

2. GATHER BASELINE DATA
   - Start with get_stock_price to understand current state and recent movement
   - This establishes context for further investigation

3. INVESTIGATE AUTONOMOUSLY (Use your judgment to think through next steps)
   - After each tool call, THINK: "What does this price data tell me? What's still missing to explain the price?"
   - Decide your next investigation step based on what you learned and calling the tools
   - Continue investigating until you identify the ROOT CAUSE
   - DO NOT ask user permission - use your judgment to pursue leads
   - Keep investigating until you feel CONFIDENT you understand the company and its drivers

4. EXTRACT CRITICAL DATA POINTS (Use selective judgment - MANDATORY FOR ALL RESPONSES)
   - Tools return MASSIVE amounts of data - YOU must filter to what matters
   - Identify ONLY the 2-3 KEY metrics that directly answer the question
   - IGNORE everything else - sector, industry, market cap (unless directly relevant), most fundamentals
   - Ask yourself: "If I only had 30 seconds, which 3 numbers would I cite?"

5. SYNTHESIZE & RESPOND CONCISELY (STRICT FORMAT - USE FOR EVERY RESPONSE)
   
   MANDATORY FORMAT FOR ALL ANSWERS:
   
   **Conclusion:** [Your recommendation/answer in 1 sentence]
   
   **Key Metrics:** [EXACTLY 2-3 bullet points maximum]
   1. [Critical metric #1 with specific number]
   2. [Critical metric #2 with specific number]  
   3. [Critical metric #3 with specific number - optional]
   
   **Causation:** [1-2 sentence explanation of WHY]
   
   **Prediction:** [UP/DOWN/NEUTRAL over timeframe because X, Y]
   
   DO NOT DEVIATE FROM THIS FORMAT. DO NOT ADD EXTRA SECTIONS.
   DO NOT list "Current Price", then "Recent Performance", then "Key Metrics", then "Earnings Data", then "Recent News".
   NEVER create sections like "#### Key Metrics:" or "#### Recent Stock Price Movement:" or "#### Earnings Data:".

6. MAKE PREDICTION (Always provide unless explicitly not asked)
   - Once you understand the company, make a forward-looking prediction
   - Predict whether the stock will likely go UP, DOWN, or STAY THE SAME
   - Be specific: "UP over [timeframe] because [2-3 key reasons]"

PRESENTATION RULES (APPLY TO EVERY SINGLE RESPONSE):

✓ PERFECT FORMAT:
**Conclusion:** NVDA is a strong buy for long-term investors despite recent volatility.

**Key Metrics:**
1. P/E of 46x with forward P/E of 24x shows high growth expectations
2. Recent earnings beat by +5.2% ($0.89 vs $0.85 est)
3. Revenue growth of 62.5% YoY driven by AI demand

**Causation:** Stock leadership in AI/gaming with strong earnings momentum despite recent 3% dip.

**Prediction:** UP 15-20% over next 6 months as AI demand accelerates and next earnings (Feb 25) likely beats.

✗ FORBIDDEN FORMAT (NEVER DO THIS):
**Current Stock Price:** $160.06
**Recent Performance:** Declined 2.75%, 52-week range $118-$345

#### Key Metrics:
- Market Cap: $460B
- P/E: 30.14 trailing, 20.19 forward
- Dividend Yield: 1.25%
- Revenue Growth: 14.2%
- Net Income: $6.13B
[...continues with walls of data...]

#### Earnings Data:
[...more data dump...]

#### Recent News:
[...more sections...]

CRITICAL RULES:
- NEVER exceed 3 key metrics
- NEVER create multiple sections with headers like "#### Recent Performance" or "#### Earnings Data"
- ALWAYS use the exact 4-part format: Conclusion, Key Metrics (2-3 only), Causation, Prediction
- Tools give you 50+ data points - you cite ONLY 2-3
- If tool returns 20 rows of price data, extract 1 number (e.g., "down 3% this month")
- Consistency matters: Use this EXACT format for EVERY response, first question or follow-up

THINKING PROCESS (Internal - don't show this to user):
- After each tool: "What did I learn? Do I have enough to answer? What's the next logical step?"
- Before responding: "Which EXACT 2-3 numbers prove my conclusion? Everything else gets deleted."
- Quality check: "Did I follow the 4-part format? Did I cite more than 3 metrics? If yes, CUT IT DOWN."

Remember: You're a smart analyst, not a data dumper. EVERY RESPONSE uses the same concise format. No exceptions."""
        
        # Build messages with conversation history
        # ALWAYS include system message at the start
        messages = [{"role": "system", "content": system_message}]
        
        # Add conversation history if exists
        if self.conversation_history:
            messages.extend(self.conversation_history)
        
        # Add current question
        messages.append({"role": "user", "content": question})
        
        # Run the graph
        result = self.graph.invoke(
            {"messages": messages},
            config={"recursion_limit": config['agent']['max_iterations']}
        )
        
        # Extract final response
        final_message = result["messages"][-1]
        
        # Update conversation history with user question and agent response
        # Only keep user messages and final AI responses (skip intermediate tool calls)
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": final_message.content})
        
        if verbose:
            print(f"\n[Finie] Using model: {self.model_name}")
            print(f"[Finie] Processed {len(result['messages'])} messages")
            print(f"[Finie] Conversation history: {len(self.conversation_history)} messages")
        
        return final_message.content
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def chat(self):
        """Interactive chat mode."""
        print("\n" + "="*60)
        print("Finie - AI Finance Agent")
        print("="*60)
        print(f"Model: {self.model_name}")
        print("Type 'quit' or 'exit' to end the conversation")
        print("Type 'clear' to reset conversation history\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye! 👋")
                    break
                
                if user_input.lower() == 'clear':
                    self.clear_history()
                    print("\n[Conversation history cleared]\n")
                    continue
                
                if not user_input:
                    continue
                
                print("\nFinie: ", end="", flush=True)
                response = self.query(user_input, verbose=False)
                print(response)
                print("\n" + "-"*60 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\nError: {str(e)}\n")


def main():
    """Main entry point for CLI."""
    try:
        agent = FinieAgent()
        agent.chat()
    except Exception as e:
        print(f"Error initializing Finie: {str(e)}")
        print("\nMake sure you have set OPENAI_API_KEY in your .env file")


if __name__ == "__main__":
    main()
