import gradio as gr
import os
from dotenv import load_dotenv
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

load_dotenv()

# Conversation state
chat_history = []

# Step-by-step selections
selections = {
    "ticker": None,
    "analysis_date": None,
    "analysts": [],
    "research_depth": None,
    "shallow_thinker": None,
    "deep_thinker": None,
}

steps = [
    "ticker",
    "analysis_date",
    "analysts",
    "research_depth",
    "shallow_thinker",
    "deep_thinker",
    "done"
]

step_descriptions = {
    "ticker": "📈 What stock or token would you like to analyze? (e.g., TSLA, BTC, ETH)",
    "analysis_date": "📅 What is the analysis date? (YYYY-MM-DD, or type 'today')",
    "analysts": "👥 Which agents do you want to include? (Choose any of: market, social, news, fundamentals)",
    "research_depth": "🔍 Select research depth (1 = shallow, 2 = moderate, 3 = deep)",
    "shallow_thinker": "🤖 Choose model for quick-thinking agents (e.g., gpt-3.5-turbo)",
    "deep_thinker": "🧠 Choose model for deep-thinking agents (e.g., gpt-4)",
}

current_step = 0

def chatbot(user_input):
    global current_step, selections

    if current_step >= len(steps):
        return "❌ All steps completed. Please refresh to start again."

    step = steps[current_step]

    # Step logic
    if step == "ticker":
        selections["ticker"] = user_input.strip().upper()
    elif step == "analysis_date":
        if user_input.strip().lower() == "today":
            import datetime
            selections["analysis_date"] = datetime.datetime.now().strftime("%Y-%m-%d")
        else:
            selections["analysis_date"] = user_input.strip()
    elif step == "analysts":
        selections["analysts"] = [a.strip().lower() for a in user_input.split(",")]
    elif step == "research_depth":
        selections["research_depth"] = int(user_input.strip())
    elif step == "shallow_thinker":
        selections["shallow_thinker"] = user_input.strip()
    elif step == "deep_thinker":
        selections["deep_thinker"] = user_input.strip()

    current_step += 1

    if current_step < len(steps) - 1:
        return step_descriptions[steps[current_step]]

    # === Final step: run TradingAgentsGraph ===
    try:
        config = DEFAULT_CONFIG.copy()
        config["max_debate_rounds"] = selections["research_depth"]
        config["max_risk_discuss_rounds"] = selections["research_depth"]
        config["quick_think_llm"] = selections["shallow_thinker"]
        config["deep_think_llm"] = selections["deep_thinker"]

        graph = TradingAgentsGraph(selections["analysts"], config=config, debug=False)

        init_state = graph.propagator.create_initial_state(
            selections["ticker"], selections["analysis_date"]
        )
        args = graph.propagator.get_graph_args()
        trace = list(graph.graph.stream(init_state, **args))
        final = trace[-1]

        result = final.get("final_trade_decision", "No final decision found.")
        decision = graph.process_signal(result)

        return f"✅ Final Trading Decision for {selections['ticker']}:\n\n{result}\n\n📌 Recommendation: {decision}"
    except Exception as e:
        return f"❌ Error during analysis: {str(e)}"

# Launch chatbot
gr.ChatInterface(fn=chatbot, title="TradingAgents: Full Multi-Agent Trading Bot",
                 description="Answer step-by-step to receive a complete multi-agent investment analysis.").launch()
