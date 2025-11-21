from dotenv import load_dotenv
from langchain_core.runnables import graph
from langchain.chat_models.base import init_chat_model
#from langchain_groq import ChatGroq
#from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

memory = MemorySaver()

# ejemplo de uso de tool
#tavily_tool = TavilySearch(max_results=2)
#tools = [tavily_tool]

#llm = init_chat_model("ChatGroq:llama-3.1-8b-instant")



llm = init_chat_model(
    model="llama-3.1-8b-instant",
    model_provider="groq"
)

#llm_with_tools = llm.bind_tools(tools)

graph_config = {"configurable":{"thread_id": "user-1"}}
