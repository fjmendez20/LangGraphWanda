from langgraph import graph
from langgraph.graph import StateGraph, START, END
#from langgraph.prebuilt import ToolNode, tools_condition
from config import  memory, llm
from state import State
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

# --- Definición de la Lógica y la Cadena ---

# 1. Definir el prompt de sistema
SYSTEM_PROMPT = (
    "Eres FabiBot, un asistente amigable y útil, experto en tecnología y programación. "
    "Tu objetivo es responder a las preguntas del usuario. Responde de forma clara, "
    "concisa, y siempre en español. Mantén el contexto de la conversación."
)

# 2. Crear el template del prompt
# El placeholder '{messages}' indica dónde se insertará la lista de mensajes del estado
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("placeholder", "{messages}"), 
    ]
)

# 3. Encadenar el Prompt con el LLM
# Ahora, 'llm_with_prompt' es la cadena que recibe el estado y produce la respuesta.
llm_with_prompt = prompt | llm


def build_graph() -> StateGraph:
    graph_builder = StateGraph(State)

    def chatbot(state: State):
        """
        Nodo que invoca la cadena (Prompt + LLM) para generar la respuesta.
        """
        # 4. Usar la cadena encadenada (prompt | llm) y pasar el estado COMPLETO
        # La cadena mapeará automáticamente state["messages"] al placeholder {messages}
        message = llm_with_prompt.invoke(state) 
        
        # El resultado es un AIMessage que se añade al historial.
        return {"messages": [message]}
    
    graph_builder.add_node("nodo1", chatbot)
    
    #crear nodo de tools
    #tool_node = ToolNode(tools)
    #graph_builder.add_node("tools", tool_node)
    
    #usar nodo de tools (enrutar el nodo1 al toolnode)
    #graph_builder.add_conditional_edges("nodo1", tools_condition)

    #devolver el toolnode a nodo1 (LLM)
    #graph_builder.add_edge("tools", "nodo1")

    #arista del inicio 
    graph_builder.add_edge(START, "nodo1")

    return graph_builder.compile(checkpointer=memory)
    #return graph_builder.compile()

#verificar que el graph se compile correctamente con langgraph-cli
graph = build_graph()
