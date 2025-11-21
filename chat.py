import config
from graph import build_graph
from config import graph_config
from langchain_core.messages import HumanMessage

def stream_graph_updates(user_input: str):
    graph = build_graph()
    
    # 💡 CREAR la instancia de HumanMessage
    user_message = HumanMessage(content=user_input)
    
    for event in graph.stream(
        # 🟢 Pasar la lista de mensajes de LangChain
        {"messages": [user_message]}, 
        config=graph_config,
        stream_mode="values"
    ):
        event["messages"][-1].pretty_print()


def run_chat_loop():
    while True:
        try:
            user_input= input("User: ")
            if user_input.lower() in ["quit", "salir", "exit"]:
                print("Adios!!")
                break
            stream_graph_updates(user_input)
        except:
            stream_graph_updates(user_input="despidete")
            break
 