from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class State(TypedDict):
    """
    Representa el estado del grafo de chat.

    messages: Una lista de BaseMessages que rastrea el historial de conversación.
              El operador 'add_messages' asegura que los mensajes nuevos se
              añadan a la lista, en lugar de reemplazarla.
    """
    messages: Annotated[List[BaseMessage], add_messages]

