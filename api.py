from fastapi import FastAPI, HTTPException
# importacion de StreamingResponse para la respuestas Asyncronas
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List

# Importar el grafo compilado desde graph.py
from graph import graph 
from langchain_core.messages import HumanMessage, AIMessage

# --- Modelos de Pydantic ---
# Define la estructura de la solicitud que recibirá la API
class ChatRequest(BaseModel):
    # El mensaje de texto del usuario
    message: str 
    # El identificador de la conversación (se usa como thread_id en LangGraph)
    session_id: str 

# Define la estructura de la respuesta que enviará la API
class ChatResponse(BaseModel):
    response: str
    session_id: str


# --- Inicialización de la Aplicación ---
app = FastAPI(title="FabiBot LangGraph API")

# --- Endpoint de Conversación ---

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    
    # 1. Crear el mensaje de entrada en el formato esperado (HumanMessage)
    user_message = HumanMessage(content=request.message)
    
    # 2. Definir la configuración de la sesión (Thread ID)
    # LangGraph usará esta ID para cargar/guardar el estado en el checkpointer.
    # Usamos el 'session_id' proporcionado por el cliente.
    config = {"configurable": {"thread_id": request.session_id}}
    
    # 3. Preparar la entrada para el grafo
    # El grafo espera una estructura de estado inicial con la clave 'messages'
    input_state = {"messages": [user_message]}

    try:
        # 4. Invocar el grafo
        # Esto ejecuta el grafo desde START, pasando por nodo1 y guardando el estado
        # Retorna el estado final (un diccionario con la clave 'messages')
        final_state = graph.invoke(input_state, config=config)
        
        # 5. Extraer la respuesta del LLM
        # El último mensaje de la lista es la respuesta del agente (AIMessage)
        # Usamos .content para obtener el texto.
        ai_response = final_state["messages"][-1].content
        
    except Exception as e:
        # Manejo básico de errores de LangGraph o LLM
        raise HTTPException(status_code=500, detail=f"Error en el grafo: {e}")

    # 6. Devolver la respuesta al cliente
    return ChatResponse(
        response=ai_response,
        session_id=request.session_id
    )

async def stream_generator(session_id: str, user_message: str):
    """
    Función generadora asíncrona que produce chunks de texto.
    """
    user_msg_obj = HumanMessage(content=user_message)
    config = {"configurable": {"thread_id": session_id}}
    input_state = {"messages": [user_msg_obj]}

    try:
        # Usamos graph.stream()
        async for event in graph.stream(input_state, config=config):
            # El streaming en LangGraph devuelve eventos por cada paso.
            # Solo nos interesa el contenido de la última actualización del historial.
            
            # Verificamos si el nodo que se acaba de ejecutar fue 'nodo1' (nuestro chatbot)
            if 'nodo1' in event:
                # El evento contiene el estado del grafo en ese paso.
                messages = event['nodo1'].get("messages", [])
                
                # Buscamos el último mensaje (el chunk del AI)
                if messages:
                    last_message = messages[-1]
                    
                    # 💡 Extraemos el contenido. Como es un stream, vendrá chunk a chunk.
                    if isinstance(last_message, BaseMessage) and last_message.content:
                        # Devolvemos el texto recién generado.
                        # NOTA: Debemos codificarlo para que el cliente lo reciba correctamente.
                        yield last_message.content.encode('utf-8')
        
        # Opcional: Una vez que el stream termina, enviamos un marcador de fin
        # yield b'\\n' 

    except Exception as e:
        print(f"Error durante el streaming: {e}")
        # Manejo de errores para enviar al cliente
        yield f"ERROR: Fallo del agente ({e})".encode('utf-8')

# --- Nuevo Endpoint de Streaming ---

@app.post("/stream_chat")
async def stream_chat_endpoint(request: ChatRequest):
    """
    Endpoint que usa StreamingResponse para enviar la respuesta en tiempo real.
    """
    
    # El generador prepara y ejecuta el grafo en modo stream.
    generator = stream_generator(request.session_id, request.message)
    
    # ❗ Usar StreamingResponse de FastAPI
    # El generador enviará los chunks y el Content-Type 'text/event-stream' 
    # es común para este tipo de comunicación (Server-Sent Events)
    return StreamingResponse(
        generator, 
        media_type="text/event-stream" 
    )