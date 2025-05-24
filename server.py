from datetime import datetime
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Загружаем переменные окружения
load_dotenv()

# Функция получения времени
def get_current_time():
    """Возвращает текущее UTC время в ISO-8601 формате."""
    return {"utc": datetime.utcnow().isoformat() + "Z"}

# Получение ключа для Qwen API
qwen_api_key = os.getenv("QWEN_API_KEY")

# Настройка модели
model = ChatOpenAI(model="qwen/qwen3-235b-a22b", api_key=qwen_api_key, openai_api_base="https://openrouter.ai/api/v1").bind_tools([get_current_time])

# Логика агента

# Упрощенная логика - нейросеть отвечает на каждый запрос, просто ловя вопрос о времени - срабатывает tool
# def agent(state: MessagesState):
#     response = model.invoke(state['messages'])
#     return {"messages": [response]}

# Усложненная версия - нейросеть реагирует только на вопрос о времени, на все остальное не отвечает
def agent(state: MessagesState):
    messages = state["messages"]

    last_message = messages[-1] if messages else None

    if isinstance(last_message, HumanMessage):
        user_input = last_message.content.lower()

        if "time" in user_input or "час" in user_input:
            # Вызываем модель, которая сама решит, использовать ли инструмент
            response = model.invoke(messages)
            return {"messages": [response]}
        else:
            return {"messages": [AIMessage(content="Я простой бот. Спросите 'What time is it?'")]}

    return {"messages": [AIMessage(content="Не понимаю запроса.")]}

# Граф
workflow = StateGraph(MessagesState)

workflow.add_node("agent", agent)
workflow.add_node("tools", ToolNode([get_current_time]))

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

# Компилируем граф
app = workflow.compile()

if __name__ == "__main__":
    from langgraph.cli import run
    run(app)