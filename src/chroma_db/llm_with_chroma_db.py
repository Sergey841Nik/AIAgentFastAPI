from typing import Any

from langchain_gigachat import GigaChat
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.config import setup_logger, settings
from src.chroma_db.functions import save_file


logger = setup_logger(__name__)

class ChatWithLLM:
    def __init__(self) -> None:
        self.llm = GigaChat(
            credentials=settings.GIGACHAT_API_KEY,
            temperature=settings.TEMPERATURE_LLM,
            model=settings.MODEL_LLM_NAME,
            scope=settings.SCOPE_LLM,
            verify_ssl_certs=False,
            )
        logger.info("✅ Модель GigaChat успешно инициализирована")
        
    async def response(self, query: str, formatted_context: str, tools: list):
        """Генерация ответа на основе запроса и контекста."""
        messages = [
            SystemMessage(content="""
                Ты AI-помощник, работающий с контекстом информации. Ты умеешь создавать файлы по запросу. 
                Ты умеешь отвечать на вопросы, связанные с контекстом информации.
                
                Правила:
                1. Сразу переходи к сути, без фраз типа "На основе контекста"
                2. После запроса пользователя о создании файла или о сохранении информации вызывай функцию save_file
                """,
            ),
            HumanMessage(
                content = f"Вопрос: {query}\nКонтекст: {formatted_context}"
            ),
        ]
        logger.info("🔄 Готовим ответа для запроса: «%s»", query)
        try:
            llm_with_tools = self.llm.bind_tools(tools)

            ai_msg = await llm_with_tools.ainvoke(messages)
            
            messages.append(ai_msg)

            if ai_msg.tool_calls:
                # Модель вызвала функцию
                for tool_call in ai_msg.tool_calls:
                    selected_tool = {"save_file": save_file}[tool_call["name"].lower()]
                    tool_output = await selected_tool.ainvoke(tool_call["args"])
                    messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
                    logger.info("✅ Модель вызвала функцию %s и вернула результат: %s", tool_call["name"], tool_output)
                response = await llm_with_tools.ainvoke(messages)

                return response.content
            
            else:
                # Простое получение результата без вызова функции
                response = await llm_with_tools.ainvoke(messages)
                return response.content
            
        except Exception as e:
            logger.error("Ошибка при генерации ответа: %s", e)
            return "Произошла ошибка при генерации ответа."