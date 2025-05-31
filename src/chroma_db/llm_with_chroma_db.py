import uuid
from typing import Sequence

from langchain_gigachat.chat_models import GigaChat
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import setup_logger, settings
from src.chroma_db.functions import save_file


logger = setup_logger(__name__)

class ChatWithLLM:
    def __init__(self, tools: Sequence[BaseTool] = [save_file]) -> None:
        llm = GigaChat(
            credentials=settings.GIGACHAT_API_KEY,
            temperature=settings.TEMPERATURE_LLM,
            model=settings.MODEL_LLM_NAME,
            scope=settings.SCOPE_LLM,
            verify_ssl_certs=False,
            )
        self._config: RunnableConfig = {
                "configurable": {"thread_id": uuid.uuid4().hex}}
        
        self.agent = create_react_agent(
            llm,
            tools=tools,
            checkpointer=InMemorySaver(),
            )
        
        logger.info("✅ Модель GigaChat успешно инициализирована")
        
    def response(
            self, 
            query: str, 
            attachments: str | None=None,
            ):
        """Генерация ответа на основе запроса и контекста."""
    
        messages = [
            SystemMessage(content="""
                Ты AI-помощник, работающий с контекстом информации. Ты умеешь создавать файлы по запросу. 
                Ты умеешь отвечать на вопросы.
                Если вопрос связан с контекстом в первую очередь бери информацию оттуда.
                
                Правила:
                1. Сразу переходи к сути, без фраз типа "На основе контекста"
                2. После запроса пользователя о создании файла или о сохранении информации вызывай функцию save_file
                """ 
            ),
            HumanMessage(
                content = f"Вопрос: {query}\nКонтекст: {attachments}"
                
            ),
        ]
        logger.info("🔄 Готовим ответа для запроса: «%s»", query)
        try:
            result = self.agent.invoke(
                {"messages": messages},
                config=self._config,
            )
            return result["messages"][-1].content
            
        except Exception as e:
            logger.error("Ошибка при генерации ответа: %s", e)
            return "Произошла ошибка при генерации ответа."
        
    