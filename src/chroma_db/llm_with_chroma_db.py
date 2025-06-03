import uuid
from typing import Sequence

from langchain_gigachat.chat_models import GigaChat
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import setup_logger, settings
from src.chroma_db.functions import save_file


logger = setup_logger(__name__)


class ChatWithLLM:
    def __init__(self, model: str = "gigachat", tools: Sequence[BaseTool] = [save_file]) -> None:
        if model == "gigachat":
            llm = GigaChat(
                credentials=settings.GIGACHAT_API_KEY,
                temperature=settings.TEMPERATURE_LLM,
                model=settings.MODEL_LLM_NAME_GIGA,
                scope=settings.SCOPE_LLM,
                verify_ssl_certs=False,
            )

        if model == "deepseek":
            llm = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                model=settings.MODEL_LLM_NAME_DEEP,
                api_key=settings.DEEPSEEC_API_KEY,
                temperature=settings.TEMPERATURE_LLM,
            )
            
        self._config: RunnableConfig = {"configurable": {"thread_id": uuid.uuid4().hex}}

        self.agent = create_react_agent(
            llm,
            tools=tools,
            checkpointer=InMemorySaver(),
        )
        logger.info("✅ Модель %s успешно инициализирована", model)

    def response(
        self,
        query: str,
        attachments: str | None = None,
    ):
        """Генерация ответа на основе запроса и контекста."""

        messages = [
            SystemMessage(
                content="""
                Ты AI-помощник, работающий с контекстом информации. Ты умеешь создавать файлы по запросу. 
                Ты умеешь отвечать на вопросы.
                Если вопрос связан с контекстом в первую очередь бери информацию из контекста, если нет, то используй свои знания.
                """
            ),
            HumanMessage(content=f"Вопрос: {query}\nКонтекст: {attachments}"),
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
