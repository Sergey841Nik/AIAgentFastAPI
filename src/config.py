from logging import (
    Formatter,
    Logger,
    handlers,
    StreamHandler,
    getLogger,
    INFO,
    ERROR,
)
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR: Path = Path(__file__).parent.parent


class Settings(BaseSettings):
    # chroma vector db and LLM model
    GIGACHAT_API_KEY: str
    MODEL_LLM_NAME_GIGA: str = "GigaChat-Pro"
    TEMPERATURE_LLM: float = 0.7
    SCOPE_LLM: str = "GIGACHAT_API_PERS"
    MODEL_LLM_NAME_DEEP: str = "deepseek/deepseek-chat-v3-0324:free"
    DEEPSEEC_API_KEY: SecretStr
    LM_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # db config
    DB_HOST: str
    DB_PORT: int
    DB_USER: str
    DB_PASS: str
    DB_NAME: str

    @property
    def url(self):
        return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    echo: bool = False
    model_config = SettingsConfigDict(env_file=f"{BASE_DIR}/.env")

    # auth
    ALGORITHM: str
    private_key_path: Path = BASE_DIR / "cert" / "private.pem"
    public_key_path: Path = BASE_DIR / "cert" / "public.pem"
    access_token_expire: int = 15


settings = Settings()


def setup_logger(name: str = "my_project") -> Logger:
    """Настройка логгера с конфигурацией для вывода в файл и консоль."""

    # Создаем логгер
    logger: Logger = getLogger(name)
    logger.setLevel(INFO)

    # Создаем директорию для логов, если ее нет
    logs_dir = BASE_DIR / Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Форматтер для логов
    formatter = Formatter(
        fmt="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Обработчик для записи в файл (ротация по дням)
    file_handler = handlers.TimedRotatingFileHandler(
        filename=logs_dir / "aiagent.log",
        when="midnight",
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(ERROR)  # В файл пишем INFO и выше

    # Обработчик для вывода в консоль
    console_handler = StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(INFO)

    # Добавляем обработчики к логгеру
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
