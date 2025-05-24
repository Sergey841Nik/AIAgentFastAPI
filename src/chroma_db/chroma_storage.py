from logging import Logger
import time
import shutil
from typing import Literal
from pathlib import Path

from langchain_core.documents import Document
import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import BASE_DIR, settings, setup_logger
from src.chroma_db.utils import document_generator

logger: Logger = setup_logger(__name__) 

class ChromaVectorStorage:
   """
   Класс для работы с хранилищем векторов Chroma.

   Args:
       name_vector_storage (str): Имя хранилища векторов.

   Attributes:
       name_vector_storage (str): Имя хранилища векторов.
       _store (Chroma | None): Объект Chroma для работы с базой данных.
   """

   def __init__(self, name_vector_storage) -> None:
       """
       Инициализирует пустой экземпляр хранилища векторов c именем name_vector_storage.
       Соединение с базой данных будет установлено позже с помощью метода init().
       """
       self.name_vector_storage = name_vector_storage
       self._store: Chroma | None = None

   def create(
           self,
           folder_path: Path,
           persist_dir: str,
           embedding_model: str = settings.LM_MODEL_NAME
       ) -> Chroma:
       """
        Создает ChromaDB из всех файлов в папке
        Args:
            folder_path: Путь к папке с файлами
            persist_dir: Директория для сохранения БД
            embedding_model: Модель для эмбеддингов
        Return: Объект ChromaDB
        Raises:
           Exception: Ошибка при создании базы данных Chroma
       """
       try:
           start_time: float = time.time()
           logger.info("Загрузка модели эмбеддингов...")
           # Определяем устройство для вычислений: GPU если доступен, иначе CPU
           device: Literal['cuda'] | Literal['cpu'] = "cuda" if torch.cuda.is_available() else "cpu"
           embeddings = HuggingFaceEmbeddings(
                   model_name=embedding_model,
                   model_kwargs={"device": device},
                   encode_kwargs={"normalize_embeddings": True},
               )
           logger.info("Модель загружена за %.2f сек", time.time() - start_time)

           documents = list(document_generator(folder_path))

           logger.info("Количество документов: %d", len(documents))
           ids: list[str] = [f"doc_{i}" for i in range(len(documents))]

           logger.info("Создание базы данных Chroma...")
           db = Chroma.from_documents(
               documents=documents,
               embedding=embeddings,
               persist_directory=persist_dir,
               collection_name=self.name_vector_storage,
               ids=ids
           )
           logger.info("База данных создана за %.2f сек", time.time() - start_time)
           shutil.rmtree(folder_path)
           logger.info("Папка удалена")
           
           return db
       except Exception as e:
           logger.error("Ошибка при создании базы данных Chroma: %s", str(e))
           raise
                   

   async def init(self):
       """
       Асинхронный метод для инициализации соединения с базой данных Chroma.
       Создает embeddings на основе модели из настроек, используя CUDA если доступно.
       """
       logger.info("🤔 Инициализация ChromaVectorStore...")
       try:
           # Определяем устройство для вычислений: GPU если доступен, иначе CPU
           device: Literal['cuda'] | Literal['cpu'] = "cuda" if torch.cuda.is_available() else "cpu"
           logger.info("🚀 Используем устройство для эмбеддингов: %s", device)

           # Создаем модель эмбеддингов с указанными параметрами
           embeddings = HuggingFaceEmbeddings(
               model_name=settings.LM_MODEL_NAME,
               model_kwargs={"device": device},
               encode_kwargs={"normalize_embeddings": True},
           )

           # Инициализируем соединение с базой данных Chroma
           self._store = Chroma(
               persist_directory=BASE_DIR / "src" / "chroma_db" / "vector_db",
               embedding_function=embeddings,
               collection_name=self.name_vector_storage,
           )

           logger.info(
               "✅ ChromaVectorStorage успешно подключен к коллекции  \
               '%s' в '%s/src/chroma_db/vector_db'",
               self.name_vector_storage, BASE_DIR
           )
       except Exception as e:
           logger.error("❌ Ошибка при инициализации ChromaVectorStorage: %s", e)
           raise

   async def asimilarity_search(self, query: str, with_score: bool, k: int = 5) -> list:
       """
       Асинхронный метод для поиска похожих документов в базе данных Chroma.

       Args:
           query (str): Текстовый запрос для поиска
           with_score (bool): Включать ли оценку релевантности в результаты
           k (int): Количество возвращаемых результатов

       Returns:
           list: Список найденных документов, возможно с оценками если with_score=True

       Raises:
           RuntimeError: Если хранилище не инициализировано
       """
       if not self._store:
           raise RuntimeError("ChromaVectorStore is not initialized.")

       logger.info("🔍 Поиск похожих документов по запросу: «%s», top_k=%s", query, k)
       try:
           if with_score:
               results: list[tuple[Document, float]] = await self._store.asimilarity_search_with_score(
                   query=query, k=k
               )
           else:
               results: list[Document] = await self._store.asimilarity_search(query=query, k=k)

           logger.info("📄 Найдено %s результатов.", len(results))
           return results
       except Exception as e:
           logger.error("❌ Ошибка при поиске: %s", e)
           raise

if __name__ == "__main__":
    vector_storage = ChromaVectorStorage(name_vector_storage="my_collection_test")
    vector_storage.create(folder_path=Path(BASE_DIR / "upload_files" / "test"), persist_dir="src/chroma_db/vector_db")
   