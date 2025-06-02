import asyncio
import json
from typing import Any, cast

import chromadb
import litellm
import numpy as np
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function
from litellm import embedding as litellm_embedding
from litellm.caching.caching import Cache
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from docsbot.retriever.reranker import Rerank, RerankSettings
from docsbot.utils import _get_id

litellm.cache = Cache(disk_cache_dir="data/cache/litellm")



@register_embedding_function
class LitellmEmbeddingFunction(EmbeddingFunction):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int = 768,
        caching: bool = True,
        batch_size: int = 100,
        api_key: str = None,
        **kwargs,
    ):
        self.model = model
        self.dimensions = dimensions
        self.caching = caching
        self.batch_size = batch_size
        self.api_key = api_key
        super().__init__(**kwargs)

    def _embed_batch(self, docs: list[str]):
        response = litellm_embedding(
            model=self.model,
            input=docs,
            dimensions=self.dimensions,
            caching=self.caching,
            api_key=self.api_key,
        )
        return [np.array(s["embedding"]) for s in response.data]

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for i in range(0, len(input), self.batch_size):
            batch = input[i : i + self.batch_size]
            embeddings.extend(self._embed_batch(batch))

        embeddings = cast(Embeddings, embeddings)
        return embeddings

    @staticmethod
    def name() -> str:
        return "litellm"

    @staticmethod
    def build_from_config(config: dict) -> "LitellmEmbeddingFunction":
        return LitellmEmbeddingFunction(
            model=config["model"],
            dimensions=config["dimensions"],
            caching=config["caching"],
            batch_size=config["batch_size"],
            api_key=config["api_key"],
        )

    def get_config(self) -> dict:
        return {
            "model": self.model,
            "dimensions": self.dimensions,
            "caching": self.caching,
            "batch_size": self.batch_size,
            "api_key": self.api_key,
        }


class ChromaRetrieverSettings(BaseSettings):
    db_uri: str
    collection_name: str
    embedding_model: str = "text-embedding-3-small"
    embedding_model_api_key: str = Field(
        ..., description="The API KEY for the embedding model", env="EMBEDDING_MODEL_API_KEY"
    )
    embedding_dimension: int = 768
    cache_embeddings: bool = True
    text_field: str = "text"
    id_field: str = "id"
    metadata_fields: list[str] = None
    batch_size: int = 2500
    default_top_k: int = 5
    model_config = SettingsConfigDict(
        arbitrary_types_allowed=True, env_prefix="", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


class ChromaRetriever(BaseModel):
    client: Any = None
    collection: Any = None
    embedding_function: LitellmEmbeddingFunction | None = None
    settings: ChromaRetrieverSettings = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, ctx):
        self.client = chromadb.PersistentClient(path=self.settings.db_uri)
        self.embedding_function = LitellmEmbeddingFunction(
            model=self.settings.embedding_model,
            dimensions=self.settings.embedding_dimension,
            caching=self.settings.cache_embeddings,
            api_key=self.settings.embedding_model_api_key,
        )

    def _ensure_collection(self):
        if self.collection is None:
            self.collection = self.client.get_or_create_collection(
                name=self.settings.collection_name, embedding_function=self.embedding_function
            )

    def index(self, documents: list[dict]):
        self._ensure_collection()
        texts = [d[self.settings.text_field] for d in documents]
        if self.settings.id_field in documents[0]:
            ids = [d[self.settings.id_field] for d in documents]
        else:
            ids = [_get_id(json.dumps(d)) for d in documents]
        if self.settings.metadata_fields:
            metadata = [{k: d[k] for k in self.settings.metadata_fields} for d in documents]
        else:
            metadata = None
        for i in range(0, len(texts), self.settings.batch_size):
            batch_texts = texts[i : i + self.settings.batch_size]
            batch_ids = ids[i : i + self.settings.batch_size]
            batch_metadata = metadata[i : i + self.settings.batch_size]
            self.collection.upsert(documents=batch_texts, ids=batch_ids, metadatas=batch_metadata)

    def _query(self, query: str | list[str], k: int = 5):
        self._ensure_collection()
        if isinstance(query, str):
            results = self.collection.query(query_texts=[query], n_results=k)
        else:
            results = self.collection.query(query_texts=query, n_results=k)
        texts = [item for sublist in results["documents"] for item in sublist]
        doc_ids = [item for sublist in results["ids"] for item in sublist]
        metadata = [item for sublist in results["metadatas"] for item in sublist]
        output = [
            {
                self.settings.text_field: texts[i],
                self.settings.id_field: doc_ids[i],
                "metadata": metadata[i],
            }
            for i in range(len(texts))
        ]
        return output

    async def query(self, query: str | list[str], k: int = 5):
        results = self._query(query, k)
        return results


class ChromaRetrieverWithReranker(ChromaRetriever):
    rerank: Rerank
    rerank_multiplier: float = 4.0

    async def query(self, query: str | list[str], k: int = 5):
        results = self._query(query, k=int(k * self.rerank_multiplier))
        output = await self.rerank(query, results, k=k)
        return output


async def main():
    qna_retriever = ChromaRetrieverWithReranker(
        settings=ChromaRetrieverSettings(
            db_uri="data/chromadb",
            collection_name="qna",
            embedding_model="text-embedding-3-small",
            embedding_dimension=768,
            cache_embeddings=True,
            text_field="question",
            id_field="id",
            metadata_fields=["answer", "context"],
        ),
        rerank=Rerank(
            settings=RerankSettings(
                id_field="id",
                text_field="question",
            )
        ),
    )

    wandb_retriever = ChromaRetrieverWithReranker(
        settings=ChromaRetrieverSettings(
            db_uri="data/chromadb",
            collection_name="wandb",
            embedding_model="text-embedding-3-small",
            embedding_dimension=768,
            cache_embeddings=True,
            text_field="content",
            id_field="id",
            metadata_fields=["source", "type", "url"],
        ),
        rerank=Rerank(
            settings=RerankSettings(
                id_field="id",
                text_field="content",
            )
        ),
    )

    blogs_and_guides_retriever = ChromaRetrieverWithReranker(
        settings=ChromaRetrieverSettings(
            db_uri="data/chromadb",
            collection_name="blogs_and_guides",
            embedding_model="text-embedding-3-small",
            embedding_dimension=768,
            cache_embeddings=True,
            text_field="content",
            id_field="id",
            metadata_fields=["source", "type", "url"],
        ),
        rerank=Rerank(
            settings=RerankSettings(
                id_field="id",
                text_field="content",
            )
        ),
    )

    query = "can I embed an iframe in a report?"
    qna_results = await qna_retriever.query(query, 10)
    queries = (
        [query]
        + [result[qna_retriever.settings.text_field] for result in qna_results]
        + [result["metadata"]["answer"] for result in qna_results]
    )
    wandb_results, bng_results = await asyncio.gather(
        wandb_retriever.query(queries, 10),
        blogs_and_guides_retriever.query(queries, 10),
    )

    final_results = await wandb_retriever.rerank(queries, wandb_results + bng_results, k=10)
    for result in final_results:
        print(result["id"])
        print(result["metadata"]["url"])
        print("---")


if __name__ == "__main__":
    asyncio.run(main())
