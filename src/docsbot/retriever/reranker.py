import asyncio

from litellm.rerank_api.main import arerank
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RerankSettings(BaseSettings):
    model_config = SettingsConfigDict(
        arbitrary_types_allowed=True, env_prefix="", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
    reranker_model: str = "rerank-v3.5"
    rerank_model_api_key: str = Field(
        ..., description="The API KEY to use for the reranker", env="RERANK_MODEL_API_KEY"
    )
    id_field: str
    text_field: str


class Rerank(BaseModel):
    settings: RerankSettings
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def _rerank(self, query: str, results: list[dict], k: int = 5):
        if len(results) == 0:
            return []

        texts = [result[self.settings.text_field] for result in results]

        reranked_results = await arerank(
            model=self.settings.reranker_model,
            query=query,
            documents=texts,
            top_n=k,
            caching=True,
            api_key=self.settings.rerank_model_api_key,
        )
        final_results = []
        for result in reranked_results.results:
            res = results[result["index"]]
            res["relevance_score"] = result["relevance_score"]
            final_results.append(res)
        return final_results

    async def __call__(self, query: str | list[str], results: list[dict], k: int = 5):
        if isinstance(query, list):
            output = await asyncio.gather(*[self._rerank(q, results, k=k) for q in query])
            output = [item for sublist in output for item in sublist]
            deduped = {item[self.settings.id_field]: item for item in output}
            output = list(deduped.values())
            output = sorted(output, key=lambda x: x["relevance_score"], reverse=True)
        else:
            output = await self._rerank(query, results, k=k)
        return output
