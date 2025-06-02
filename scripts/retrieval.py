import html
import json
import re
from typing import Generator, Union, List
from typing import Optional

import ftfy
import lancedb
import litellm
import numpy as np
from lancedb.embeddings import TextEmbeddingFunction
from lancedb.embeddings.registry import register, get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import CohereReranker
from lancedb.table import Table
from litellm import embedding as litellm_embedding
from litellm.caching.caching import Cache
from tqdm import tqdm

litellm.cache = Cache(disk_cache_dir="data/cache/litellm")


def get_data_model(embed_func) -> LanceModel:
    class Schema(LanceModel):
        content: str = embed_func.SourceField()
        vector: Vector(embed_func.ndims()) = embed_func.VectorField()
        url: str
        id: str
        doc_id: str
        source: str
        type: str

    return Schema


def get_batches(
    data: Generator,
    batch_size: int = 1000,
) -> Generator:
    """
    Splits the data into batches of size batch_size
    """
    batch = []
    for record in data:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


@register("litellm")
class LiteLLMEmbeddings(TextEmbeddingFunction):
    model: str
    dimensions: int
    caching: bool = True

    def generate_embeddings(
        self, texts: Union[List[str], np.ndarray], *args, **kwargs
    ) -> list[Union[np.array, None]]:
        embedding_output = litellm_embedding(
            self.model, input=list(texts), dimensions=self.ndims(), caching=self.caching
        )
        embeddings = embedding_output.get("data")
        return [np.array(embedding.get("embedding")) for embedding in embeddings]

    def ndims(self) -> int:
        return self.dimensions


class Retriever:
    def __init__(
        self,
        db_uri: str,
        table_name: str,
        embedding_provider: str = "litellm",
        embedding_model: str = "text-embedding-3-small",
        embed_dimensions: int = 768,
        rerank_model: str = "rerank-english-v3.0",
    ):

        self.db_uri = db_uri
        self.table_name = table_name
        self.table: Optional[Table] = None
        self.db = lancedb.connect(self.db_uri)
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embed_dimensions = embed_dimensions
        self.embed_func = (
            get_registry()
            .get(self.embedding_provider)
            .create(model=self.embedding_model, dimensions=self.embed_dimensions)
        )

        self.rerank_model = rerank_model
        self.reranker = CohereReranker(model_name=self.rerank_model, column="content")
        self._fts_index = None

    def create_index(self, data: Generator, batch_size: int = 100):
        schema = get_data_model(self.embed_func)
        self.table = self.db.create_table(
            self.table_name,
            schema=schema,
            data=get_batches(data, batch_size=batch_size),
            mode="overwrite",
        )
        self._fts_index = self.table.create_fts_index(
            "content", use_tantivy=True, replace=True
        )

    def add_data(self, data: Generator, batch_size: int = 100):
        if self.table is None:
            self.table = self.ensure_table()
        self.table.add(get_batches(data, batch_size=batch_size), mode="append")
        self.table.create_fts_index("content", use_tantivy=True, replace=True)

    def ensure_table(self):
        self.db = lancedb.connect(self.db_uri)
        self.table = self.db.open_table(self.table_name)
        return self.table

    def search(self, query: str, top_k: int = 5):
        self.table = self.ensure_table()
        results = (
            self.table.search(query, query_type="hybrid", fts_columns=["content"])
            .rerank(self.reranker)
            .limit(top_k)
            .to_pandas()
        )
        return results.to_dict(orient="records")


def format_doc(doc: dict) -> str:
    """
    Format a document dictionary into a string representation with an <a> tag for the source.
    """
    document_type = (
        f"<source><a href=\"{doc['url']}\">{doc['source']}: {doc['type']}</a></source>"
    )
    content = f"<excerpt>\n{html.escape(doc['content'])}\n</excerpt>"
    formatted_doc = document_type + content
    return formatted_doc


def format_retrieval(docs: List[dict]) -> str:
    """
    Format a list of document dictionaries into a string representation.
    """
    formatted_docs = ""
    for idx, doc in enumerate(docs):
        formatted_docs += f"<doc_{idx}>"
        formatted_docs += format_doc(doc)
        formatted_docs += f"</doc_{idx}>"

    xml_string = f"<retrieval>{formatted_docs}</retrieval>"

    xml_string = re.sub(r"\n{2,}", "\n", xml_string)
    return html.unescape(xml_string)


def replace_punctuation(text: str) -> str:
    """
    Replace punctuation in the text with a space.
    """
    # Define a regex pattern to match punctuation characters
    pattern = r"[^\w\s]"
    # Replace punctuation with a space
    return re.sub(pattern, " ", text)


def main():
    db_uri = "data/lancedb"
    table_name = "docstore"
    embedding_provider = "litellm"
    embedding_model = "text-embedding-3-small"
    embed_dimensions = 768

    retriever = Retriever(
        db_uri,
        table_name,
        embedding_provider,
        embedding_model,
        embed_dimensions,
    )

    def data_gen(f_name: str):
        required_fields = ["content", "url", "id", "doc_id", "source", "type"]
        with open(f_name) as f:
            for idx, line in enumerate(f):
                record = json.loads(line)
                record = {k: record[k] for k in required_fields}
                yield record

    # retriever.create_index(
    #     data_gen(f_name="data/chunked_documents.jsonl"), batch_size=256
    # )
    # print("Index created successfully.")

    def truncate_result(doc):
        required_cols = ["id", "doc_id", "content", "url", "source", "type"]
        return {k: doc[k] for k in required_cols}

    query_data = json.load(open("data/pipelines/outputs/pipeline.json"))
    with open("data/qna_with_retrieval.jsonl", "w") as f:
        for item in tqdm(query_data, total=len(query_data)):
            question = item["question"]
            answer = item["answer"]
            query = replace_punctuation(
                ftfy.fix_text((question + " " + answer).lower().replace("\n", " "))
            )
            results = retriever.search(f'"{query}"', top_k=10)
            item["retrieval"] = [truncate_result(res) for res in results]
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
