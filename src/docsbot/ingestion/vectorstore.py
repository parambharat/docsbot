import json

import pandas as pd

from docsbot.retriever import ChromaRetrieverWithReranker
from docsbot.retriever.reranker import Rerank, RerankSettings
from docsbot.retriever.retriever import ChromaRetrieverSettings


def load_docs(fname):
    df = pd.read_json(fname, lines=True, orient="records")
    required_cols = ["id", "url", "content", "source", "type"]
    df = df[required_cols]
    df = df.drop_duplicates("id").reset_index(drop=True)
    wandb_df = df[df["source"] == "wandb"]
    weave_df = df[df["source"] == "weave"]
    coreweave_df = df[df["source"] == "coreweave"]
    blogs_and_guides_df = df.loc[
        df.index.difference(wandb_df.index.tolist() + weave_df.index.tolist() + coreweave_df.index.tolist())
    ]
    return {
        "wandb": wandb_df.to_dict(orient="records"),
        "weave": weave_df.to_dict(orient="records"),
        "coreweave": coreweave_df.to_dict(orient="records"),
        "blogs_and_guides": blogs_and_guides_df.to_dict(orient="records"),
    }


def read_qna(fname):
    df = pd.read_json(fname, lines=True, orient="records")
    df = df[df["annotated_answer"].map(lambda x: x["factually_consistent"])]
    df["answer"] = df["annotated_answer"].map(lambda x: x["annotated_answer"])
    df["context"] = df["context"].map(json.dumps)
    df = df.drop_duplicates("idx").reset_index(drop=True)
    df = df.rename(columns={"idx": "id"})
    required_cols = ["id", "question", "answer", "context"]
    df = df[required_cols].reset_index(drop=True)
    return df.to_dict(orient="records")


def main():
    qna_data = read_qna("data/qna_judgements_with_verification.jsonl")
    docs_data = load_docs("data/chunked_documents.jsonl")

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
        rerank=Rerank(settings=RerankSettings()),
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
        rerank=Rerank(settings=RerankSettings()),
    )

    weave_retriever = ChromaRetrieverWithReranker(
        settings=ChromaRetrieverSettings(
            db_uri="data/chromadb",
            collection_name="weave",
            embedding_model="text-embedding-3-small",
            embedding_dimension=768,
            cache_embeddings=True,
            text_field="content",
            id_field="id",
            metadata_fields=["source", "type", "url"],
        ),
        rerank=Rerank(settings=RerankSettings()),
    )

    coreweave_retriever = ChromaRetrieverWithReranker(
        settings=ChromaRetrieverSettings(
            db_uri="data/chromadb",
            collection_name="coreweave",
            embedding_model="text-embedding-3-small",
            embedding_dimension=768,
            cache_embeddings=True,
            text_field="content",
            id_field="id",
            metadata_fields=["source", "type", "url"],
        ),
        rerank=Rerank(settings=RerankSettings()),
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
        rerank=Rerank(settings=RerankSettings()),
    )

    # Add the data to the retrievers
    qna_retriever.index(qna_data)

    wandb_retriever.index(docs_data["wandb"])
    weave_retriever.index(docs_data["weave"])
    coreweave_retriever.index(docs_data["coreweave"])
    blogs_and_guides_retriever.index(docs_data["blogs_and_guides"])


if __name__ == "__main__":
    main()
