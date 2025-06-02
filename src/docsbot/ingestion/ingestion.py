import asyncio
import json
import pathlib
import re
from urllib.parse import urljoin, urlparse, urlunparse

import aiofiles
import yaml

from docsbot.ingestion.chunking import LengthFunction, MarkdownChunker, PythonChunker
from docsbot.ingestion.crawlers import (
    CodeCrawler,
    CoreweaveCrawler,
    ReportsCrawler,
    WeaveServiceApiDocsCrawler,
)
from docsbot.ingestion.preprocessors import (
    BaseMarkdownConverter,
    NotebookConverter,
    WandbMarkdownConverter,
    WeaveMarkdownConverter,
)
from docsbot.utils import _get_id

CHUNK_SIZE = 768
ENCODING_MODEL = "text-embedding-3-small"


def _fix_url(url_string: str) -> str:
    """Fixes a URL by normalizing its path component."""
    parsed_url = urlparse(url_string)
    path_segments = parsed_url.path.split("/")
    normalized_path = "/".join(segment for segment in path_segments if segment)
    fixed_url = urlunparse(
        (
            parsed_url.scheme,
            parsed_url.netloc,
            normalized_path,
            parsed_url.params,
            parsed_url.query,
            parsed_url.fragment,
        )
    )
    return fixed_url


def read_repo_documents(
    files: list[pathlib.Path],
    docs_dir: pathlib.Path,
    base_url: str,
    source: str,
    document_type: str,
):
    files = [f for f in files if ".github" not in f.parts]

    def _get_url(file_path):
        file_path = pathlib.Path(file_path).resolve()
        rel_path = file_path.relative_to(docs_dir)
        parts = list(rel_path.parts)
        url_path = "/".join(parts)
        return f"{base_url}/{url_path}".rstrip("/")

    for file in files:
        content = file.read_text()
        yield {
            "id": _get_id(content),
            "url": _fix_url(_get_url(file)),
            "uri": file.as_uri(),
            "directory": docs_dir.as_uri(),
            "content": content,
            "source": source,
            "type": document_type,
        }


def read_wandb_code(docs_dir: pathlib.Path, base_url: str):
    files = list(pathlib.Path(docs_dir / "wandb").rglob("*.py"))
    files.extend(list(pathlib.Path(docs_dir / "tests").rglob("*.py")))
    return read_repo_documents(files, docs_dir, base_url, "wandb", "code")


def read_weave_code(docs_dir: pathlib.Path, base_url: str):
    files = list(pathlib.Path(docs_dir / "weave").rglob("*.py"))
    files.extend(list(pathlib.Path(docs_dir / "tests").rglob("*.py")))
    return read_repo_documents(files, docs_dir, base_url, "weave", "code")


def read_weave_notebooks(docs_dir: pathlib.Path, base_url: str):
    files = list(pathlib.Path(docs_dir / "docs/notebooks").rglob("*.ipynb"))
    return read_repo_documents(files, docs_dir, base_url, "weave", "notebook")


def read_weave_docs(docs_dir: pathlib.Path, base_url: str):
    files = list((docs_dir / "docs").rglob("*.md"))

    def _get_url(file_path: pathlib.Path):
        file_path = pathlib.Path(file_path).resolve()
        rel_path = file_path.relative_to(docs_dir)
        parts = list(rel_path.parts)

        if file_path.name.lower() in ("index.md", "index.mdx"):
            sanitized_parts = [re.sub(r"^[\d\-_]+", "", p) for p in parts[:-1]]
            url_path = "/".join(sanitized_parts)
        else:
            sanitized_parts = [re.sub(r"^[\d\-_]+", "", p) for p in parts]
            sanitized_parts[-1] = pathlib.Path(sanitized_parts[-1]).with_suffix("").as_posix()
            url_path = "/".join(sanitized_parts)

        return f"{base_url}/{url_path}".rstrip("/")

    for file in files:
        content = file.read_text()
        yield {
            "id": _get_id(content),
            "url": _fix_url(_get_url(file)),
            "uri": file.as_uri(),
            "directory": docs_dir.as_uri(),
            "content": content,
            "source": "weave",
            "type": "documentation",
        }


async def crawl_weave_service_api_docs(docs_dir: pathlib.Path):
    docs_dir = (docs_dir / "docs/docs/reference/service-api").resolve()
    service_api_crawler = WeaveServiceApiDocsCrawler()
    converter = BaseMarkdownConverter()
    chunker = MarkdownChunker(
        max_chunk_size=512,
        length_fn=LengthFunction(encoding_model="gpt-4o-mini"),
    )
    async for record in service_api_crawler.crawl():
        record = {
            "id": _get_id(record["url"] + record["content"]),
            "url": _fix_url(record["url"]),
            "uri": record["url"],
            "directory": docs_dir.as_uri(),
            "content": record["content"],
            "source": "weave",
            "type": "documentation",
        }
        record = converter(record)
        chunks = chunker(record)
        for chunk in chunks:
            yield chunk


def read_example_docs(docs_dir: pathlib.Path, base_url: str):
    files = [pathlib.Path(docs_dir / "README.md")]
    files.extend(pathlib.Path(docs_dir / "colabs").rglob("*.md"))
    files.extend(pathlib.Path(docs_dir / "examples").rglob("*.md"))
    files.extend(pathlib.Path(docs_dir / "weave").rglob("*.md"))

    return read_repo_documents(files, docs_dir, base_url, "examples", "documentation")


def read_examples_notebooks(docs_dir: pathlib.Path, base_url: str):
    files = list(pathlib.Path(docs_dir / "colabs").rglob("*.ipynb"))
    files.extend(list(pathlib.Path(docs_dir / "weave").rglob("*.ipynb")))
    return read_repo_documents(files, docs_dir, base_url, "examples", "notebook")


def read_examples_code(docs_dir: pathlib.Path, base_url: str):
    files = list(pathlib.Path(docs_dir / "examples").rglob("*.py"))
    return read_repo_documents(files, docs_dir, base_url, "examples", "code")


def read_edu_docs(docs_dir: pathlib.Path, base_url: str):
    files = list(pathlib.Path(docs_dir).rglob("*.md"))
    return read_repo_documents(files, docs_dir, base_url, "edu", "documentation")


def read_edu_notebooks(docs_dir: pathlib.Path, base_url: str):
    files = list(pathlib.Path(docs_dir).rglob("*.ipynb"))
    return read_repo_documents(files, docs_dir, base_url, "edu", "notebook")


def read_edu_code(docs_dir: pathlib.Path, base_url: str):
    files = list(pathlib.Path(docs_dir).rglob("*.py"))
    return read_repo_documents(files, docs_dir, base_url, "edu", "code")


def read_wandb_docs(docs_dir: pathlib.Path, base_url: str):
    files: list[pathlib.Path] = []
    docs_dir = docs_dir / "content"
    files.extend(pathlib.Path(docs_dir / "guides").rglob("*.md"))
    files.extend(pathlib.Path(docs_dir / "launch").rglob("*.md"))
    files.extend(pathlib.Path(docs_dir / "ref").rglob("*.md"))
    files.extend(pathlib.Path(docs_dir / "support").rglob("*.md"))
    files.extend(pathlib.Path(docs_dir / "tutorials").rglob("*.md"))

    frontmatter_re = re.compile(r"^---(.*?)---\n+(.*)", flags=re.DOTALL)

    def _parse_frontmatter(content):
        match = frontmatter_re.match(content)
        if match:
            frontmatter = yaml.safe_load(match.group(1))
            body = match.group(2)
            return frontmatter, body
        return {}, content

    def _get_url(file_path, frontmatter):
        file_path = pathlib.Path(file_path)
        if "url" in frontmatter and frontmatter["url"]:
            url = frontmatter["url"]
            return f"{base_url}/{url.strip('/')}"
        dir_path = file_path.parent
        if file_path.name == "_index.md":
            if dir_path == docs_dir:
                return base_url
            return f"{base_url}/{dir_path.name.strip('/')}"
        index_path = dir_path / "_index.md"
        if index_path.exists():
            index_content = index_path.read_text()
            index_frontmatter, _ = _parse_frontmatter(index_content)
            if "url" in index_frontmatter and index_frontmatter["url"]:
                index_base_url = urljoin(base_url + "/", index_frontmatter["url"].lstrip("/"))
                if "cascade" in index_frontmatter:
                    cascade = index_frontmatter["cascade"]
                    if isinstance(cascade, list):
                        for c in cascade:
                            if isinstance(c, dict) and "url" in c and ":filename" in c["url"]:
                                filename = file_path.stem
                                return f"{index_base_url}/{c['url'].replace(':filename', filename).strip('/')}"
                    elif isinstance(cascade, dict) and "url" in cascade and ":filename" in cascade["url"]:
                        filename = file_path.stem
                        return f"{index_base_url}/{cascade['url'].replace(':filename', filename).strip('/')}"
                filename = file_path.stem
                return f"{index_base_url}/{filename}"
        try:
            rel_path = file_path.relative_to(docs_dir).with_suffix("").as_posix()
        except ValueError:
            rel_path = file_path.with_suffix("").as_posix()
        return f"{base_url}/{rel_path}"

    for file in files:
        content = file.read_text()
        frontmatter, _ = _parse_frontmatter(content)
        url = _fix_url(_get_url(file, frontmatter))
        yield {
            "id": _get_id(url + "\n" + content),
            "url": url,
            "uri": file.as_uri(),
            "directory": file.as_uri(),
            "content": content,
            "source": "wandb",
            "type": "documentation",
        }


async def crawl_wandb_docs(
    repo_owner: str = "wandb",
    repo_name: str = "docodile",
    output_dir: str = "../data/code",
    chunk_size: int = 512,
    encoding_model: str = "gpt-4o-mini",
):
    wandb_docs_crawler = CodeCrawler(
        repo_owner=repo_owner,
        repo_name=repo_name,
        output_dir=output_dir,
        branch_name="main",
    )
    wandb_docs_crawler = await wandb_docs_crawler.crawl()
    docs_dir = pathlib.Path(output_dir) / repo_name
    converter = WandbMarkdownConverter()
    chunker = MarkdownChunker(
        max_chunk_size=chunk_size,
        length_fn=LengthFunction(encoding_model=encoding_model),
    )
    for item in read_wandb_docs(
        docs_dir=pathlib.Path(docs_dir).resolve(),
        base_url="https://docs.wandb.ai/",
    ):
        record = converter(item)
        chunks = chunker(record)
        for chunk in chunks:
            yield chunk


async def crawl_wandb_code(
    repo_owner: str = "wandb",
    repo_name: str = "wandb",
    output_dir: str = "../data/code",
    chunk_size: int = 512,
    encoding_model: str = "gpt-4o-mini",
):
    wandb_code_crawler = CodeCrawler(
        repo_owner=repo_owner,
        repo_name=repo_name,
        output_dir=output_dir,
    )
    wandb_code_crawler = await wandb_code_crawler.crawl()
    docs_dir = pathlib.Path(output_dir) / repo_name
    final_branch_name = wandb_code_crawler.final_branch_name
    code_chunker = PythonChunker(
        max_chunk_size=chunk_size,
        length_fn=LengthFunction(encoding_model=encoding_model),
    )
    for item in read_wandb_code(
        docs_dir=docs_dir.resolve(),
        base_url=f"https://github.com/{repo_owner}/{repo_name}/blob/{final_branch_name}/",
    ):
        chunks = code_chunker(item)
        for chunk in chunks:
            yield chunk


async def crawl_weave_code(
    repo_owner: str = "wandb",
    repo_name: str = "weave",
    output_dir: str = "../data/code",
    chunk_size: int = 512,
    encoding_model: str = "gpt-4o-mini",
):
    weave_code_crawler = CodeCrawler(
        repo_owner=repo_owner,
        repo_name=repo_name,
        output_dir=output_dir,
    )
    weave_code_crawler = await weave_code_crawler.crawl()
    docs_dir = pathlib.Path(output_dir) / repo_name
    final_branch_name = weave_code_crawler.final_branch_name
    notebook_converter = NotebookConverter()
    markdown_converter = WeaveMarkdownConverter()
    base_converter = BaseMarkdownConverter()
    markdown_chunker = MarkdownChunker(
        max_chunk_size=chunk_size,
        length_fn=LengthFunction(encoding_model=encoding_model),
    )
    code_chunker = PythonChunker(
        max_chunk_size=chunk_size,
        length_fn=LengthFunction(encoding_model=encoding_model),
    )
    for item in read_weave_notebooks(
        docs_dir=pathlib.Path(docs_dir).resolve(),
        base_url=f"https://github.com/{repo_owner}/{repo_name}/blob/{final_branch_name}/",
    ):
        record = notebook_converter(item)
        chunks = markdown_chunker(record)
        for chunk in chunks:
            yield chunk
    for item in read_weave_docs(
        docs_dir=pathlib.Path(docs_dir).resolve(),
        base_url="https://weave-docs.wandb.ai/",
    ):
        record = markdown_converter(item)
        chunks = markdown_chunker(record)
        for chunk in chunks:
            yield chunk

    async for item in crawl_weave_service_api_docs(docs_dir=docs_dir):
        record = base_converter(item)
        chunks = markdown_chunker(record)
        for chunk in chunks:
            yield chunk

    for item in read_weave_code(
        docs_dir=pathlib.Path(docs_dir).resolve(),
        base_url=f"https://github.com/{repo_owner}/{repo_name}/blob/{final_branch_name}/",
    ):
        chunks = code_chunker(item)
        for chunk in chunks:
            yield chunk


async def crawl_examples(
    repo_owner: str = "wandb",
    repo_name: str = "examples",
    output_dir: str = "../data/code",
    chunk_size: int = 512,
    encoding_model: str = "gpt-4o-mini",
):
    examples_code_crawler = CodeCrawler(
        repo_owner=repo_owner,
        repo_name=repo_name,
        output_dir=output_dir,
        branch_name="master",
    )
    examples_code_crawler = await examples_code_crawler.crawl()
    docs_dir = pathlib.Path(output_dir) / repo_name
    final_branch_name = examples_code_crawler.final_branch_name
    markdown_converter = BaseMarkdownConverter()
    notebook_converter = NotebookConverter()
    markdown_chunker = MarkdownChunker(
        max_chunk_size=chunk_size,
        length_fn=LengthFunction(encoding_model=encoding_model),
    )
    code_chunker = PythonChunker(
        max_chunk_size=chunk_size,
        length_fn=LengthFunction(encoding_model=encoding_model),
    )

    for item in read_example_docs(
        docs_dir=pathlib.Path(docs_dir).resolve(),
        base_url=f"https://github.com/{repo_owner}/{repo_name}/blob/{final_branch_name}/",
    ):
        record = markdown_converter(item)
        chunks = markdown_chunker(record)
        for chunk in chunks:
            yield chunk
    for item in read_examples_notebooks(
        docs_dir=pathlib.Path(docs_dir).resolve(),
        base_url=f"https://github.com/{repo_owner}/{repo_name}/blob/{final_branch_name}/",
    ):
        record = notebook_converter(item)
        chunks = markdown_chunker(record)
        for chunk in chunks:
            yield chunk
    for item in read_examples_code(
        docs_dir=pathlib.Path(docs_dir).resolve(),
        base_url=f"https://github.com/{repo_owner}/{repo_name}/blob/{final_branch_name}/",
    ):
        chunks = code_chunker(item)
        for chunk in chunks:
            yield chunk


async def crawl_edu(
    repo_owner: str = "wandb",
    repo_name: str = "edu",
    output_dir: str = "../data/code",
    chunk_size: int = 512,
    encoding_model: str = "gpt-4o-mini",
):
    edu_code_crawler = CodeCrawler(
        repo_owner=repo_owner,
        repo_name=repo_name,
        output_dir=output_dir,
    )
    edu_code_crawler = await edu_code_crawler.crawl()
    docs_dir = pathlib.Path(output_dir) / repo_name
    final_branch_name = edu_code_crawler.final_branch_name
    markdown_converter = BaseMarkdownConverter()
    notebook_converter = NotebookConverter()
    code_chunker = PythonChunker(
        max_chunk_size=chunk_size,
        length_fn=LengthFunction(encoding_model=encoding_model),
    )
    markdown_chunker = MarkdownChunker(
        max_chunk_size=chunk_size,
        length_fn=LengthFunction(encoding_model=encoding_model),
    )
    for item in read_edu_docs(
        docs_dir=pathlib.Path(docs_dir).resolve(),
        base_url=f"https://github.com/{repo_owner}/{repo_name}/blob/{final_branch_name}/",
    ):
        record = markdown_converter(item)
        chunks = markdown_chunker(record)

        for chunk in chunks:
            yield chunk
    for item in read_edu_notebooks(
        docs_dir=pathlib.Path(docs_dir).resolve(),
        base_url=f"https://github.com/{repo_owner}/{repo_name}/blob/{final_branch_name}/",
    ):
        record = notebook_converter(item)
        chunks = markdown_chunker(record)
        for chunk in chunks:
            yield chunk
    for item in read_edu_code(
        docs_dir=pathlib.Path(docs_dir).resolve(),
        base_url=f"https://github.com/{repo_owner}/{repo_name}/blob/{final_branch_name}/",
    ):
        chunks = code_chunker(item)
        for chunk in chunks:
            yield chunk


async def crawl_fc_reports(
    chunk_size: int = 512,
    encoding_model: str = "gpt-4o-mini",
):
    reports_crawler = ReportsCrawler()
    converter = BaseMarkdownConverter()
    chunker = MarkdownChunker(
        max_chunk_size=chunk_size,
        length_fn=LengthFunction(encoding_model=encoding_model),
    )
    async for report in reports_crawler.crawl():
        record = {
            "id": _get_id(report["url"] + report["content"]),
            "url": _fix_url(report["url"]),
            "uri": report["url"],
            "directory": None,
            "content": report["content"],
            "source": "fully-connected",
            "type": "report",
        }
        record = converter(record)
        chunks = chunker(record)
        for chunk in chunks:
            yield chunk


async def crawl_coreweave(
    sitemap_url: str = "https://docs.coreweave.com/sitemap.xml",
    chunk_size: int = 512,
    encoding_model: str = "gpt-4o-mini",
):
    coreweave_crawler = CoreweaveCrawler(sitemap_url=sitemap_url)
    converter = BaseMarkdownConverter()
    chunker = MarkdownChunker(
        max_chunk_size=chunk_size,
        length_fn=LengthFunction(encoding_model=encoding_model),
    )
    async for item in coreweave_crawler.crawl():
        record = {
            "id": _get_id(item["url"] + item["content"]),
            "url": _fix_url(item["url"]),
            "uri": item["url"],
            "directory": None,
            "content": item["content"],
            "source": "coreweave",
            "type": "documentation",
        }
        record = converter(record)
        chunks = chunker(record)
        for chunk in chunks:
            yield chunk


async def write_async_generator_to_file(async_gen, file_handle, lock, semaphore):
    async for item in async_gen:
        async with semaphore:
            async with lock:
                await file_handle.write(json.dumps(item) + "\n")


async def main():
    # Step 1: Crawl and chunk documents

    generators = [
        crawl_wandb_code(chunk_size=CHUNK_SIZE, encoding_model=ENCODING_MODEL),
        crawl_weave_code(chunk_size=CHUNK_SIZE, encoding_model=ENCODING_MODEL),
        crawl_examples(chunk_size=CHUNK_SIZE, encoding_model=ENCODING_MODEL),
        crawl_edu(chunk_size=CHUNK_SIZE, encoding_model=ENCODING_MODEL),
        crawl_wandb_docs(chunk_size=CHUNK_SIZE, encoding_model=ENCODING_MODEL),
        crawl_fc_reports(chunk_size=CHUNK_SIZE, encoding_model=ENCODING_MODEL),
        crawl_coreweave(chunk_size=CHUNK_SIZE, encoding_model=ENCODING_MODEL),
    ]
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent tasks
    async with aiofiles.open("data/chunked_documents.jsonl", "w") as file_handle:
        await asyncio.gather(*[write_async_generator_to_file(gen, file_handle, lock, semaphore) for gen in generators])


if __name__ == "__main__":
    asyncio.run(main())
