import io
import os
import re
import shutil
import zipfile
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerMonitor,
    CrawlerRunConfig,
    DefaultMarkdownGenerator,
    MemoryAdaptiveDispatcher,
    RateLimiter,
)
from jinja2 import Environment, PackageLoader
from lxml.etree import fromstring
from openapi_core import Spec
from openapi_markdown.generator import ref_to_link, ref_to_schema, to_json


# os.environ["CRAWL4_AI_BASE_DIRECTORY"] = str(Path("../data/cache").resolve())


class CodeCrawler:
    def __init__(
        self,
        repo_owner: str = "wandb",
        repo_name: str = "wandb",
        output_dir: str = "../data/code",
        specific_tag: str | None = None,
        branch_name: str | None = None,
        default_branch: str = "main",
    ):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.output_dir = output_dir
        self.specific_tag = specific_tag
        self.branch_name = branch_name
        self.default_branch = default_branch
        self.final_branch_name = None

    async def _get_latest_release_tag(self) -> str | None:
        url = f"https://github.com/{self.repo_owner}/{self.repo_name}/releases/latest"
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(url)
                actual_url = str(response.url)
                tag_match = re.search(r"/releases/tag/([^/]+)", actual_url)
                if tag_match:
                    return tag_match.group(1)
        except Exception as e:
            print(f"Error getting latest release tag: {e}")
        return None

    async def _download_and_extract_release(
        self,
        tag_name: str | None,
        branch_name: str,
        extract_dir: Path,
    ) -> bool | str:
        target_path = Path(extract_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        repo_dir = target_path / self.repo_name
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        if tag_name:
            zip_url = f"https://github.com/{self.repo_owner}/{self.repo_name}/archive/refs/tags/{tag_name}.zip"
            self.final_branch_name = tag_name
        else:
            zip_url = f"https://github.com/{self.repo_owner}/{self.repo_name}/archive/refs/heads/{branch_name}.zip"
            self.final_branch_name = branch_name
        try:
            temp_extract_dir = target_path / f"temp_extract_{self.repo_name}"
            if temp_extract_dir.exists():
                shutil.rmtree(temp_extract_dir)
            temp_extract_dir.mkdir()
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(zip_url)
                response.raise_for_status()
                zip_content = response.content
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_ref:
                zip_ref.extractall(temp_extract_dir)
            extracted_dirs = [
                d for d in os.listdir(temp_extract_dir) if os.path.isdir(os.path.join(temp_extract_dir, d))
            ]
            if extracted_dirs:
                src_dir = os.path.join(temp_extract_dir, extracted_dirs[0])
                dst_dir = os.path.join(target_path, self.repo_name)
                if not os.path.exists(dst_dir):
                    os.mkdir(dst_dir)
                for item in os.listdir(src_dir):
                    s = os.path.join(src_dir, item)
                    d = os.path.join(dst_dir, item)
                    if os.path.isdir(s):
                        if os.path.exists(d):
                            shutil.rmtree(d)
                        shutil.copytree(s, d)
                    else:
                        shutil.copy2(s, d)
                shutil.rmtree(temp_extract_dir)
                return f"{extract_dir}/{self.repo_name}"
            return False
        except Exception:
            return False

    async def crawl(self) -> "CodeCrawler":
        download_dir = Path(self.output_dir)
        tag_name = None
        final_branch_name = self.default_branch
        if self.specific_tag:
            tag_name = self.specific_tag
        elif self.branch_name:
            final_branch_name = self.branch_name
        else:
            tag_name = await self._get_latest_release_tag()
        await self._download_and_extract_release(tag_name, final_branch_name, download_dir)
        return self


class ReportsCrawler:
    def __init__(self, base_url="https://wandb.ai"):
        self.base_url = base_url
        self.fc_url = urljoin(base_url, "/fully-connected")
        self.browser_cfg = BrowserConfig(
            headless=True,
            text_mode=True,
            light_mode=True,
        )
        self.fc_config = CrawlerRunConfig(
            wait_for="css:.sc-jAQbxs.bFBUHn",
            css_selector=".sc-gGTSdS.jZEkjL",
            exclude_external_images=True,
            exclude_social_media_links=True,
            stream=True,
            cache_mode=CacheMode.ENABLED,
            page_timeout=100000,
        )

        self.reports_config = CrawlerRunConfig(
            wait_for='css:div[data-test="wbslate-report-content"]',
            css_selector="""title, meta[name='description'], div[data-test='wbslate-report-content']""",
            excluded_selector="""div[data-test='inline-comment-button'], div[id='comments']""",
            exclude_external_images=True,
            exclude_social_media_links=True,
            stream=True,
            markdown_generator=DefaultMarkdownGenerator(
                options={
                    "ignore_images": True,
                }
            ),
            cache_mode=CacheMode.ENABLED,
            page_timeout=100000,
        )

    def _get_dispatcher(self, num_urls):
        return MemoryAdaptiveDispatcher(
            memory_threshold_percent=80.0,
            check_interval=1.0,
            max_session_permit=10,
            rate_limiter=RateLimiter(
                base_delay=(1.0, 4.0),
                max_delay=30.0,
                max_retries=10,
                rate_limit_codes=[429, 503],
            ),
            monitor=CrawlerMonitor(urls_total=num_urls, enable_ui=False),
        )

    async def crawl_fc(self):
        urls = [self.fc_url] + [f"{self.fc_url}?page={i}" for i in range(2, 55)]
        reports_urls = []
        async with AsyncWebCrawler(config=self.browser_cfg) as crawler:
            results = await crawler.arun_many(
                urls=urls,
                config=self.fc_config,
                dispatcher=self._get_dispatcher(len(urls)),
            )
            async for res in results:
                if res.success:
                    try:
                        soup = BeautifulSoup(res.html, features="html.parser")
                        for tag in soup.find_all("a"):
                            href = tag.get("href")
                            if href and "/reports/" in href:
                                reports_urls.append(href)
                    except Exception as e:
                        logger.warning("Error parsing HTML for %s: %s", res.url, e)
                else:
                    logger.warning("Crawl failed for %s: %s", res.url, res.error_message)
        reports_urls = list(set(reports_urls))
        full_urls = [urljoin(self.base_url, url) for url in reports_urls]
        return full_urls

    async def crawl_reports(self, urls) -> AsyncGenerator[dict, None]:
        async with AsyncWebCrawler(config=self.browser_cfg) as crawler:
            results = await crawler.arun_many(
                urls=urls,
                config=self.reports_config,
                dispatcher=self._get_dispatcher(len(urls)),
            )
            async for res in results:
                if res.success:
                    try:
                        soup = BeautifulSoup(res.html, features="html.parser")
                        title_tag = soup.find("title")
                        title = title_tag.get_text() if title_tag else ""
                        description_tag = soup.find("meta", {"name": "description"})
                        description = description_tag.get("content") if description_tag else ""
                        if description:
                            description = description.strip()
                        else:
                            description = ""
                        markdown_content = res.markdown.strip()
                        markdown_content_lines = markdown_content.splitlines()
                        if markdown_content_lines:
                            _, rest = (
                                markdown_content_lines[0],
                                markdown_content_lines[1:],
                            )
                            title = title.strip("# ").split("|")[0].strip()
                            full_content = f"# {title}\n\n{description}\n\n{chr(10).join(rest)}"
                        else:
                            full_content = description
                        if len(markdown_content) > 10:
                            record = {"url": res.url, "content": full_content}
                            yield record
                    except Exception as e:
                        logger.warning("Error parsing HTML for %s: %s", res.url, e)
                else:
                    logger.warning("Crawl failed for %s: %s", res.url, res.error_message)

    async def crawl(self) -> AsyncGenerator[dict, None]:
        """
        Pipeline: crawl the fully connected page, then crawl the reports.
        Returns a list of report dicts with 'url' and 'content'.
        """
        report_urls = await self.crawl_fc()
        async for item in self.crawl_reports(report_urls):
            yield item


class WeaveServiceApiDocsCrawler:
    def __init__(
        self,
        base_url="https://weave-docs.wandb.ai/reference/service-api",
        openapi_url="https://trace.wandb.ai/openapi.json",
    ):
        self.base_url = base_url
        self.openapi_url = openapi_url

    async def _get_raw_json(self):
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(self.openapi_url)
            response.raise_for_status()
        return response.json()

    def _apply_mapper(self, raw_json, mapper):
        if isinstance(raw_json, dict):
            return mapper({k: self._apply_mapper(v, mapper) for k, v in raw_json.items()})
        elif isinstance(raw_json, list):
            return mapper([self._apply_mapper(v, mapper) for v in raw_json])
        else:
            return mapper(raw_json)

    def _apply_doc_fixes(self, raw_json):
        expr = raw_json.get("components", {}).get("schemas", {}).get("Query", {}).get("properties", {}).get("$expr")
        if expr is not None:
            if "anyOf" in expr:
                del expr["anyOf"]
            expr["type"] = "object"

        remove_keys = [k for k in raw_json.get("components", {}).get("schemas", {}) if k.endswith("Operation")]
        for k in remove_keys:
            del raw_json["components"]["schemas"][k]

        def remove_dependencies_mapper(value):
            if isinstance(value, dict) and "$ref" in value and any(value["$ref"].endswith(k) for k in remove_keys):
                return {"type": "object"}
            return value

        raw_json = self._apply_mapper(raw_json, remove_dependencies_mapper)

        def optional_any_fix_mapper(value):
            if isinstance(value, dict) and "anyOf" in value:
                if value["anyOf"] == [{}, {"type": "null"}]:
                    del value["anyOf"]
                    value["type"] = "object"
            return value

        raw_json = self._apply_mapper(raw_json, optional_any_fix_mapper)

        def optional_dict_fix_mapper(value):
            if isinstance(value, dict) and "anyOf" in value:
                if value["anyOf"] == [{"type": "object"}, {"type": "null"}]:
                    del value["anyOf"]
                    value["type"] = "object"
            return value

        raw_json = self._apply_mapper(raw_json, optional_dict_fix_mapper)

        return raw_json

    def _set_servers(self, raw_json):
        raw_json["servers"] = [{"url": "https://trace.wandb.ai"}]
        return raw_json

    def _render_endpoint_markdown(self, spec_data: dict) -> str:
        spec = Spec.from_dict(spec_data)
        env = Environment(loader=PackageLoader("openapi_markdown", "templates"))
        env.filters |= {"ref_to_link": ref_to_link, "to_json": to_json}
        env.globals["ref_to_schema"] = lambda r: ref_to_schema(r, spec_data)
        template = env.get_template("api_doc_template.md.j2")
        return template.render(spec=spec)

    async def crawl(self) -> AsyncIterator[dict]:
        """
        Returns a list of dicts: [{"url": ..., "content": ...}, ...]
        """
        raw_json = await self._get_raw_json()
        safe_for_docs_json = self._apply_doc_fixes(raw_json)
        spec_data = self._set_servers(safe_for_docs_json)
        skip_keywords = {"health", "version", "server_info"}
        for path, path_item in spec_data["paths"].items():
            for method, operation in path_item.items():
                if not isinstance(operation, dict):
                    continue
                operation_id = operation.get("operationId", "")
                if any(kw in operation_id.lower() for kw in skip_keywords) or any(
                    kw in path.lower() for kw in skip_keywords
                ):
                    continue
                if operation_id:
                    slug = operation_id.replace("_", "-")
                else:
                    slug = path.lstrip("/").replace("/", "-").replace("{", "").replace("}", "")
                output_spec = spec_data.copy()
                output_spec["paths"] = {path: {method: operation}}
                md = self._render_endpoint_markdown(output_spec)
                idx = md.find("# APIs\n") + len("# APIs\n") + 2
                record = {"url": f"{self.base_url}/{slug}", "content": md[idx:]}
                yield record


class CoreweaveCrawler:
    def __init__(self, sitemap_url="https://docs.coreweave.com/sitemap.xml"):
        self.sitemap_url = sitemap_url
        self.browser_cfg = BrowserConfig(
            headless=True,
            text_mode=True,
            light_mode=True,
        )
        self.crawler_config = CrawlerRunConfig(
            wait_for="css:.theme-doc-markdown.markdown",
            css_selector=".theme-doc-markdown.markdown",
            exclude_external_images=True,
            exclude_social_media_links=True,
            stream=True,
            cache_mode=CacheMode.ENABLED,
            page_timeout=60000,
        )

    async def _fetch_sitemap(self, url: str) -> str:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(url)
            return response.text

    def _parse_sitemap(self, sitemap_text: str) -> list[str]:
        root = fromstring(sitemap_text.encode("utf-8"))
        ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = [loc.text for loc in root.findall(".//ns:url/ns:loc", ns)]
        return urls

    def _get_dispatcher(self, num_urls):
        return MemoryAdaptiveDispatcher(
            memory_threshold_percent=80.0,
            check_interval=1.0,
            max_session_permit=10,
            rate_limiter=RateLimiter(
                base_delay=(1.0, 4.0),
                max_delay=30.0,
                max_retries=10,
                rate_limit_codes=[429, 503],
            ),
            monitor=CrawlerMonitor(urls_total=num_urls, enable_ui=False),
        )

    async def crawl(self) -> AsyncIterator[dict]:
        """
        Crawls the coreweave documentation from the sitemap.
        Returns an async iterator of document dicts with 'url' and 'content'.
        """
        sitemap_text = await self._fetch_sitemap(self.sitemap_url)
        urls = self._parse_sitemap(sitemap_text)

        async with AsyncWebCrawler(config=self.browser_cfg) as crawler:
            results = await crawler.arun_many(
                urls=urls,
                config=self.crawler_config,
                dispatcher=self._get_dispatcher(len(urls)),
            )
            async for res in results:
                if res.success:
                    content = str(res.markdown)
                    if content.strip():
                        yield {"url": res.url, "content": content}
