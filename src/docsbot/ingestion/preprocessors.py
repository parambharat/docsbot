import hashlib
import pathlib
import re
import textwrap
import uuid
from urllib.parse import quote, urljoin

import ftfy
import nbformat
import pypandoc
import yaml
from multilspy import LanguageServer
from multilspy.multilspy_config import MultilspyConfig
from multilspy.multilspy_logger import MultilspyLogger
from nbconvert import MarkdownExporter


class BaseMarkdownConverter:
    def __init__(self):
        self.multiple_newlines_re = re.compile(r"\n{3,}")
        self.html_comment_re = re.compile(r"<!--[\s\S]*?-->", re.MULTILINE)
        self.svg_re = re.compile(r"<svg[\s\S]*?</svg>", re.MULTILINE)
        self.frontmatter_re = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
        self.link_re = re.compile(r"\[(.*?)\]\((.*?)\)")
        self.html_link_re = re.compile(r'<a [^>]*href="[^"]*"[^>]*>(.*?)</a>', re.DOTALL)
        self.img_md_re = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
        self.img_html_re = re.compile(r'<img [^>]*src="[^"]*"[^>]*alt="([^"]*)"[^>]*>', re.DOTALL)
        self.img_shortcode_re = re.compile(r'{{< img src="(.*?)".*?alt="(.*?)".*?>}}')
        self.relref_re = re.compile(r'{{<\s*relref "(.*?)"\s*>}}')
        self.link_ext_re = re.compile(r"\(([^)]+)\.mdx?\)")

    def _collapse_blank_lines(self, s: str) -> str:
        return self.multiple_newlines_re.sub("\n\n", s)

    def _normalize_links(self, content):
        content = self.relref_re.sub(r"\1", content)
        content = self.link_ext_re.sub(lambda m: f"({m.group(1)})", content)
        content = self.link_re.sub(r"\1", content)
        content = self.html_link_re.sub(r"\1", content)
        return content

    def _normalize_images(self, content):
        content = self.img_shortcode_re.sub(lambda m: f"![{m.group(2)}]({m.group(1)})", content)

        def md_img_replacer(match):
            alt = match.group(1).strip()
            return alt if alt else ""

        content = self.img_md_re.sub(md_img_replacer, content)

        def html_img_replacer(match):
            alt = match.group(1).strip()
            return alt if alt else ""

        content = self.img_html_re.sub(html_img_replacer, content)
        return content

    def _remove_html_comments_and_svgs(self, content):
        content = self.html_comment_re.sub("", content)
        content = self.svg_re.sub("", content)
        return content

    def _final_normalize(self, content):
        content = ftfy.fix_text(content)
        _, content = self._extract_frontmatter_and_body(content)
        try:
            return pypandoc.convert_text(
                content,
                to="gfm",
                format="markdown-implicit_figures",
                extra_args=["--wrap=none"],
            )
        except RuntimeError:
            return content

    def _extract_frontmatter_and_body(self, content):
        match = self.frontmatter_re.match(content)
        if match:
            frontmatter = yaml.safe_load(match.group(1))
            body = content[match.end() :]
            return frontmatter, body
        return {}, content

    def finalize_content(self, content):
        content = self._collapse_blank_lines(content)
        content = self._normalize_links(content)
        content = self._normalize_images(content)
        return self._final_normalize(content)

    def __call__(self, doc):
        content = doc["content"]
        doc["raw_content"] = doc["content"]
        content = self._remove_html_comments_and_svgs(content)
        frontmatter, content = self._extract_frontmatter_and_body(content)
        title = frontmatter.get("title", "")
        description = frontmatter.get("description", "")
        header_md = f"# {title}\n\n{description}\n\n"
        final_content = header_md + content.strip() if header_md.strip() != "#" else content.strip()
        doc["content"] = self.finalize_content(final_content)
        return doc


class WandbMarkdownConverter(BaseMarkdownConverter):
    def __init__(self):
        super().__init__()
        self.alert_re = re.compile(r"{{%? alert(.*?)%?}}(.*?){{%? /alert %?}}", flags=re.DOTALL)
        self.tabpane_re = re.compile(r"{{< tabpane.*?>}}(.*?){{< /tabpane >}}", flags=re.DOTALL)
        self.tab_re = re.compile(r'{{%? tab header="(.*?)".*?%?}}(.*?){{%? /tab %?}}', flags=re.DOTALL)
        self.cardpane_re = re.compile(r"{{< cardpane >}}(.*?){{< /cardpane >}}", flags=re.DOTALL)
        self.card_re = re.compile(r"{{< card >}}(.*?){{< /card >}}", flags=re.DOTALL)
        self.card_title_re = re.compile(r'<a href="([^"]+)">.*?<h2.*?>(.*?)<\\/h2>', flags=re.DOTALL)
        self.card_desc_re = re.compile(r"<p.*?>(.*?)<\\/p>", flags=re.DOTALL)
        self.cta_button_re = re.compile(r'{{< cta-button .*?(\w+Link)="(.*?)".*?>}}')
        self.prism_re = re.compile(r'{{< prism file="(.*?)" title="(.*?)">}}{{< \/prism >}}')
        self.generic_shortcode_re = re.compile(r"{{<.*?>}}")
        self.generic_shortcode_percent_re = re.compile(r"{{%.*?%}}")

    def _alert_replacer(self, match):
        params = match.group(1).strip()
        inner_content = match.group(2).strip()
        title_match = re.search(r'title="(.*?)"', params)
        color_match = re.search(r'color="(.*?)"', params)
        title = title_match.group(1) if title_match else "Note"
        color = color_match.group(1).capitalize() if color_match else "Note"
        header = f"**{title}:**" if title_match else f"**{color}:**"
        lines = inner_content.split("\n")
        return "\n".join(["> " + header] + ["> " + line for line in lines])

    def _tabpane_replacer(self, match):
        tabpane_content = match.group(1)

        def tab_replacer(tab_match):
            header = tab_match.group(1)
            body = tab_match.group(2).strip()
            return f"\n### {header}\n\n{body}\n"

        tabpane_markdown = self.tab_re.sub(tab_replacer, tabpane_content)
        return tabpane_markdown

    def _cardpane_replacer(self, match):
        cardpane_content = match.group(1)

        def card_replacer(card_match):
            card_content = card_match.group(1)
            title_match = self.card_title_re.search(card_content)
            desc_match = self.card_desc_re.search(card_content)
            title = title_match.group(2).strip() if title_match else "Title"
            link = title_match.group(1).strip() if title_match else "#"
            desc = desc_match.group(1).strip() if desc_match else ""
            return f"- **[{title}]({link})**: {desc}\n"

        cardpane_markdown = self.card_re.sub(card_replacer, cardpane_content)
        return cardpane_markdown

    def __call__(self, record):
        content = record["content"]
        record["raw_content"] = content
        content = self._remove_html_comments_and_svgs(content)
        frontmatter, content = self._extract_frontmatter_and_body(content)
        title = frontmatter.get("title", "")
        description = frontmatter.get("description", "")
        header_md = f"# {title}\n\n{description}\n\n"

        content = self.alert_re.sub(self._alert_replacer, content)
        content = self.tabpane_re.sub(self._tabpane_replacer, content)
        content = self.cardpane_re.sub(self._cardpane_replacer, content)
        content = self.cta_button_re.sub("", content)
        content = self.prism_re.sub(r"**\\2** (`\\1`)", content)
        content = self.generic_shortcode_re.sub("", content)
        content = self.generic_shortcode_percent_re.sub("", content)

        full_content = header_md + content.strip() if header_md.strip() != "#" else content.strip()
        record["content"] = self.finalize_content(full_content)
        return record


class WeaveMarkdownConverter(BaseMarkdownConverter):
    def __init__(self):
        super().__init__()
        self.import_re = re.compile(r"^import .*?$", flags=re.MULTILINE)
        self.admonition_re = re.compile(r":::([a-zA-Z]+)\s*(.*?)\n([\s\S]*?)\n:::", flags=re.DOTALL)
        self.heading_re = re.compile(
            r'<Heading[^>]*as=\{"h([1-6])"\}[^>]*children=\{"([^"]+)"\}[^>]*>[\s\S]*?</Heading>'
        )
        self.badge_button_re = re.compile(
            r'<a [^>]*class="[^"]*notebook-cta-button[^"]*"[^>]*>[\s\S]*?</a>',
            re.IGNORECASE,
        )
        self.methodendpoint_re = re.compile(
            r'<MethodEndpoint[^>]*method=\{"([a-zA-Z]+)"\}[^>]*path=\{"([^"]+)"\}[^>]*>[\s\S]*?</MethodEndpoint>'
        )
        self.tabs_re = re.compile(r"<(?:Tabs|ApiTabs)[^>]*>([\s\S]*?)</(?:Tabs|ApiTabs)>")
        self.tabitem_re = re.compile(r'<TabItem[^>]*label=\{"([^"]+)"[^>]*>([\s\S]*?)</TabItem>')
        self.prettier_ignore_re = re.compile(r"{/?\s*prettier-ignore\s*}")
        self.jsx_html_re = re.compile(r"<([^/][^ >]*)(?: [^>]*)?>([\s\S]*?)</\\1>")

    def _admonition(self, m):
        kind = m.group(1).capitalize()
        title = m.group(2).strip() or kind
        body = m.group(3).strip()
        return "> **%s:** %s" % (title, body.replace("\n", "\n> "))

    def _heading(self, m):
        level = int(m.group(1))
        text = m.group(2).strip()
        return "\n" + "#" * level + " " + text + "\n"

    def _tabitem(self, m):
        label, body = m.group(1), m.group(2).strip()
        return f"\n### {label}\n\n{body}\n"

    def _tabs(self, m):
        inner = self.prettier_ignore_re.sub("", m.group(1))
        return self.tabitem_re.sub(self._tabitem, inner)

    def __call__(self, doc):
        content = doc["content"]
        doc["raw_content"] = content
        content = self._remove_html_comments_and_svgs(content)
        _, content = self._extract_frontmatter_and_body(content)
        content = self.badge_button_re.sub("", content)
        content = self.import_re.sub("", content)
        content = self.admonition_re.sub(self._admonition, content)
        content = self.heading_re.sub(self._heading, content)
        content = self.methodendpoint_re.sub(lambda m: f"**{m.group(1).upper()} {m.group(2)}**", content)
        content = self.tabs_re.sub(self._tabs, content)
        content = self.jsx_html_re.sub(lambda m: m.group(2), content)
        doc["content"] = self.finalize_content(content.strip())
        return doc


class NotebookConverter(BaseMarkdownConverter):
    def __init__(self):
        super().__init__()
        self.docusaurus_meta_re = re.compile(
            r"<!--\s*docusaurus_head_meta::start[\s\S]*?docusaurus_head_meta::end\s*-->",
            re.MULTILINE,
        )

    def _clean_markdown(self, markdown):
        markdown = self.docusaurus_meta_re.sub("", markdown)

        return markdown

    def _extract_title_from_front_matter(self, nb):
        if not nb.cells or nb.cells[0].cell_type != "markdown":
            return None
        cell = nb.cells[0]
        docusaurus_meta_match = re.search(
            r"<!--\s*docusaurus_head_meta::start\s*\n(.*?)\n\s*docusaurus_head_meta::end\s*-->",
            cell.source,
            re.DOTALL,
        )
        if docusaurus_meta_match:
            yaml_block = docusaurus_meta_match.group(1)
            frontmatter, _ = self._extract_frontmatter_and_body(yaml_block)
            return frontmatter.get("title") if frontmatter else None
        frontmatter, _ = self._extract_frontmatter_and_body(cell.source)
        return frontmatter.get("title") if frontmatter else None

    def _clear_outputs(self, nb):
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        return nb

    def __call__(self, doc):
        content = doc["content"]
        doc["raw_content"] = content
        content = self._remove_html_comments_and_svgs(content)
        nb = nbformat.reads(content, as_version=4)
        title = self._extract_title_from_front_matter(nb)
        nb = self._clear_outputs(nb)
        exporter = MarkdownExporter()
        exporter.markdown_format = "gfm"
        body, _ = exporter.from_notebook_node(nb)
        body = self._clean_markdown(body)
        if title:
            body = f"# {title}\n\n{body}"
        doc["content"] = self.finalize_content(body)
        return doc


class PythonRepoConverter:
    def __init__(self, repo_root: pathlib.Path, repo_url: str):
        self.repo_root = repo_root.resolve()
        self.repo_url = repo_url
        self.skip_dirs = {"site-packages", ".venv", "venv", "__pypackages__"}
        self.kind_to_name = {
            5: "Class",
            6: "Method",
            12: "Function",
        }
        self.config = MultilspyConfig.from_dict({"code_language": "python"})
        self.logger = MultilspyLogger()
        self.lsp = LanguageServer.create(self.config, self.logger, str(self.repo_root))

    def _get_id(self, path, code_snippet):
        return str(
            uuid.uuid3(
                uuid.UUID(bytes=hashlib.md5(str(path).encode()).digest()),
                (str(path) + code_snippet),
            )
        )

    def _is_in_repo(self, path: pathlib.Path) -> bool:
        if any(part in self.skip_dirs for part in path.parts):
            return False
        try:
            return path.resolve().is_relative_to(self.repo_root)
        except Exception:
            return str(path.resolve()).startswith(str(self.repo_root))

    def _is_external_definition(self, defn):
        if not defn or not isinstance(defn, list):
            return False
        for d in defn:
            abs_path = d.get("absolutePath") or d.get("path") or ""
            if abs_path and not str(pathlib.Path(abs_path).resolve()).startswith(str(self.repo_root)):
                return True
        return False

    def _github_url_from_path_and_range(self, defn, fallback_path=None) -> str | None:
        """
        Accepts a definition object (defn[0]) and an optional fallback_path.
        Extracts the file path and range, and returns the GitHub URL.
        """
        if defn and isinstance(defn, list) and defn[0].get("range"):
            defn = defn[0]
            file_path = defn.get("absolutePath") or defn.get("path") or fallback_path
            if file_path is None or "range" not in defn:
                return None
            rel_path = self.repo_root.name / pathlib.Path(file_path).resolve().relative_to(self.repo_root)
            rel_path_str = quote(str(rel_path).replace("\\", "/"))
            start = defn["range"]["start"]
            end = defn["range"]["end"]
            start_line = start["line"] + 1
            end_line = end["line"] + 1
            start_char = start["character"]
            end_char = end["character"]
            anchor = f"#L{start_line}C{start_char}-L{end_line}C{end_char}"
            base_url = urljoin(f"{self.repo_url.rstrip('/')}/", f"{rel_path_str}{anchor}")
            return base_url
        return None

    async def harvest(self):
        symbols = []
        async with self.lsp.start_server() as ls:
            for path in self.repo_root.rglob("*.py"):
                if not self._is_in_repo(path):
                    continue
                uri = str(path)
                content = path.read_text()
                try:
                    doc_symbols = await ls.request_document_symbols(uri)
                    for s in doc_symbols[0]:
                        if s["kind"] not in (5, 6, 12):
                            continue
                        try:
                            start = s["selectionRange"]["start"]
                            defn = await ls.request_definition(uri, start["line"], start["character"])
                            if self._is_external_definition(defn):
                                continue
                            hover = await ls.request_hover(uri, start["line"], start["character"])
                            code_snippet = "\n".join(
                                content.splitlines()[s["range"]["start"]["line"] : s["range"]["end"]["line"] + 1]
                            )
                            github_url = self._github_url_from_path_and_range(defn[0], fallback_path=path)
                            record = {
                                "id": self._get_id(path, code_snippet),
                                "symbol": s["name"],
                                "kind": self.kind_to_name.get(s["kind"], "Unknown"),
                                "full_name": (hover["contents"]["value"].split("**Full name:**")[-1] if hover else ""),
                                "code": textwrap.dedent(code_snippet),
                                "url": github_url,
                            }
                            symbols.append(record)
                        except Exception as e:
                            print(f"Lookup failed for {s['name']} at {start}: {e}")
                except Exception as e:
                    print(f"Lookup failed for {path}: {e}")
        return symbols
