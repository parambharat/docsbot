import re
from copy import deepcopy
from typing import Any

import tree_sitter_markdown as tsm
from langchain_text_splitters import (
    Language as LangChainLanguage,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from litellm import encode
from tree_sitter import Language, Node, Parser

from docsbot.utils import _get_id, get_logger

logger = get_logger(__name__)


def get_parser(language: str):
    """
    Get the parser for the specified language.
    """
    if language == "markdown" or language == "python":
        lang = Language(tsm.language())
    else:
        raise ValueError(f"Unsupported language: {language}")
    parser = Parser(lang)
    return parser





class LengthFunction:
    def __init__(self, encoding_model=None):
        self.encoding_model = encoding_model

    @property
    def length_fn(self):
        def normalize_whitespace(text):
            """
            Normalize all whitespace or whitespace-like characters in the text.
            """
            return re.sub(r"\s+", " ", text).strip()

        def _length_fn(text):
            """
            Returns the length of the text in tokens.
            """
            if self.encoding_model:
                return len(encode(model=self.encoding_model, text=normalize_whitespace(text)))
            else:
                return len(normalize_whitespace(text))

        return _length_fn

    def __call__(self, text):
        """
        Returns the length of the text in tokens.
        """
        return self.length_fn(text)


class MarkdownChunker:
    def __init__(self, max_chunk_size: int = 512, length_fn: LengthFunction = None):
        self.max_chunk_size = max_chunk_size
        self.length_fn = length_fn
        self.parser = get_parser("markdown")
        self.fallback_splitter = RecursiveCharacterTextSplitter.from_language(
            language=LangChainLanguage.MARKDOWN,
            chunk_size=self.max_chunk_size * 2,
            chunk_overlap=0,
            length_function=self.length_fn,
        )

    def _extract_text(self, node: Node) -> str:
        return node.text.decode("utf-8").strip()

    def _get_heading_level(self, node: Node) -> int:
        return node.children[0].text.decode("utf-8").count("#")

    def _create_section(self, node: Node, heading_level: int) -> dict:
        return {
            "title": self._extract_text(node),
            "level": heading_level,
            "node_id": _get_id(node.text.decode("utf-8")),
            "type": f"heading_{heading_level}",
            "start_byte": node.start_byte,
            "end_byte": node.end_byte,
            "children": [],
        }

    def _create_content_block(self, node: Node) -> dict:
        return {
            "type": node.type,
            "text": self._extract_text(node),
            "node_id": _get_id(node.text.decode("utf-8")),
            "start_byte": node.start_byte,
            "end_byte": node.end_byte,
            "children": [],
        }

    def _node_to_json(self, node: Node) -> dict:
        root_json = {
            "children": [],
            "node_id": _get_id(node.text.decode("utf-8")),
            "type": "document",
        }
        section_stack = [root_json]

        for child in node.named_children:
            if child.type == "section":
                new_section = {
                    "type": "section",
                    "node_id": _get_id(child.text.decode("utf-8")),
                    "start_byte": child.start_byte,
                    "end_byte": child.end_byte,
                    "children": self._node_to_json(child)["children"],
                }
                section_stack[-1]["children"].append(new_section)
            elif child.type.startswith("atx_heading"):
                heading_level = self._get_heading_level(child)
                new_section = self._create_section(child, heading_level)

                while len(section_stack) > 1 and section_stack[-1].get("level", 0) >= heading_level:
                    section_stack.pop()

                section_stack[-1]["children"].append(new_section)
                section_stack.append(new_section)
            elif child.type in [
                "paragraph",
                "fenced_code_block",
                "table",
                "tight_list",
                "loose_list",
                "block_quote",
                "indented_code_block",
                "html_block",
                "list",
                "pipe_table",
                "block_continuation",
            ]:
                current_section = section_stack[-1]
                if (
                    current_section["children"]
                    and current_section["children"][-1]["type"] == child.type
                    and child.type in ["paragraph", "block_continuation"]
                ):
                    prev_node = current_section["children"][-1]
                    prev_node["text"] += "\n\n" + self._extract_text(child)
                    prev_node["end_byte"] = child.end_byte
                else:
                    content_block = self._create_content_block(child)
                    current_section["children"].append(content_block)

                if current_section != root_json:
                    current_section["end_byte"] = max(current_section.get("end_byte", 0), child.end_byte)
            else:
                logger.warning(
                    f"Unhandled node type: {child.type} at bytes {child.start_byte}-{child.end_byte}"
                    f"child.text: {child.text.decode('utf-8')[:100]}"
                )

        return root_json

    def _convert_to_tree(self, content):
        doc_text = content.encode("utf-8")
        tree = self.parser.parse(doc_text)
        return self._node_to_json(tree.root_node)["children"]

    def node_to_markdown(self, node):
        if node["type"].startswith("heading_"):
            level = int(node["type"].split("_")[1])
            return "#" * level + " " + node["title"].split("#" * level, 1)[-1].strip()
        elif "text" in node:
            return node["text"]
        else:
            # Handle nodes without text (e.g., sections)
            return ""

    def get_context_header(self, path):
        if not path:
            return ""

        header = []
        for heading in path:
            level = int(heading["type"].split("_")[1])
            title = heading["title"].split("#" * level, 1)[-1].strip()
            header.append("#" * level + " " + title)

        return "\n".join(header) + "\n\n... [CONTENT TRUNCATED] ...\n\n"

    def process_nodes(self, nodes, current_chunk, path, chunks):
        if path is None:
            path = []

        for node in nodes:
            node_text = self.node_to_markdown(node)

            current_path = path.copy()
            if node["type"].startswith("heading_"):
                current_path.append(node)

            combined_text = current_chunk
            if current_chunk and node_text:
                combined_text += "\n\n" + node_text
            else:
                combined_text += node_text

            if current_chunk and self.length_fn(combined_text) > self.max_chunk_size:
                current_chunk += "\n\n... [NEXT CONTENT TRUNCATED] ..."
                chunks.append(current_chunk)

                current_chunk = self.get_context_header(current_path)
                current_chunk += node_text
            else:
                current_chunk = combined_text

            if node.get("children"):
                current_chunk = self.process_nodes(node["children"], current_chunk, current_path, chunks)

        return current_chunk

    def chunk_markdown_tree(self, tree):
        """
        Chunk a markdown tree into semantically coherent pieces of maximum size.

        Args:
            tree: list of nodes from _convert_to_tree

        Returns:
            list of chunks, each a string containing markdown text
        """
        chunks = []  # Local variable instead of instance attribute

        final_chunk = self.process_nodes(tree, "", None, chunks)

        if final_chunk:
            chunks.append(final_chunk)

        return chunks

    def __call__(self, doc):
        content = doc["content"]
        content_lines = content.split("\n")
        content_lines = list(filter(lambda x: x.strip(), content_lines))
        if not len(content_lines) > 5:
            return []
        tree = self._convert_to_tree(content)
        chunks = self.chunk_markdown_tree(tree)
        final_chunks = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            elif self.length_fn(chunk) > self.max_chunk_size * 2:
                fallback_chunks = self.fallback_splitter.split_text(chunk)
                for fallback_chunk in fallback_chunks:
                    if not fallback_chunk.strip():
                        continue
                    chunk_doc = deepcopy(doc)
                    chunk_doc["doc_id"] = doc["id"]
                    chunk_doc["content"] = fallback_chunk
                    chunk_doc["id"] = _get_id(fallback_chunk)
                    final_chunks.append(chunk_doc)
                continue
            chunk_doc = deepcopy(doc)
            chunk_doc["doc_id"] = doc["id"]
            chunk_doc["content"] = chunk
            chunk_doc["id"] = _get_id(chunk)
            final_chunks.append(chunk_doc)
        return final_chunks


class Block:
    def __init__(
        self,
        text: str,
        length: int,
        block_type: str,
        parent: str | None = None,
    ):
        self.text = text
        self.length = length
        self.block_type = block_type
        self.parent = parent  # e.g. class name for methods


class PythonChunker:
    """
    Splits Python code into semantic blocks and packs them into
    context-aware chunks around max_chunk_size, preserving
    functions, classes, methods, decorators, and comments.
    """

    def __init__(
        self,
        max_chunk_size: int = 512,
        length_fn: Any | None = None,
        encoding_model: str | None = None,
    ):
        self.max_chunk_size = max_chunk_size
        if length_fn:
            self.length_fn = length_fn
        else:
            self.length_fn = self._make_length_fn(encoding_model)
        self.parser = get_parser("python")
        self.fallback_splitter = RecursiveCharacterTextSplitter.from_language(
            language=LangChainLanguage.PYTHON,
            chunk_size=self.max_chunk_size * 2,
            chunk_overlap=0,
            length_function=self.length_fn,
        )

    def _make_length_fn(self, encoding_model: str | None):
        def normalize_ws(text: str) -> str:
            return re.sub(r"\s+", " ", text).strip()

        def fn(text: str) -> int:
            txt = normalize_ws(text)
            if encoding_model:
                return len(encode(model=encoding_model, text=txt))
            return len(txt)

        return fn

    def _get_source_lines(self, source: bytes) -> list[str]:
        return source.decode("utf-8").splitlines(keepends=True)

    def _collect_leading(self, lineno: int, lines: list[str]) -> int:
        i = lineno
        while i > 0 and (lines[i - 1].strip().startswith("#") or lines[i - 1].strip() == ""):
            i -= 1
        return i

    def _collect_blocks(self, root: Node, lines: list[str]) -> list[Block]:
        blocks: list[Block] = []
        for child in root.named_children:
            ctype = child.type
            # imports
            if ctype in ("import_statement", "import_from_statement"):
                start = child.start_point[0]
                lead = self._collect_leading(start, lines)
                end = child.end_point[0]
                text = "".join(lines[lead : end + 1])
                blocks.append(Block(text, self.length_fn(text), "import"))

            # decorated definitions
            elif ctype == "decorated_definition":
                start = child.start_point[0]
                lead = self._collect_leading(start, lines)
                end = child.end_point[0]
                text = "".join(lines[lead : end + 1])
                def_node = child.child_by_field_name("definition")
                if def_node and def_node.type == "function_definition":
                    blocks.append(Block(text, self.length_fn(text), "decorated_function"))
                elif def_node and def_node.type == "class_definition":
                    blocks.append(Block(text, self.length_fn(text), "decorated_class"))
                else:
                    blocks.append(Block(text, self.length_fn(text), "decorated_unknown"))

            # functions
            elif ctype == "function_definition":
                start = child.start_point[0]
                lead = self._collect_leading(start, lines)
                end = child.end_point[0]
                text = "".join(lines[lead : end + 1])
                blocks.append(Block(text, self.length_fn(text), "function"))

            # classes
            elif ctype == "class_definition":
                cname = child.child_by_field_name("name").text.decode()
                start = child.start_point[0]
                lead = self._collect_leading(start, lines)
                hdr_end = child.start_point[0]
                header_text = "".join(lines[lead : hdr_end + 1])
                blocks.append(
                    Block(
                        header_text,
                        self.length_fn(header_text),
                        "class_header",
                        parent=cname,
                    )
                )
                body = child.child_by_field_name("body")
                if body:
                    # class-level attrs
                    for stmt in body.named_children:
                        stype = stmt.type
                        if stype in (
                            "assignment",
                            "expression_statement",
                            "pass_statement",
                            "import_statement",
                            "import_from_statement",
                        ):
                            sstart = stmt.start_point[0]
                            lead_s = self._collect_leading(sstart, lines)
                            send = stmt.end_point[0]
                            text = "".join(lines[lead_s : send + 1])
                            blocks.append(
                                Block(
                                    text,
                                    self.length_fn(text),
                                    "class_attr",
                                    parent=cname,
                                )
                            )
                    # methods and decorated methods
                    for stmt in body.named_children:
                        stype = stmt.type
                        if stype == "decorated_definition":
                            sstart = stmt.start_point[0]
                            lead_s = self._collect_leading(sstart, lines)
                            send = stmt.end_point[0]
                            text = "".join(lines[lead_s : send + 1])
                            # always method
                            blocks.append(
                                Block(
                                    text,
                                    self.length_fn(text),
                                    "decorated_method",
                                    parent=cname,
                                )
                            )
                        elif stype == "function_definition":
                            sstart = stmt.start_point[0]
                            lead_s = self._collect_leading(sstart, lines)
                            send = stmt.end_point[0]
                            text = "".join(lines[lead_s : send + 1])
                            blocks.append(Block(text, self.length_fn(text), "method", parent=cname))

            # main guard
            elif ctype == "if_statement":
                start = child.start_point[0]
                lead = self._collect_leading(start, lines)
                end = child.end_point[0]
                text = "".join(lines[lead : end + 1])
                if "__name__" in text and "main" in text:
                    blocks.append(Block(text, self.length_fn(text), "main"))
                else:
                    blocks.append(Block(text, self.length_fn(text), "stmt"))

            # other top-level statements
            elif ctype in (
                "for_statement",
                "while_statement",
                "with_statement",
                "expression_statement",
                "assignment",
            ):
                start = child.start_point[0]
                lead = self._collect_leading(start, lines)
                end = child.end_point[0]
                text = "".join(lines[lead : end + 1])
                blocks.append(Block(text, self.length_fn(text), "stmt"))

            # else: ignore comments, nested definitions already in function/class bodies
        return blocks

    def _pack_blocks(self, blocks: list[Block]) -> list[str]:
        # map class headers and trunc markers
        headers: dict[str, Block] = {}
        truncs: dict[str, Block] = {}
        for b in blocks:
            if b.block_type == "class_header" and b.parent:
                headers[b.parent] = b
                indent = re.match(r"^(\s*)", b.text).group(1)
                txt = f"{indent}# ... [class body truncated] ...\n"
                truncs[b.parent] = Block(txt, self.length_fn(txt), "class_trunc", parent=b.parent)

        chunks: list[str] = []
        curr_blocks: list[Block] = []
        curr_len = 0

        def flush():
            nonlocal curr_blocks, curr_len
            if curr_blocks:
                chunks.append("".join(b.text for b in curr_blocks))
            curr_blocks = []
            curr_len = 0

        for b in blocks:
            if b.block_type == "import":
                continue
            # flush if adding b would overflow soft limit
            if curr_len > 0 and curr_len + b.length > self.max_chunk_size:
                flush()
            # inject class context for methods in new chunk
            if b.block_type in ("method", "decorated_method") and curr_len == 0:
                hdr = headers.get(b.parent)
                if hdr:
                    curr_blocks.append(hdr)
                    curr_len += hdr.length
                    curr_blocks.append(truncs[b.parent])
                    curr_len += truncs[b.parent].length
            curr_blocks.append(b)
            curr_len += b.length
            # break context on non-methods
            if b.block_type not in (
                "method",
                "decorated_method",
                "class_attr",
                "class_header",
                "class_trunc",
            ):
                # next chunk won't retain headers automatically
                pass
        flush()
        return chunks

    def _attach_imports(self, chunks: list[str], imports: list[Block]) -> list[str]:
        imp_list: list[tuple] = []
        for imp in imports:
            txt = imp.text.strip()
            after = txt.split("import", 1)[1]
            names = [p.strip().split(" as ")[-1] for p in after.split(",")]
            imp_list.append((names, txt))
        final: list[str] = []
        for chunk in chunks:
            needed = []
            for names, txt in imp_list:
                for nm in names:
                    if re.search(rf"\b{re.escape(nm)}\b", chunk) and txt not in needed:
                        needed.append(txt)
                        break
            prefix = ""
            if needed:
                prefix = "\n".join(needed) + "\n\n"
            final.append(prefix + chunk)
        return final

    def has_functions_or_classes(self, root: Node) -> bool:
        """
        Return True if the given Python module source contains at least one top-level
        function, class, or decorated definition; False otherwise.

        Args:
            source: Python source code as a string.
        """
        # Check only top-level definitions
        for child in root.named_children:
            if child.type in (
                "function_definition",
                "class_definition",
                "decorated_definition",
            ):
                return True
        return False

    def __call__(self, doc: dict[str, Any]) -> list[dict[str, Any]]:
        content = doc.get("content", "")
        src = content.encode("utf-8")
        lines = self._get_source_lines(src)
        tree = self.parser.parse(src)
        root = tree.root_node
        if not self.has_functions_or_classes(root):
            return []
        blocks = self._collect_blocks(root, lines)
        imports = [b for b in blocks if b.block_type == "import"]
        others = [b for b in blocks if b.block_type != "import"]

        packed = self._pack_blocks(others)
        final_chunks = self._attach_imports(packed, imports)

        docs: list[dict[str, Any]] = []
        for text in final_chunks:
            if not text.strip():
                continue
            elif self.length_fn(text) > self.max_chunk_size * 2:
                fallback_chunks = self.fallback_splitter.split_text(text)
                for fallback_chunk in fallback_chunks:
                    if not fallback_chunk.strip():
                        continue
                    chunk_doc = deepcopy(doc)
                    chunk_doc["doc_id"] = doc["id"]
                    chunk_doc["content"] = fallback_chunk
                    chunk_doc["id"] = _get_id(fallback_chunk)
                    docs.append(chunk_doc)
                continue
            cd = deepcopy(doc)
            cd["doc_id"] = doc.get("id")
            cd["content"] = text
            cd["id"] = _get_id(text)
            docs.append(cd)
        return docs


def main():
    import json

    length_fn = LengthFunction(encoding_model="gpt-4.1-mini")

    print("\nPython Chunker Example:")
    chunker = MarkdownChunker(max_chunk_size=768, length_fn=length_fn)
    with open("../data/chunked_documents.jsonl") as f:
        for line in f:
            record = json.loads(line)
            chunks = chunker(record)
            for chunk in chunks:
                print("\n--- Chunk ---")
                print(json.dumps(chunk, indent=2))
                print("--------------------")


if __name__ == "__main__":
    main()
