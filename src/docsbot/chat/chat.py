import asyncio
import datetime
import html
import inspect
import pathlib
import re
import uuid
from typing import Any

import weave
from agents import Agent, Runner, RunContextWrapper, TResponseInputItem, function_tool
from agents.extensions.models.litellm_model import LitellmModel
from openai.types.responses import EasyInputMessageParam
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from docsbot.chat.agent_runner import EarlyExitRunConfig, EarlyExitRunner
from docsbot.chat.context import DocsContext
from docsbot.retriever import ChromaRetrieverWithReranker
from docsbot.retriever.reranker import Rerank, RerankSettings
from docsbot.retriever.retriever import ChromaRetrieverSettings

weave.init("parambharat/wandbot-agents")


@weave.op
def format_doc(doc: dict, text_field: str = "text") -> str:
    """
    Format a document dictionary into a string representation with an <a> tag for the source.
    """
    metadata = doc["metadata"]
    document_type = f'<source><a href="{metadata["url"]}">{metadata["source"]}: {metadata["type"]}</a></source>'
    content = f"<excerpt>\n{html.escape(doc[text_field])}\n</excerpt>"
    formatted_doc = document_type + content
    return formatted_doc


@weave.op
def format_retrieval(docs: list[dict], text_field: str = "text") -> str:
    """
    Format a list of document dictionaries into a string representation.
    """
    formatted_docs = ""
    for idx, doc in enumerate(docs):
        formatted_docs += f"\n<doc_{idx}>"
        formatted_docs += format_doc(doc, text_field)
        formatted_docs += f"</doc_{idx}>\n"

    xml_string = f"<retrieval>{formatted_docs}</retrieval>"

    xml_string = re.sub(r"\n{2,}", "\n", xml_string)
    return html.unescape(xml_string)


class ExpertQuestion(BaseModel):
    """A question to ask an expert"""

    question: str = Field(
        ...,
        description="A detailed natural language question to address a user request",
        examples=[
            "How can I track wandb model metadata?",
            "How to get the metrics from a weave project?",
            "Where to monitor node metrics in coreweave?",
        ],
    )


class Citation(BaseModel):
    """A citation or reference to a retrieved document/source"""

    source: str = Field(..., description="The source of the citation or reference")
    url: str = Field(..., description="The url of the retrieved document/source")


class CitedAnswer(BaseModel):
    """An answer to a question with citations to the retrieved documents/sources used to answer the question"""

    answer: str = Field(
        ..., description="The answer to the question, in the specified markdown format along with numbered citations"
    )
    citations: list[Citation] = Field(
        ...,
        description="List of citations to sources and references used to answer the question",
    )


def format_examples(qna_pairs: list[dict[str, Any]]) -> str:
    examples = ""
    for qna in qna_pairs:
        examples += f"<example>\n<question>{qna['question']}</question>\n<answer>{qna['metadata']['answer']}</answer>\n</example>\n\n"
    return f"<examples>\n{examples}\n</examples>"


def load_instruction(filepath: pathlib.Path) -> str:
    """Load instructions from a markdown file.

    Args:
        filepath: Full path to the instruction file or just the filename.
                 If only filename is provided, it looks in the prompts directory.
    """

    if not filepath.exists():
        raise FileNotFoundError(f"Instruction file not found: {filepath}")

    return filepath.read_text()


class ExpertSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", env_file_encoding="utf-8", extra="ignore")
    chat_model: str = Field(default="gpt-4.1-mini", description="The chat model to use")
    chat_model_api_key: str = Field(..., description="The API key to use for the chat model", env="CHAT_MODEL_API_KEY")


class Expert(BaseModel):
    settings: ExpertSettings = ExpertSettings()
    model_config = ConfigDict(arbitrary_types_allowed=True)
    prompt: pathlib.Path
    retriever: ChromaRetrieverWithReranker | None = None
    num_retrieval: int = 5

    @property
    def as_tool(self):
        docstring = inspect.getdoc(self.__call__)
        description = docstring.split("Args:\n")[0] if docstring else ""
        description = " ".join(description.strip().split())

        @function_tool(
            name_override=self.name,
            description_override=description,
        )
        async def tool_wrapper(wrapper: RunContextWrapper[DocsContext], question: ExpertQuestion):
            return await self.__call__(wrapper, question)

        return tool_wrapper


class WandbExpert(Expert):
    name: str = "wandb_expert"
    qna_retriever: ChromaRetrieverWithReranker
    blogs_and_guides_retriever: ChromaRetrieverWithReranker

    async def __call__(self, wrapper: RunContextWrapper[DocsContext], question: ExpertQuestion) -> CitedAnswer:
        """An expert agent to answer questions about Weights & Biases (wandb) and wandb sdk.
        Ask the agent a detailed natural language question like you would a human expert.
        Args:
            wrapper(RunContextWrapper[DocsContext]): Context wrapper containing session information
            question(ExpertQuestion): A detailed natural language question to address a user request
        Returns:
            A string answer to the question that can be directly displayed to a user
        """
        context = DocsContext.get_context(wrapper)

        # We can use context data if needed
        session_id = context.session_id
        # Log the query if needed using context data
        if context.metadata.get("should_log", True):
            # You could log to your db_client here
            pass

        qna_results = await self.qna_retriever.query(question.question, self.num_retrieval)
        queries = [question.question] + [result["metadata"]["answer"] for result in qna_results]
        docs_results, bng_results = await asyncio.gather(
            self.retriever.query(queries, self.num_retrieval),
            self.blogs_and_guides_retriever.query(queries, self.num_retrieval),
        )

        results = await self.retriever.rerank(question.question, docs_results + bng_results, k=self.num_retrieval)

        formatted_retrieval = format_retrieval(results, text_field=self.retriever.settings.text_field)

        instructions = load_instruction(self.prompt)

        # Add context-aware information to instructions if needed
        context_instructions = (
            f"\nSession ID: {session_id}" if context.metadata.get("include_session_in_prompt", False) else ""
        )

        wandb_agent = Agent[DocsContext](
            name=self.name,
            instructions=f"""{instructions}\n---\n{format_examples(qna_results)}\n---\n{formatted_retrieval}\n{context_instructions}""",
            model=LitellmModel(model=self.settings.chat_model, api_key=self.settings.chat_model_api_key),
            output_type=CitedAnswer,
        )

        result = await Runner.run(
            wandb_agent, question.question, context=context  # Pass the same context to maintain state
        )
        return result.final_output


class WeaveExpert(Expert):
    name: str = "weave_expert"  # Fixed the name
    qna_retriever: ChromaRetrieverWithReranker
    blogs_and_guides_retriever: ChromaRetrieverWithReranker

    async def __call__(self, wrapper: RunContextWrapper[DocsContext], question: ExpertQuestion) -> str:
        """An expert agent to answer questions about Weave.
        Ask the agent a detailed natural language question like you would a human expert.
        Args:
            wrapper(RunContextWrapper[DocsContext]): Context wrapper containing session information
            question(ExpertQuestion): A detailed natural language question to address a user request
        Returns:
            A string answer to the question that can be directly displayed to a user
        """
        context = DocsContext.get_context(wrapper)

        # Access context data as needed
        session_id = context.session_id

        qna_results = await self.qna_retriever.query(question.question, self.num_retrieval)
        queries = (
            [question.question]
            + [result[self.qna_retriever.settings.text_field] for result in qna_results]
            + [result["metadata"]["answer"] for result in qna_results]
        )
        docs_results, bng_results = await asyncio.gather(
            self.retriever.query(queries, self.num_retrieval),
            self.blogs_and_guides_retriever.query(queries, self.num_retrieval),
        )

        results = await self.retriever.rerank(question.question, docs_results + bng_results, k=self.num_retrieval)

        formatted_retrieval = format_retrieval(results, text_field=self.retriever.settings.text_field)

        instructions = load_instruction(self.prompt)

        # Add any context-specific instructions
        context_info = ""
        if context.metadata.get("user_history"):
            context_info = f"\nUser has previously asked about: {', '.join(context.metadata['user_history'])}"

        weave_agent = Agent[DocsContext](
            name=self.name,
            instructions=f"""{instructions}\n---\n{format_examples(qna_results)}\n---\n{formatted_retrieval}{context_info}""",
            model=LitellmModel(model=self.settings.chat_model, api_key=self.settings.chat_model_api_key),
            output_type=CitedAnswer,
        )

        result = await Runner.run(weave_agent, question.question, context=context)
        return result.final_output


class CoreweaveExpert(Expert):
    name: str = "coreweave_expert"

    async def __call__(self, wrapper: RunContextWrapper[DocsContext], question: ExpertQuestion) -> str:
        """An expert agent to answer questions about CoreWeave.
        Ask the agent a detailed natural language question like you would a human expert.
        Args:
            wrapper(RunContextWrapper[DocsContext]): Context wrapper containing session information
            question(ExpertQuestion): A detailed natural language question to address a user request
        Returns:
            A string answer to the question that can be directly displayed to a user
        """
        context = DocsContext.get_context(wrapper)

        # Potentially use context to customize search or log activity
        results = await self.retriever.query(question.question, k=self.num_retrieval)
        formatted_retrieval = format_retrieval(results, text_field=self.retriever.settings.text_field)

        instructions = load_instruction(self.prompt)

        # Could use context.metadata for personalization
        user_preferences = ""
        if context.metadata.get("coreweave_preferences"):
            user_preferences = f"\nUser preferences: {context.metadata['coreweave_preferences']}"

        coreweave_agent = Agent[DocsContext](
            name=self.name,
            instructions=f"""{instructions}\n---\n{formatted_retrieval}{user_preferences}""",
            model=LitellmModel(model=self.settings.chat_model, api_key=self.settings.chat_model_api_key),
            output_type=CitedAnswer,
        )

        result = await Runner.run(coreweave_agent, question.question, context=context)
        return result.final_output


class Docsbot(Expert):
    name: str = "docsbot"
    wandb_expert: WandbExpert
    weave_expert: WeaveExpert
    coreweave_expert: CoreweaveExpert
    runner_config: EarlyExitRunConfig = EarlyExitRunConfig()

    @weave.op
    async def __call__(self, inputs: list[TResponseInputItem], session_id: str = None, user_info: dict = None):
        # Create a DocsContext object for this session
        context = DocsContext(
            session_id=session_id or str(uuid.uuid4()),
            user_info=user_info or {},
            conversation=inputs,
            metadata={
                "should_log": True,
                "timestamp": datetime.datetime.now().isoformat(),
            },
        )

        # Process conversation history to extract useful information for context
        if inputs:
            # Extract potential topics from conversation to add as metadata
            user_messages = [msg.content for msg in inputs if hasattr(msg, "role") and msg.role == "user"]
            if user_messages:
                context.metadata["user_history"] = user_messages

        agent = Agent[DocsContext](
            name=self.name,
            instructions=load_instruction(self.prompt),
            model=LitellmModel(model=self.settings.chat_model, api_key=self.settings.chat_model_api_key),
            tools=[
                self.wandb_expert.as_tool,
                self.weave_expert.as_tool,
                self.coreweave_expert.as_tool,
            ],
            output_type=CitedAnswer,
        )
        result = await EarlyExitRunner.run(agent, inputs, context=context, run_config=self.runner_config)
        return result


Docsbot.model_rebuild()


class ChatRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    request: EasyInputMessageParam
    conversation: list[TResponseInputItem] = Field(default_factory=list)
    session_id: str = None
    user_info: dict = Field(default_factory=dict)


class ChatResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    response: EasyInputMessageParam
    conversation: list[TResponseInputItem]


class ChatApp:
    qna_retriever: ChromaRetrieverWithReranker = ChromaRetrieverWithReranker(
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

    blogs_and_guides_retriever: ChromaRetrieverWithReranker = ChromaRetrieverWithReranker(
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

    wandb_retriever: ChromaRetrieverWithReranker = ChromaRetrieverWithReranker(
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

    weave_retriever: ChromaRetrieverWithReranker = ChromaRetrieverWithReranker(
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
        rerank=Rerank(
            settings=RerankSettings(
                id_field="id",
                text_field="content",
            )
        ),
    )

    coreweave_retriever: ChromaRetrieverWithReranker = ChromaRetrieverWithReranker(
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
        rerank=Rerank(
            settings=RerankSettings(
                id_field="id",
                text_field="content",
            )
        ),
    )
    docsbot: Docsbot | None = None

    def __init__(self):
        self.docsbot = Docsbot(
            prompt=pathlib.Path(__file__).parent / "prompts/docsbot.md",
            wandb_expert=WandbExpert(
                prompt=pathlib.Path(__file__).parent / "prompts/wandb_expert.md",
                retriever=self.wandb_retriever,
                qna_retriever=self.qna_retriever,
                blogs_and_guides_retriever=self.blogs_and_guides_retriever,
            ),
            weave_expert=WeaveExpert(
                prompt=pathlib.Path(__file__).parent / "prompts/weave_expert.md",
                retriever=self.weave_retriever,
                qna_retriever=self.qna_retriever,
                blogs_and_guides_retriever=self.blogs_and_guides_retriever,
            ),
            coreweave_expert=CoreweaveExpert(
                prompt=pathlib.Path(__file__).parent / "prompts/coreweave_expert.md",
                retriever=self.coreweave_retriever,
            ),
        )

    @weave.op
    async def __call__(self, user_input: ChatRequest) -> ChatResponse:
        use_input_dict = user_input.model_dump(mode="json")
        current_input = use_input_dict["conversation"] + [use_input_dict["request"]]

        result = await self.docsbot(
            inputs=current_input,
            session_id=user_input.session_id,
            user_info=user_input.user_info,
        )

        assistant_content = result.final_output
        assistant_response = EasyInputMessageParam(role="assistant", content=assistant_content.model_dump_json())
        return ChatResponse(response=assistant_response, conversation=result.to_input_list())
