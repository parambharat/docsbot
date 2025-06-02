import asyncio
import html
import json
import re

import instructor
import litellm
import pandas as pd
from litellm import acompletion
from litellm.caching.caching import Cache
from pydantic import BaseModel, Field
from tqdm import tqdm, trange

litellm.cache = Cache(disk_cache_dir="data/cache/litellm")


class Evaluation(BaseModel):
    """evaluation of a response based on the provided context and question."""

    judgement_reason: str = Field(
        ..., description="Brief and concise justification for the judgment"
    )
    score: int = Field(
        ...,
        description="integer score with the following meanings: 1 - Incorrect, 2 - Partially Correct, 3 - Fully Correct",
    )
    relevant_docs: list[str] = Field(
        [],
        description="List of relevant documents supporting the judgment",
        min_length=0,
        max_length=10,
    )


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


def format_retrieval(docs: list[dict]) -> str:
    """
    Format a list of document dictionaries into a string representation.
    """
    formatted_docs = ""
    for idx, doc in enumerate(docs):
        formatted_docs += f"\n<doc_{idx}>"
        formatted_docs += format_doc(doc)
        formatted_docs += f"</doc_{idx}>\n"

    xml_string = f"<retrieval>{formatted_docs}</retrieval>"

    # Escape special characters in the XML string
    # xml_string = prettify_xml(xml_string)
    # normalize the 2 or more newlines to single newline
    xml_string = re.sub(r"\n{2,}", "\n", xml_string)
    return html.unescape(xml_string)


def format_record(record):
    context = format_retrieval(record["retrieval"])

    question = f"<question>\n{record['question']}\n</question>\n"
    answer = f"<answer>\n{record['answer']}\n</answer>\n"
    formatted_record = question + context + answer
    return formatted_record


async def judge_record(record):
    EVAL_PROMPT = """You are an evaluator tasked with assessing the quality of responses provided by a technical AI assistant based on the provided context documents and user questions. Follow the detailed step-by-step instructions below to consistently evaluate and clearly justify your assessments.

**Step-by-Step Evaluation Process**

1. **Review the Question, Context, and Answer**:

   * Carefully read the user question.
   * Thoroughly examine the retrieved contextual documents.
   * Closely review the AI assistant's response.

2. **Conduct a Relevance Check**:

   * Verify whether the AI response directly addresses the user's question.
   * Confirm that the response is based explicitly on the retrieved context.

3. **Assess Correctness**:

   * Use the predefined correctness scale (see below) to judge the response.
   * Clearly articulate why the response was judged as such, providing specific details from the context.

4. **Identify Relevant Documents**:

   * Clearly indicate which documents from the provided context support or relate to the AI response.

5. **Document Judgment and Justification**:

   * Provide a concise, clear rationale for your judgment.

6. **Prepare JSON Output**:

   * Follow the predefined JSON format precisely for detailing your evaluation.

---

**Evaluation Criteria**

Evaluate based on:

* **Helpfulness**: Does the answer assist the user?
* **Relevance**: Is the answer relevant to the user's question and context?
* **Accuracy**: Is the answer factually correct?
* **Depth**: Does it demonstrate in-depth understanding?
* **Level of Detail**: Is the level of detail appropriate?

---

 **Correctness Assessment Scale**

* **1 – Incorrect**: The answer contains critical factual errors or irrelevant information.
* **2 – Partially Correct**: The answer includes correct information but has notable omissions, inaccuracies, or ambiguity.
* **3 – Fully Correct**: The answer is accurate, complete, and well-aligned with the provided context.

---

**Judgment Justification**

Justify clearly, referencing specific context information:

Examples:

* "The assistant accurately cited information from Doc\_1."
* "The answer omits critical details clearly mentioned in Doc\_2."
* "The explanation was vague, despite detailed information available in the context."

---

**JSON Output Structure**

Use the following JSON format:

```json
{
  "judgement_reason": "<Brief and concise justification for your judgment>",
  "score": <1 | 2 | 3>,
  "relevant_docs": ["<doc_0>", "<doc_1>", "..."]
}
```

---

**Example Evaluation**

* **Question**: "What is the function of mitochondria?"
* **Context**: \["Doc\_1: Mitochondria are the powerhouse of the cell, generating ATP...", "Doc\_2: Overview of organelles"]
* **AI Response**: "Mitochondria generate energy in the form of ATP, making them vital for cellular respiration."

```json
{
  "judgement_reason": "The response accurately identifies the function of mitochondria, aligning factually with Doc_1.",
  "score": 3,
  "relevant_docs": ["<doc_1>"]
}
```"""
    formatted_record = format_record(record)

    client = instructor.from_litellm(acompletion, mode=instructor.mode.Mode.JSON)
    try:
        response = await client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": EVAL_PROMPT},
                {"role": "user", "content": formatted_record},
            ],
            temperature=0.5,
            max_tokens=1024,
            response_model=Evaluation,
            caching=True,
            max_retries=5,
        )
    except Exception as e:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": EVAL_PROMPT},
                {"role": "user", "content": formatted_record},
            ],
            temperature=0.5,
            max_tokens=1024,
            response_model=Evaluation,
            caching=True,
            max_retries=5,
        )
    record["judgement"] = response.model_dump()
    return record


def format_record_with_judgement(record):
    judgement_map = {1: "Incorrect", 2: "Partially Correct", 3: "Fully Correct"}
    formatted_record = format_record(record)
    judgement = judgement_map.get(record["judgement"]["score"], "Incorrect")
    formatted_record += f"\n<judgement>\n{judgement}\n</judgement>\n"
    formatted_record += f"\n<judgement_reason>\n{record['judgement']['judgement_reason']}\n</judgement_reason>\n"
    formatted_record += f"\n<relevant_docs>\n{','.join(record['judgement']['relevant_docs'])}\n</relevant_docs>\n"
    return formatted_record


async def correction(record):
    CORRECTION_PROMPT = """You are tasked with generating a corrected, comprehensive, and contextually accurate answer based on the provided information:

* **Original Question**: Clearly understand the user's initial inquiry.
* **Retrieved Context**: Review the provided contextual information thoroughly.
* **AI-generated Answer**: Assess the initial answer provided by the AI.
* **Judgement and Judgement Reason**: Understand why the provided AI-generated answer was marked as incorrect or partially correct. **However, prioritize accuracy and validation through retrieved context over the judgement reason, as the judgement itself may contain inaccuracies.**
* **Relevant Documents**: Primarily consider the indicated relevant documents, but also refer to the broader retrieved context if necessary for additional clarity or corrections.

**Step-by-Step Response Generation**

1. **Review Provided Information**:

   * Carefully read and understand the original user question.
   * Examine the retrieved contextual excerpts.
   * Understand the AI-generated answer and provided judgment. Validate judgment reasons against the retrieved context.

2. **Identify Necessary Corrections**:

   * Independently verify specific inaccuracies, omissions, or irrelevant information mentioned in the judgment reason by directly referencing the retrieved documents.

3. **Generate the Corrected Answer**:

   * Provide a comprehensive and accurate response clearly addressing the original user question.
   * Integrate specific details, facts, or examples explicitly found in the indicated relevant documents and broader context as necessary.

4. **Validate Against Context**:

   * Cross-check the final answer with relevant documents and additional retrieved context to ensure accuracy and completeness.
   * If discrepancies are identified between the judgment reason and context, explicitly trust and rely on the context.

---

**Final Answer Format**

Provide your corrected and comprehensive answer clearly and concisely.

---

**Example**

* **Original Question**:

  ```
  How do I get started with Weave?
  ```

* **AI-generated Answer**:

  ```
  To get started with Weave, install it with 'pip install weave --upgrade', load your data into Pandas, and use 'weave.show' for visualizations.
  ```

* **Judgement**: Partially Correct

* **Judgement Reason**:

  ```
  The response incorrectly mentions 'pip install weave --upgrade' instead of the correct command 'pip install streamlit pandas plotly weave' and makes assumptions about specific data types and uses without direct references to context documents.
  ```

* **Relevant Documents**: \[doc\_1, doc\_2, doc\_3, doc\_4]

* **Corrected Answer**:

To get started with Weave, first install the toolkit using:

```bash
  pip install weave --upgrade
````

Next, initialize Weave in your Python script:

```python
import weave
weave.init("your_project_name")

@weave.op()
def example_function(input_data):
    # Your function logic here
    output = process(input_data)
    return output
```

By decorating your functions with `@weave.op()`, Weave automatically traces inputs and outputs, enabling detailed monitoring and evaluation of your LLM applications.
"""
    formatted_record = format_record_with_judgement(record)
    client = instructor.from_litellm(acompletion)
    try:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": CORRECTION_PROMPT},
                {"role": "user", "content": formatted_record},
            ],
            temperature=0,
            max_tokens=4096,
            response_model=str,
        )
    except Exception as e:
        response = await client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": CORRECTION_PROMPT},
                {"role": "user", "content": formatted_record},
            ],
            temperature=0,
            max_tokens=4096,
            response_model=str,
        )
    record["answer"] = response
    record = await judge_record(record)
    return record


def gen_batches(file_name: str, batch_size: int):
    """
    Generator function to yield batches of records from a JSONL file.
    """
    with open(file_name) as f:
        batch = []
        for line in tqdm(f):
            record = json.loads(line)
            batch.append(record)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def gen_correction_batches(file_name: str, batch_size: int):
    """
    Generator function to yield batches of records from a JSONL file.
    """
    with open(file_name) as f:
        batch = []
        for line in tqdm(f):
            record = json.loads(line)
            if record["judgement"]["score"] < 3:
                batch.append(record)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


class Verification(BaseModel):
    """verification of the answer based on the provided context and question."""

    annotated_answer: str = Field(
        ..., description="Annotated answer with numeric citations"
    )
    factual_inconsistencies: list[str] = Field(
        ...,
        description="List of factually inconsistent or unverifiable statements from the answer",
    )
    factually_consistent: bool = Field(
        ...,
        description="Whether the answer is factually consistent with the provided context",
    )


async def verify_facts(record):
    VERIFICATION_PROMPT = """You are a careful and thorough **Fact-Checking Annotator** tasked with verifying and annotating provided answers for factual correctness and consistency.

### Goal:

Your primary goal is to ensure the provided answer is entirely factually consistent with the retrieved context and clearly annotated to indicate sources and highlight any factual inconsistencies or unverifiable claims.

### Task:

You will receive a question (`<question>`), retrieved context (`<retrieval>`), and an answer (`<answer>`). You must carefully read through the provided answer and:

* **Verify** every fact, statement, code block, method, or reference in the answer against the retrieved context.
* **Annotate** each fact, statement, code block, reference, or any other information in the answer that directly corresponds to information in the retrieved context with numeric citations like `[0]`, `[1]`, etc., corresponding to the document IDs.
* **Identify and list any unverifiable or potentially incorrect statements** clearly in the output key `factual_inconsistencies`.

### Guidelines:

1. **Do not alter the provided answer's content**, except to add numeric citations.
2. **Ensure markdown formatting in the provided answer is preserved or enhanced** during annotation, but **do not change the actual content**.
3. Remove any existing sections like summaries, references, or sources from the provided answer, since annotations will directly indicate sources.
4. For inaccuracies identified, clearly list these questionable statements under the output key `factual_inconsistencies`.

### Input Format:

```
<question>
Question text here
</question>

<retrieval>
<doc_0><source>URL</source><excerpt>Document excerpt text</excerpt></doc_0>
<doc_1><source>URL</source><excerpt>Document excerpt text</excerpt></doc_1>
...
</retrieval>

<answer>
Answer text to be annotated here
</answer>
```

### Output Format:

Your final output must be in the following JSON format:

```json
{
  "annotated_answer" : "Annotated answer with numeric citations",
  "factual_inconsistencies": ["List of factually inconsistent or unverifiable statements from the answer"],
  "factually_consistent": true or false
}
```

Provide your JSON-formatted output clearly following these instructions.
"""
    formatted_record = format_record(record)
    client = instructor.from_litellm(acompletion)
    try:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": VERIFICATION_PROMPT},
                {"role": "user", "content": formatted_record},
            ],
            response_model=Verification,
            temperature=0,
            max_tokens=4096,
        )
    except Exception as e:
        response = await client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": VERIFICATION_PROMPT},
                {"role": "user", "content": formatted_record},
            ],
            response_model=Verification,
            temperature=0,
            max_tokens=4096,
        )
    record["annotated_answer"] = response.model_dump()
    return record


def gen_verification_batches(records: list[dict], batch_size: int):

    for i in trange(0, len(records), batch_size):
        yield records[i : i + batch_size]


async def main():
    # with open("data/qna_judgements.jsonl", "w") as out_file_handle:
    #     for batch in gen_batches("data/qna_with_retrieval.jsonl", batch_size=50):
    #         tasks = []
    #         for record in batch:
    #             tasks.append(judge_record(record))
    #         responses = await asyncio.gather(*tasks)
    #         for response in responses:
    #             out_file_handle.write(json.dumps(response) + "\n")
    df = pd.read_json(
        "data/qna_judgements_final.jsonl", orient="records", lines=True
    )
    df = df.iloc[df.index.difference([415, 1552, 3208])]
    df = df[df["judgement"].map(lambda x: x["score"] == 3)]
    df = df.reset_index(drop=True)
    subset_df = df.loc[:, ["idx", "question", "answer", "retrieval", "judgement"]]

    def extract_retrieval(row):
        retrieval = row["retrieval"]
        relevant_docs = row["judgement"]["relevant_docs"]
        doc_ids = [int(doc.split("_")[-1].strip(">")) for doc in relevant_docs]
        return [retrieval[i] for i in doc_ids]

    subset_df.loc[:, "context"] = subset_df.apply(extract_retrieval, axis=1)
    subset_df = subset_df[subset_df.context.map(len) != 0]
    subset_df = subset_df.reset_index(drop=True)
    subset_records = subset_df.to_dict(orient="records")

    with open(
        "data/qna_judgements_with_verification.jsonl", "w+"
    ) as out_file_handle:
        for batch in gen_verification_batches(subset_records, batch_size=50):
            tasks = []
            for record in batch:
                tasks.append(verify_facts(record))
            responses = await asyncio.gather(*tasks)
            for response in responses:
                out_file_handle.write(json.dumps(response) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
