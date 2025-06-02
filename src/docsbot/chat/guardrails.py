import os
from typing import Literal

from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
)
from agents.extensions.models.litellm_model import LitellmModel
from pydantic import BaseModel, Field


class ScopeCheckOutput(BaseModel):
    """Scope of the user's query."""

    domain: Literal["wandb", "weave", "coreweave", "meta"] | None = Field(
        ...,
        description="The domain of the user's query if it is in scope, otherwise None.",
    )
    reasoning: str = Field(..., description="The reasoning for the scope check.")
    is_in_scope: bool = Field(
        ...,
        description="Whether the user's query is a support-related question about one of the domains.",
    )


guardrail_agent = Agent(
    name="Scope Validation",
    instructions="""You’ll receive either a raw string (the user’s latest message) or a list of chat messages (each with `role` and `content`).  

**Steps**  
1. Identify the user’s *current intent*:
   - If input is a list, pick the **last** item where `role == "user"`.  
   - If input is a string, that is the intent.  
2. Determine if that intent is:
   - **wandb** — support for Weights & Biases or the wandb SDK (including integrations like transformers callback, wandb.log, etc.)  
   - **weave** — support for the Weave framework or its APIs (weave.compile, weave.widget, etc.)  
   - **coreweave** — support for CoreWeave’s infrastructure, services, APIs, deployment, pricing, etc.  
   - **meta** — conversational chatter (greetings, confirmations, farewells, small talk)  
   - Otherwise, it’s out-of-scope.  
3. Fill in your Pydantic model fields:
   - `domain`: set to exactly `"wandb"`, `"weave"`, `"coreweave"`, or `"meta"` if in scope; otherwise `null`.  
   - `reasoning`: a brief note explaining which keywords or context you used to classify.  
   - `is_in_scope`: `True` if `domain` is one of the four above; `False` for out-of-scope.

**Note**  
- Do **not** set the `is_in_scope` to `False` if `domain` is `null` (support request without specified product) during follow-up requests. - for example, if the user says "I'm having trouble with my account" or "I'm having trouble with my API key", "can you help me debug an issue?" you should set `is_in_scope` to `True`.

""".strip(),
    output_type=ScopeCheckOutput,
    model=LitellmModel(model="gpt-4.1-mini", api_key=os.environ["OPENAI_API_KEY"]),
)


@input_guardrail
async def scope_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, user_input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    # Run the small guardrail agent to classify scope
    if isinstance(user_input, list):
        inputs = []
        for item in user_input:
            if "role" in item and item["role"] in ["user", "assistant"]:
                inputs.append(item)
    else:
        inputs = user_input
    print(inputs)
    result = await Runner.run(guardrail_agent, inputs, context=ctx.context)

    # Extract the strongly-typed output
    output: ScopeCheckOutput = result.final_output

    return GuardrailFunctionOutput(
        output_info=output,
        tripwire_triggered=not output.is_in_scope,
    )
