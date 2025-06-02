from dataclasses import dataclass, field

from agents import RunConfig, Runner
from agents._run_impl import RunImpl, ToolsToFinalOutputResult


@dataclass
class EarlyExitRunConfig(RunConfig):
    """Extends RunConfig to add early exit on tool call capability"""

    auto_exit_on_tool: bool = True
    """Whether to automatically exit and return the result when a single expert tool is called"""

    early_exit_tool_list: list[str] = field(
        default_factory=lambda: ["wandb_expert", "weave_expert", "coreweave_expert"]
    )
    """List of tool names that should trigger early exit when only one is called"""

    _called_expert_tools: set = None


class EarlyExitRunner(Runner):
    @classmethod
    async def run(cls, starting_agent, input, **kwargs):
        # Use our custom config, preserving any passed-in config values
        if "run_config" in kwargs and kwargs["run_config"] is not None:
            if not isinstance(kwargs["run_config"], EarlyExitRunConfig):
                # Copy the original config's attributes to a new EarlyExitRunConfig
                original_config = kwargs["run_config"]
                custom_config = EarlyExitRunConfig()

                # Transfer all attributes from original to custom config
                for field in original_config.__dataclass_fields__:
                    if hasattr(original_config, field):
                        setattr(custom_config, field, getattr(original_config, field))

                kwargs["run_config"] = custom_config
            # If it's already an EarlyExitRunConfig, use it as is
        else:
            # Create a default config
            kwargs["run_config"] = EarlyExitRunConfig()

        # Initialize the set of called expert tools
        kwargs["run_config"]._called_expert_tools = set()

        # Store original methods
        original_check_method = RunImpl._check_for_final_output_from_tools
        original_process_response = RunImpl.process_model_response

        # Define a method to intercept and track tool calls
        def patched_process_response(*, agent, all_tools, response, output_schema, handoffs):
            result = original_process_response(
                agent=agent, all_tools=all_tools, response=response, output_schema=output_schema, handoffs=handoffs
            )

            # Track which expert tools were called
            if isinstance(kwargs["run_config"], EarlyExitRunConfig):
                exit_tools = kwargs["run_config"].early_exit_tool_list
                for tool_name in result.tools_used:
                    if tool_name in exit_tools:
                        kwargs["run_config"]._called_expert_tools.add(tool_name)

            return result

        # Define our patched method that uses the original
        async def patched_check_method(*, agent, tool_results, context_wrapper, config):
            # Check if we're using our custom config and have tool results
            if tool_results and isinstance(config, EarlyExitRunConfig) and config.auto_exit_on_tool:
                # Check if any result is from our configured exit tools
                exit_tools = config.early_exit_tool_list
                exit_tool_results = [tr for tr in tool_results if tr.tool.name in exit_tools]

                # Only exit early if EXACTLY ONE exit tool was called
                if len(exit_tool_results) == 1 and len(config._called_expert_tools) == 1:
                    # Return the tool result as the final output
                    return ToolsToFinalOutputResult(is_final_output=True, final_output=exit_tool_results[0].output)

            # Otherwise, use the original behavior by calling the stored reference
            return await original_check_method(
                agent=agent, tool_results=tool_results, context_wrapper=context_wrapper, config=config
            )

        # Apply the patches
        RunImpl._check_for_final_output_from_tools = patched_check_method
        RunImpl.process_model_response = patched_process_response

        try:
            # Run with our modified implementation
            result = await super().run(starting_agent, input, **kwargs)
            return result
        finally:
            # Always restore the original implementations
            RunImpl._check_for_final_output_from_tools = original_check_method
            RunImpl.process_model_response = original_process_response
