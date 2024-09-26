from typing import Dict, Type


from jupyter_ai.chat_handlers.base import BaseChatHandler, SlashCommandRoutingType
from jupyter_ai.models import HumanChatMessage, CellSelection, CellWithErrorSelection
from jupyter_ai_magics.providers import BaseProvider

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class TestSlashCommand(BaseChatHandler):
    """
    A test slash command implementation that developers should build from. The
    string used to invoke this command is set by the `slash_id` keyword argument
    in the `routing_type` attribute. The command is mainly implemented in the
    `process_message()` method. See built-in implementations under
    `jupyter_ai/handlers` for further reference.

    The provider is made available to Jupyter AI by the entry point declared in
    `pyproject.toml`. If this class or parent module is renamed, make sure the
    update the entry point there as well.
    """

    id = "test"
    name = "Test"
    help = "A test slash command."
    routing_type = SlashCommandRoutingType(slash_id="test")

    uses_llm = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def process_message(self, message: HumanChatMessage):
        self.reply("This is the `/test` slash command.")

OBSERVE_STRING_TEMPLATE = """
You are a pyglotaran assistant, a conversational assistant that lives in jupyter lab. 
The way to interact is through a chat window. You help your users by learning from the notebook cell input
and output. 

Additional instructions:

{extra_instructions}

Input cell:

```
{cell_content}
```

Output:

```
{output}
```
""".strip()

OBSERVE_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=[
        "extra_instructions",
        "cell_content",
        "output",
    ],
    template=OBSERVE_STRING_TEMPLATE,
)

class ObserveSlashCommand(BaseChatHandler):
    """
    A test slash command implementation that developers should build from. The
    string used to invoke this command is set by the `slash_id` keyword argument
    in the `routing_type` attribute. The command is mainly implemented in the
    `process_message()` method. See built-in implementations under
    `jupyter_ai/handlers` for further reference.

    The provider is made available to Jupyter AI by the entry point declared in
    `pyproject.toml`. If this class or parent module is renamed, make sure the
    update the entry point there as well.
    """

    id = "observe"
    name = "Observe"
    help = "A command to use to observe a particular cell input and output."
    routing_type = SlashCommandRoutingType(slash_id="observe")

    uses_llm = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_llm_chain(
        self, provider: Type[BaseProvider], provider_params: Dict[str, str]
    ):
        unified_parameters = {
            **provider_params,
            **(self.get_model_parameters(provider, provider_params)),
        }
        llm = provider(**unified_parameters)

        self.llm = llm
        self.llm_chain = LLMChain(llm=llm, prompt=OBSERVE_PROMPT_TEMPLATE, verbose=True)

    async def process_message(self, message: HumanChatMessage):
        print(message.dict())
        print(message.selection.dict())
        # if not (message.selection and message.selection.type == "cell"):
        #     self.reply(
        #         "`/observe` requires an active code cell without error output. Please click on a cell without error output and retry.",
        #         message,
        #     )
        #     return

        # hint type of selection
        selection: CellSelection = message.selection

        # parse additional instructions specified after `/observe`
        extra_instructions = message.prompt[8:].strip() or "None."

        self.get_llm_chain()
        with self.pending("Analyzing"):
            response = await self.llm_chain.apredict(
                extra_instructions=extra_instructions,
                stop=["\nHuman:"],
                cell_content=selection.source,
                output="temp"
            )
        self.reply(response, message)