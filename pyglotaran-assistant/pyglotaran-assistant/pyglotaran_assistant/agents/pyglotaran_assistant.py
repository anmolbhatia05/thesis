from autogen.agentchat.contrib.society_of_mind_agent import SocietyOfMindAgent

from pyglotaran_assistant.agents.agent_group import AgentGroup
from pyglotaran_assistant.agents.utils import get_llm_config, termination_msg

class PyglotaranAssistant:
    def __init__(self) -> None:
        _group = AgentGroup()
        self.agent = SocietyOfMindAgent(
            "Pyglotaran-assistant",
            chat_manager=_group.setup(),
            llm_config=get_llm_config(),
            is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
            response_preparer="""
            Output a standalone response to the original request, without mentioning any of the intermediate discussion.

            This is the structure of the conversation you will receive :-
            * Previous conversation between the Human and the AI is before "Human: " This is usually not relevant here and is mostly noise.
            * The Human's query THIS time is after "Human:" This is important!
            * DISCUSSION == Followed by the conversation between the agents that work for the AI. This is important!

            So, your main focus should be on the QUESTION and the DISCUSSION. You standalone response should be from the DISCUSSION and 
            should either answer or further clarify the QUESTION. 
            """
        )
