from typing import Any, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM



from .agents.pyglotaran_assistant import PyglotaranAssistant

class TestLLM(LLM):
    model_id: str

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        a = PyglotaranAssistant()
        return a.agent.generate_inner_monologue_reply(messages=[{"role":"user", "content":prompt}])[-1]

        
    
