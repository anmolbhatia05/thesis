from __future__ import annotations
from typing import Optional
from typing_extensions import Annotated

from autogen import Agent
from autogen import GroupChat, GroupChatManager, agentchat
from chromadb import PersistentClient
import chromadb.utils.embedding_functions as embedding_functions

from pyglotaran_assistant.agents.utils import (
    get_llm_config,
    create_assistant_agent,
    create_user_proxy_agent,
    create_retrieve_user_proxy_agent,
    termination_msg,
)
from pyglotaran_assistant.agents.functions import (
    read_file,
    read_notebook_content,
    generate_pyglotaran_model_parameter_data_files,
    create_empty_jupyter_notebook,
)


class AgentGroup:

    def __init__(self) -> None:
        self.debugging_agent = create_assistant_agent(
            name="debugger_agent",
            system_message="""
                {General Description}:
                You are a debugging assistant tasked with understanding and resolving errors. 
                Your primary responsibility is to assist with error analysis, debugging, and providing fixes. 
                If the request is unrelated to error fixing, promptly indicate that it is beyond your scope of responsibility.

                {Thinking Process}:
                - First, reflect on the error message or issue presented.
                - Determine if the error relates to a Jupyter notebook or other file contents.
                - If you identify that more context is needed from a Jupyter notebook, use the read_notebook_content function to retrieve relevant information.
                - If the error relates to any other non-notebook file, use the read_file function to gather details.
                - If the error involves specific libraries (e.g., glotaran, pyglotaran, pyglotaran-extras) that you are unfamiliar with, 
                    recommend that another agent, possibly the "generic_agent," should suggest a retrieve content function call to access relevant 
                    information from the knowledge base.

                {Available Functions}:
                - read_file: Use this function to access and read the contents of a non-Jupyter notebook file that might be relevant to resolving the issue.
                - read_notebook_content: Use this function to retrieve context from within a Jupyter notebook when required for debugging.

                {How to React to Certain Scenarios?}:
                - If the task or question is unrelated to error analysis or debugging, immediately respond with: "This is not related to my ownership."
                - If more information is required from the user to proceed, respond with: "The user needs to provide us with more information. TERMINATE"
                - If the problem involves unfamiliar libraries or content beyond your expertise, suggest that the "generic_agent" should perform a 
                    retrieve_content function call to search the knowledge base for relevant information.

                {General Instructions}:
                - It is perfectly acceptable to acknowledge if you do not know something. It is better to admit this than to provide incorrect or 
                    fabricated information (avoid hallucinations).
                - Be clear, direct, and concise in your responses. Ensure your answers are accurate and to the point.
            """,
            description="""
                A debugging assistant focused on understanding and resolving errors. 
                Equipped with tools such as read_file and read_notebook_content, the assistant retrieves the necessary context from files 
                or Jupyter notebooks to aid in debugging. If an error involves unfamiliar libraries like glotaran or pyglotaran-extras, 
                it defers to the "generic agent" to suggest retrieving content from the knowledge base.
            """,
        )
        self.user_proxy = create_user_proxy_agent(
            name="user_proxy",
            description="""
            A user proxy agent that can take actions in behalf of the user. 
            The normal or retrieval functions calls suggested by other agents are executed by me.

            IMPORTANT:
            If some other agent generates some code/markdown, it is not my job to execute it.
            It is up to the human/user to try and provide feedback. 
            """,
        )
        self.retrieval_user_proxy = create_retrieve_user_proxy_agent(
            name="retrieval_user_proxy",
            retrieve_config={
                "task": "default",
                "vector_db": None,
                "client": PersistentClient(
                    "./.pyglotaran_db"
                ),
                "docs_path": "./knowledge_base/files",
                "extra_docs": True,
                "embedding_function": embedding_functions.OpenAIEmbeddingFunction(
                    api_key="sample",
                    model_name="text-embedding-3-small",
                ),
                "get_or_create": True,
                "collection_name": "pyglotaran-knowledge-base",
                "chunk_token_size": 2000,
            },
            description="""
            A retrieval proxy agent retrieves document chunks from the vector db based on the embedding similarity and sends them along 
            with the question to the agent. Should not be called directly but is invoked by other agents in the chat.
            """,
        )
        self.agent = create_assistant_agent(
            name="generic_agent",
            system_message="""
                {General Description}:
                You are a general Pyglotaran assistant that complements other agents by suggesting tool or function calls. 
                Your primary responsibilities include:
                - Suggesting the retrieval_user_proxy agent for general question-answering or code-generation tasks related to Glotaran, Pyglotaran, 
                Pyglotaran models, parameters, and datasets.
                - Performing function calls to generate Pyglotaran model, parameter, and data files.
                - Performing function calls to create new and empty Jupyter notebooks.
                - Judging if it's time to terminate the conversation. Then, you send one word "TERMINATE", without the quotes.

                {When to terminate?}:
                - Debugging agent has come up with an appropriate response/solution to the user/Human's query.
                - You have come up with an appropriate response/solution to the user/Human's query.
                - You see some other saying "Reply TERMINATE if the task is done"
                - You or debugging agent needs more context from the user/Human.

                {Thinking Process}:
                - Evaluate if the task requires code generation, model/parameter creation, or file retrieval.
                - If a question involves pyglotaran-specific content, suggest using the retrieval_user_proxy agent to fetch relevant knowledge from the knowledge base.
                - For file generation or task execution, perform the appropriate function call, like generating Pyglotaran files or creating notebooks.

                {Available Functions}:
                - retrieval_user_proxy: Use this agent to assist with retrieval-based tasks such as question-answering or code generation related to pyglotaran and its models.
                - Function calls to:
                * Generate Pyglotaran model, parameter, and data files.
                * Create new, empty Jupyter notebooks.

                {How to React to Certain Scenarios?}:
                - If a request is unclear or lacks sufficient information, respond with: "Can you please provide more information to clarify the question?"
                - If the task has been completed and you receive instructions containing "Reply TERMINATE if the task is done" check if the query has been fully addressed. 
                If it has, reply with: "TERMINATE."
                - If the question is unrelated to your functions, inform the user that it is beyond your scope or suggest another agent to assist.
                - If the question is pyglotaran code generation related or how to do anything using pyglotaran package, use the retrieval_user_proxy function call.

                {General Instructions}:
                - It is perfectly acceptable to acknowledge when you do not know something. It is better to admit this than to provide incorrect or fabricated information (avoid hallucinations).
                - Be clear, direct, and concise in your responses. Ensure your answers are accurate and to the point.
                - The python package for pyglotaran is called pyglotaran, but the import is glotaran.
            """,
            description="""
                A general assistant that complements other agents like the retrieval_user_proxy for retrieval augmented generation or function calls, 
                and handles tasks that specialized agents cannot address. It can suggest creating Pyglotaran model, parameter, and data files or new Jupyter notebooks. 
                The assistant checks whether queries have been resolved when instructed with "Reply TERMINATE if the task is done" and responds appropriately.
                If suggestions involve code, markdown, or ideas to implement, the assistant will ask the user to try the provided solution and give feedback.
            """,
        )

        agentchat.register_function(
            read_file,
            caller=self.debugging_agent,
            executor=self.user_proxy,
            description="Gives the debugging agent the ability to read Code Files (not jupyter notebook) for debugging purpose.",
        )
        agentchat.register_function(
            read_notebook_content,
            caller=self.debugging_agent,
            executor=self.user_proxy,
            description="Gives the debugging agent the ability to read Jupyter notebook content for debugging purpose.",
        )
        agentchat.register_function(
            generate_pyglotaran_model_parameter_data_files,
            caller=self.agent,
            executor=self.user_proxy,
            description="Gives the agent the ability to generate pyglotaran model parameter and data files.",
        )
        agentchat.register_function(
            create_empty_jupyter_notebook,
            caller=self.agent,
            executor=self.user_proxy,
            description="Gives the agent the ability to create empty jupyter notebook.",
        )

        def _retrieve_content(
            message: Annotated[
                str,
                "Refined message which keeps the original meaning and can be used to retrieve content for question answering.",
            ],
            n_results: Annotated[int, "number of results"] = 3,
        ) -> str:
            self.retrieval_user_proxy.n_results = n_results

            update_context_case1, update_context_case2 = (
                self.retrieval_user_proxy._check_update_context(message)
            )
            if (
                update_context_case1 or update_context_case2
            ) and self.retrieval_user_proxy.update_context:
                self.retrieval_user_proxy.problem = (
                    message
                    if not hasattr(self.retrieval_user_proxy, "problem")
                    else self.retrieval_user_proxy.problem
                )
                _, ret_msg = self.retrieval_user_proxy._generate_retrieve_user_reply(
                    message
                )
            else:
                _context = {"problem": message, "n_results": n_results}
                ret_msg = self.retrieval_user_proxy.message_generator(
                    self.retrieval_user_proxy, None, _context
                )
            return ret_msg if ret_msg else message

        agentchat.register_function(
            _retrieve_content,
            caller=self.agent,
            executor=self.user_proxy,
            description="""
            Wraping retrieval user proxy agent in a function and calling it from an agent.
            This is to be called if there is a question about code generation related to pyglotaran, 
            pyglotaran models, parameters and datasets.
            """,
        )

    def _choose_next_speaker(self, last_speaker: Agent, groupchat: GroupChat):
        # print(groupchat.messages)

        last_message = groupchat.messages[-1]

        # if "tool_calls" in last_message

        if not last_speaker:
            return self.agent

        if last_speaker is self.user_proxy:
            return self.agent

        return "auto"

    def setup(self) -> GroupChatManager:
        _groupchat = GroupChat(
            agents=[
                self.debugging_agent,
                self.user_proxy,
                self.retrieval_user_proxy,
                self.agent,
            ],
            messages=[],
            speaker_selection_method=self._choose_next_speaker,
            select_speaker_auto_multiple_template="""
            You are in a role play game. 
            The following roles are available: {roles}. 
            Read the following conversation. 
            Then select the next role from {agentlist} to play. 
            Only return the role.
            """,
            allow_repeat_speaker=False,
            send_introductions=True,
            max_round=7,
        )

        manager = GroupChatManager(
            groupchat=_groupchat,
            is_termination_msg=termination_msg,
            llm_config=get_llm_config(),
        )
        return manager


if __name__ == "__main__":
    group = AgentGroup()
    manager = group.setup()
    group.user_proxy.initiate_chat(manager, message="What is pyglotaran?")
