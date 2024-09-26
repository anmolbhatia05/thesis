# pyglotaran_assistant

`pyglotaran_assistant` is a Jupyter AI module, a package
that registers additional model providers and slash commands for the Jupyter AI
extension.

### Development install

Create a separate python virtual environment. 

Update the `config.json` file with your openai api key. 
Here as well - `./pyglotaran-assistant/pyglotaran_assistant/agents/knowledge_base/setup.py`
Here as well - `./pyglotaran-assistant/pyglotaran_assistant/agents/agent_group.py`
(Todo, make sure everything uses a settings file or config.json)

Then, setup knowledge base for RAG - run `./pyglotaran-assistant/pyglotaran_assistant/agents/knowledge_base/setup.py`
(Based on this, you might have to change retrieval user proxy agent configs here - `./pyglotaran-assistant/pyglotaran_assistant/agents/agent_group.py`) 

You might want to install pyparamgui package in this python env, for experiment - 2 (thesis) function calls.

Then, 
```bash
cd pyglotaran-assistant
pip install -e "."
jupyter lab
```

Find the jupyter-ai chat extension to interact with the pyglotaran-assistant. 

