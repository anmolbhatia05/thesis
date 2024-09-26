from pathlib import Path
import hashlib

from chromadb import PersistentClient
import chromadb.utils.embedding_functions as embedding_functions

class KnowledgeBase:
    def __init__(self) -> None:
        self.path_to_db: Path = Path.cwd() / ".pyglotaran_db"
        self.path_to_db.mkdir(parents=True, exist_ok=True)
        self.chroma_client: PersistentClient = PersistentClient(str(self.path_to_db))
        if not self._is_client_connected():
            raise Exception("Client can't connect to the db for some reason!")
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key="sample",
                model_name="text-embedding-3-small"
            )
        self.collection_name: str = "pyglotaran-knowledge-base"
        self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name, embedding_function=self.openai_ef)
        self.path_to_knowledge: Path = Path(__file__).resolve().parent / "files"

    def _is_client_connected(self) -> bool:
        try:
            self.chroma_client.heartbeat()
            return True
        except Exception:
            return False
        
    def upsert_into_knowledge_base(self):
        file_paths = list(self.path_to_knowledge.glob('*.txt'))
        
        ids = []
        metadatas = []
        documents = []

        if not self._is_client_connected():
            raise Exception("Client can't connect to the db for some reason!")
        
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append(content)

            metadata = {"file_name": file_path.stem}
            metadatas.append(metadata)

            hasher = hashlib.sha256()
            hasher.update(file_path.name.encode('utf-8'))
            file_id = hasher.hexdigest()
            ids.append(file_id)

        self.collection.upsert(
            ids=ids,
            metadatas=metadatas,
            documents=documents
        )

if __name__ == '__main__':
    kb = KnowledgeBase()
    kb.upsert_into_knowledge_base()
    print(kb.collection.count())