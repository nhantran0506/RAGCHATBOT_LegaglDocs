from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import requests
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import warnings
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings('ignore')

class RAG:
    def __init__(self):
        self.vector_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.pinecone = Pinecone(api_key=os.getenv('API_KEY'))
        self.index_name = "my-vector-db"

        if self.index_name not in self.pinecone.list_indexes().names():
            self.pinecone.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        self.index = self.pinecone.Index(self.index_name)
        self.context = []

    def read_html(self, path):
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
        soup = BeautifulSoup(content, features="html.parser")
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)
        return text, os.path.basename(path)

    def text_to_docs(self, text, filename):
        if isinstance(text, str):
            text = [text]
        page_docs = [Document(page_content=page) for page in text]
        for i, doc in enumerate(page_docs):
            doc.metadata["page"] = i + 1

        doc_chunks = []
        for doc in page_docs:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=4000,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                chunk_overlap=0,
            )
            chunks = text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={"page": doc.metadata["page"], "chunk": i},
                )
                doc.metadata["source"] = (
                    f"{doc.metadata['page']}-{doc.metadata['chunk']}"
                )
                doc.metadata["filename"] = filename
                doc_chunks.append(doc)
        return doc_chunks

    def docs_to_index(self, docs):
        for doc in docs:
            embeddings = self.vector_model.encode([doc.page_content])
            metadata = {
                "page": doc.metadata["page"],
                "chunk": doc.metadata["chunk"],
                "filename": doc.metadata["filename"],
            }
            self.index.upsert(
                vectors=[(doc.metadata["source"], embeddings[0].tolist(), metadata)]
            )

    def create_vectordb(self, paths):
        documents = []
        for path in paths:
            text, filename = self.read_html(path)
            documents.extend(self.text_to_docs(text, filename))
        self.docs_to_index(documents)

    def retrieve_relevant_docs(self, query, top_k=5, threshold=0.8):
        query_embedding = self.vector_model.encode([query])[0].tolist()
        results = self.index.query(
            vector=query_embedding, top_k=top_k, include_metadata=True
        )
        filtered_results = [
            match for match in results["matches"] if match["score"] > threshold
        ]
        return filtered_results

    def generate_response(self, query):
        relevant_docs = self.retrieve_relevant_docs(query)
        context = " ".join(
            [
                match["metadata"]["text"]
                for match in relevant_docs
                if "text" in match["metadata"]
            ]
        )
        input_text = f"""
            Bạn là một người cực kì am hiểu luật có suy luận logic xuất sắc, bạn sẽ được hỏi một số câu hỏi về luật của Việt Nam.
            Câu trả lời nên ngắn gọn, súc tích và dễ hiểu.
            Thông tin: {context} 
            Người dùng: {query}
        """

        payload = {
            "model": "llama3.1",
            "prompt": input_text,
            "context": self.context,
            "stream": False,
        }

        response = requests.post(
            url="http://localhost:11434/api/generate", json=payload
        ).json()
        self.context = response["context"]

        answer = response["response"]
        return answer


# rag = RAG()

# html_dir = '/content/drive/MyDrive/RAG/demuc'
# html_paths = [os.path.join(html_dir, filename) for filename in os.listdir(html_dir) if filename.endswith('.html')]
# rag.create_vectordb(html_paths)
