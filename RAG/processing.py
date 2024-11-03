from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
import os
import logging
import warnings
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"), 
        logging.StreamHandler()          
    ]
)

logger = logging.getLogger(__name__)

openai_api_key = os.getenv("OPENAI_API_KEY")


class RAG:
    def __init__(self, model_name="llama3.1"):
        self.model_name = model_name.lower()
        self.embedding_function = SentenceTransformerEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.index = Chroma(
            collection_name="legal_docs",
            embedding_function=self.embedding_function,
            persist_directory="./chromadb",
        )
        self.context = []

        if self.model_name != "llama3.1":
            self.chat_model = ChatOpenAI(
                model_name=model_name,
                api_key=openai_api_key,
                temperature=0.7
            )
        else:
            self.chat_model = ChatOllama(
                model=model_name,
                temperature=0.7
            )

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
                new_doc = Document(
                    page_content=chunk,
                    metadata={"page": doc.metadata["page"], "chunk": i},
                )
                new_doc.metadata["source"] = (
                    f"{new_doc.metadata['page']}-{new_doc.metadata['chunk']}"
                )
                new_doc.metadata["filename"] = filename
                doc_chunks.append(new_doc)
        return doc_chunks

    def docs_to_index(self, docs):
        self.index.add_documents(docs)
        self.index.persist()

    def create_vectordb(self, paths):
        documents = []
        for path in paths:
            text, filename = self.read_html(path)
            documents.extend(self.text_to_docs(text, filename))
        self.docs_to_index(documents)

    def retrieve_relevant_docs(self, query, top_k=5, similarity_cutoff=0.7):
        results = self.index.similarity_search_with_score(query, k=top_k)
        results.sort(key=lambda x: x[1])
        results = [result for result in results if result[1] <= similarity_cutoff]

        if results != []:
            logger.info("Got results")
        return results

    def generate_response(self, query):
        from langchain.schema import HumanMessage, SystemMessage
        
        relevant_docs = self.retrieve_relevant_docs(query)
        
        if relevant_docs:
            context = " ".join([doc.page_content for doc, score in relevant_docs])
            messages = [
                SystemMessage(content="""
                    Bạn là một người cực kì am hiểu luật có suy luận logic xuất sắc, bạn sẽ được hỏi một số câu hỏi về luật của Việt Nam.
                    Câu trả lời nên ngắn gọn, súc tích và dễ hiểu.
                    Thông tin: """ + context),
                HumanMessage(content=query)
            ]
        else:
            messages = [
                SystemMessage(content="""
                    Bạn là một người cực kì am hiểu luật có suy luận logic xuất sắc, bạn sẽ được hỏi một số câu hỏi về luật của Việt Nam.
                    Câu trả lời nên ngắn gọn, súc tích và dễ hiểu, và trả lời bằng cùng ngôn ngữ với câu hỏi.
                    Nếu bạn không chắc chắn về câu trả lời, hãy trả lời 'Tôi không có thông tin về câu hỏi này' bằng ngôn ngữ mà câu hỏi được đặt ra."""),
                HumanMessage(content=query)
            ]

        try:
            response = self.chat_model.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "Error generating response with the specified model"

# Example usage:
# Using GPT-4 (requires OPENAI_API_KEY in .env)
# rag_gpt = RAG(model_name="gpt-4")

# Using Llama 2 (requires local Ollama server)
# rag_llama = RAG(model_name="llama2")

# html_dir = 'RAG/data/demuc'
# html_paths = [os.path.join(html_dir, filename) for filename in os.listdir(html_dir) if filename.endswith('.html')]
# print("Embedding HTML")
# rag_gpt.create_vectordb(html_paths)
# print("Done")