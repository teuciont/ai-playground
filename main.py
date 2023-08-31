from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import OpenAI

load_dotenv()

# Loads the documents.
loader = DirectoryLoader('documents', glob="**/*.txt")
documents = loader.load()

# Transforms the document data.
text_splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Setting up vector store.
embeddings = OpenAIEmbeddings()
vectorStore = Chroma.from_documents(texts, embeddings)
llm = OpenAI()

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorStore.as_retriever()
)

def query(q):
    print(f"Query: {q}")
    print(f"Answer: {qa.run(q)}")

query("What are the effects of legislations surrounding emissions on the Australian coal market?")
query("What are China's pland with renewable energy?")