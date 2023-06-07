import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

from consts import INDEX_NAME

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


def ingest_docs() -> None:
    loader = ReadTheDocsLoader(path="langchain-docs", encoding="utf-8")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} documents")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs\\", "https://").replace("\\", "/")
        doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} documents into Pinecone")
    embeddings = OpenAIEmbeddings()

    index = pinecone.Index(index_name=INDEX_NAME)
    index.delete(deleteAll=True)

    print("********* Deleted all vectors from Pinecone vectorstore *********")

    Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print("********* Added to Pinecone vectorstore *********")


if __name__ == "__main__":
    ingest_docs()
