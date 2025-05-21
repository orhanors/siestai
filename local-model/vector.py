from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("model-setup/custom-test.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./.user/chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        document = Document(
            page_content=str(row["TYPE"]) + " " + str(row["CATEGORY"]) + " " + str(row["TRANSACTION DATE"]) + " " + str(row["FROM CURRENCY"]) + " " + str(row["FROM AMOUNT"]) + " " + str(row["TO CURRENCY"]) + " " + str(row["TO AMOUNT"]) + " " + str(row["FEE CURRENCY"]) + " " + str(row["FEE AMOUNT"]) + " " + str(row["NOTES"]) + " " + str(row["ORIGINAL ID"]),
            metadata={"details": str(row["TYPE"]) + " " + str(row["CATEGORY"]) + " " + str(row["TRANSACTION DATE"]) + " " + str(row["FROM CURRENCY"]) + " " + str(row["FROM AMOUNT"]) + " " + str(row["TO CURRENCY"]) + " " + str(row["TO AMOUNT"]) + " " + str(row["FEE CURRENCY"]) + " " + str(row["FEE AMOUNT"]) + " " + str(row["NOTES"]) + " " + str(row["ORIGINAL ID"])},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
        
vector_store = Chroma(
    collection_name="transactions",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)