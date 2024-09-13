from fastapi import FastAPI, UploadFile, File
from typing import List
import uvicorn
from fastapi.responses import JSONResponse
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.llms import HuggingFaceEndpoint
from unstructured.partition.auto import partition
from pydantic import BaseModel
from io import BytesIO
import logging

app = FastAPI()

class QueryRequest(BaseModel):
    user_id: int
    question: str

def split_text(docs: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    texts = text_splitter.split_text(docs)
    return texts

def create_embeddings_model() -> HuggingFaceEmbeddings:
    model_name = "all-mpnet-base-v2"
    model_kwargs = {
        'device': 'cuda',
        'trust_remote_code': True,
        'token': 'api'
    }
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        show_progress=True
    )
    return embeddings

def apply_embeddings(texts: List[str], embeddings: HuggingFaceEmbeddings) -> FAISS:
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore

def save_embeddings(vectorstore: FAISS, path: str) -> None:
    vectorstore.save_local(path)

def load_embeddings(path: str, embeddings: HuggingFaceEmbeddings) -> FAISS:
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

@app.post("/upload/{user_id}")
async def upload_documents(user_id: str, files: List[UploadFile] = File(...)):
    print(f"Handling upload for user_id: {user_id}")
    logging.INFO(user_id)
    documents = []
    for file in files:
        content = await file.read()
        pdf_stream = BytesIO(content)
        elements = partition(file=pdf_stream, metadata_filename=file.filename)
        texts = [el.text for el in elements if el.category == 'NarrativeText']
        documents.extend(texts)

    combined_text = " ".join(documents)
    texts = split_text(combined_text)
    embeddings = create_embeddings_model()
    vectorstore = apply_embeddings(texts, embeddings)
    save_embeddings(vectorstore, f"./Vectorstore/vectorstore_{user_id}.faiss")
    return {"message": "Documents processed and embeddings stored."}

@app.get("/query")
def query_documents(request: QueryRequest):
    embeddings = create_embeddings_model()
    vectorstore = load_embeddings(f"./Vectorstore/vectorstore_{request.user_id}.faiss", embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        top_k=10,
        top_p=0.25,
        max_new_tokens=2024,
        temperature=0.1,
        repetition_penalty=1.03,
    )
    prompt = ChatPromptTemplate.from_template("""
    <expertise>
    You are an expert on Toyota cars, specifically tasked with providing information based on the data uploaded by the user.
    You do not make assumptions or generate knowledge beyond the provided documents.
    </expertise>
                                              
    <context>
    {context}
    </context>

    Question: {input}
    
    <answer>
    Answer only with information directly available or inferable from the provided data.
    </answer>
    """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    response = retrieval_chain.invoke({"input": request.question})
    return JSONResponse(content={"answer": response["answer"]})

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)