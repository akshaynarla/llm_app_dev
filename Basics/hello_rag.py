# The goal here is to create a simple RAG (Retrieval-Augmented Generation) system based on the data loaded (from various sources).
# You ask LLM questions about the lengthy document and it will provide an answer.
# Fitting data to the context window of the LLM becomes important to obtain accurate answers.
# Steps: a. Indexing b. Retrieval and Generation
# Indexing: Load: with Document Loaders from LangChain, Split: Break the large documents to smaller chunks (to fit context window mainly)
#           and Store: the splits and index them for searching. This is done with VectorStore and Embeddings.
# Given a query, we can embed it as a vector of the same dimension and use similarity metrics to identify related text.
# For ex: For PDFs (most common document type), we use PyPDFLoader to load the document. Preprocessing is needed to optimize the text for embeddings.
#        Extract the text from PDF and separate other types, then it needs to be handled if there are lots of pages through segmentation and cleanup 
#        of the text by formatting removal and excessive whitespace removals.
# Currently, we will do only text here, as this is just an introduction to RAG with LangChain. So in case of images or some text on images, there will be
# loss of context or some randomness.
# https://python.langchain.com/docs/how_to/#document-loaders
# DocumentLoaders in LangChain is used to load documents from a variety of sources.

import dotenv
import time
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import List, TypedDict

# Load API key from .env file
# load_dotenv()
key_location = dotenv.dotenv_values("./geneai.env")
google_api_key = key_location['GOOGLE_API_KEY']

# PDF File Path
pdf_file_path = "./Resources/JSS_ROS_2020.pdf"

# Configure the Generative AI model. Load the emebddings (so that machine understands what the data is) and 
# the vector store.
flash = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", api_key=google_api_key)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=google_api_key)
vector_store = InMemoryVectorStore(embeddings)

template = """
You are a helpful assistant. Answer the user's question in less than 50 words based only on the following context:

Context:
{context}

Question:
{question}
"""

# Ecample: https://python.langchain.com/docs/tutorials/rag/#orchestration for keeping track of question, 
# context retreived from the pdf and generated answer
class RAGState(TypedDict):
    question: str
    context: List[Document]
    answer: str

# load the PDF document and convert each page of the document into a list element
# same step is true for any other loading.
def load_pdf(file_path):
    print("Loading the file from" + file_path + "....")
    pdf_reader = PyPDFLoader(file_path)
    loaded_doc = pdf_reader.load()
    return loaded_doc

# split loaded document into chunks for effecive storage
# splits based on characters and returns a list of documents with these split content
# RecursiveCharacterTextSplitter: allows splitting using common separators like new lines. 
#                                 This is recommended by the LangChain for generic text use-cases.
def split_doc(loaded_doc):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap=25)
    split_chunks = text_splitter.split_documents(loaded_doc)
    print("Splitting the loaded document...")
    return split_chunks

# store the split chunks onto the vector database in the form of embedding vectors
# Vector search is the common way of doing this over unstructured data. 
# Output of embeddings would be array of numbers --< allows semantic understanding ultimately
# 1. For verifying how vectors look: vector_1 = embeddings.embed_query(chunks[0].page_content; print(vector_1[:10])
# 2. Test the vector storage: results = vector_store.similarity_search("What is a ROS-based System?"); print(results[0].page_content)
# In Summary: Used for storing and querying the embedding.  
# # ideal scenarios:  document_ids = vector_store.add_documents(documents=all_chunks); causes flooding of APIs and get Error 429 response 
def store_chunks_vectordb(all_chunks, store):
    batch_size = 25  
    # Loop through the chunks in batches
    # Wait for a while before processing the next batch
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(all_chunks) - 1)//batch_size + 1}...")
        store.add_documents(documents=batch)
        print("Waiting for 30 seconds to avoid rate limiting...")
        time.sleep(30)
        
    print("All chunks have been successfully stored into the vector store")
    return store

# Retrieves relevant documents from the vector store
def retrieve(state: RAGState, retriever):
    print("Retrieving relevant documents for context......")
    question = state["question"]
    retrieved_docs = retriever.invoke(question)
    return {"context": retrieved_docs}

# Generates an answer based on the query and retrieved context
def generate(state: RAGState, llm):
    print("Now preparing your answer......")
    question = state["question"]
    context_docs = state["context"]
    # Create the prompt from the context
    context_str = "\n\n".join(doc.page_content for doc in context_docs)
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context_str})
    
    return {"answer": answer}

# main()
if __name__ == "__main__":
    # First part: Indexing
    loaded_pages = load_pdf(pdf_file_path)
    chunks = split_doc(loaded_pages)
    vector_index = store_chunks_vectordb(chunks, vector_store)

    # Second part: Retrieval and Generation
    retriever = vector_index.as_retriever(search_kwargs = {"k":4})

    print("\n RAG System with the loaded PDF ready. Ask a question or type 'exit' to quit. ---")
    while True:
        user_question = input("\nYou: ")
        if user_question.lower() in ['exit', 'quit']:
            break

        current_state = RAGState(question=user_question, context=[], answer="")

        retrieval_result = retrieve(current_state, retriever)
        current_state.update(retrieval_result)

        generation_result = generate(current_state, flash)
        current_state.update(generation_result)

        print(f"\nLLM: {current_state['answer']}")