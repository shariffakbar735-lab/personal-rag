import pymupdf

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# 1.start a local llm
llm = OllamaLLM(
    model="llama3.2:1b",
    base_url="http://localhost:11434",
)

# 2.Load a PDF document
pdf_path = "Sex_Politics_and_Religion_Consciousness.pdf"
with pymupdf.open(pdf_path) as pdf:
    pages = [
        Document(
            page_content=page.get_text(),
            metadata={"source": pdf_path, "page": page.number},
        )
        for page in pdf
    ]
print(f"Loaded {len(pages)} pages from the PDF document.")

# 3.split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
chunks = text_splitter.split_documents(pages)
print(f"Split the document into {len(chunks)} chunks.")

# 4. Generate embeddings for each chunk
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
)
# 5. Create a vector store from the embeddings
print(f"Embedding and storing {len(chunks)} chunks in Chroma...", flush=True)
vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./cprs_db")
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print(f"Created vector store with {len(chunks)} document chunks.")
# 6.processing the information and generating a response
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions based on the provided context., you must provide the information from the context and not from your own knowledge, if the answer is not in the context, say 'I don't know'"),
    ("system", "{history}"),
    ("system", "Context: {context}"),
    ("user", "{question}")
])

chat_history = []
print("\nType 'exit' to quit the conversation.\n")
# 7. conversation loop
while True:
    question = input("User: ")
    if question.lower() == "exit":
        break

    # Retrieve relevant context from the vector store
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])
    print(f"Retrieved context:\n{context}\n")
    chat_history.append(("User", question))
    history = "\n".join(f"{role}: {content}" for role, content in chat_history)

    # Generate a response using the LLM
    response = llm.invoke(prompt.format(history=history, context=context, question=question))
    print(f"Assistant: {response}\n")
    chat_history.append(("Assistant", response))