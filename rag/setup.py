from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

raw_documents = PyPDFLoader('./test.pdf', extract_images=True).load()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=4000,
    chunk_overlap=1200,
    length_function=len,
    is_separator_regex=True,
)

documents = text_splitter.split_documents(raw_documents)

embeddings = OllamaEmbeddings(model='phi3')
db = Chroma.from_documents(documents, embeddings)

retriever = db.as_retriever(search_type="mmr")

query = "Give me the row entries of \"FORWARD CARGO DOOR\""

docs = retriever.invoke(query)

template = """Answer the question based only on the following context:

{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
llm = Ollama(model="phi3")


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = chain.invoke(query)
print(response)