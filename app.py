from flask import Flask, request, jsonify, render_template
# from flask_restful import Api, Resource
from flask_cors import CORS
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)


# Set up LangChain components (same as before)
model_local = ChatOllama(model="mistral")
urls = ["https://brainlox.com/courses/category/technical"]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
doc_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OllamaEmbeddings(model='nomic-embed-text'),
)
retriever = vectorstore.as_retriever()

after_rag_template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
after_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | after_rag_prompt 
    | model_local 
    | StrOutputParser()
)

# Serve the frontend
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/query", methods=['POST'])
def query():
    data = request.get_json()
    question = data.get("question", "")
    
    try:
        # Invoke the RAG chain with the question
        response = after_rag_chain.invoke(question)
        
        # Print the response type for debugging
        print("Response type:", type(response))

        if not isinstance(response, str):
            response = str(response)

        # Return the response as JSON
        return jsonify(response=response)
    
    except Exception as e:
        # Print the exception for debugging
        print(f"Exception occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)






# from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.vectorstores import Chroma
# from langchain_ollama import OllamaEmbeddings
# from langchain_ollama import ChatOllama
# from langchain_core. runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.text_splitter import CharacterTextSplitter

# model_local = ChatOllama(model="mistral")

# urls = [
#     "https://brainlox.com/courses/category/technical"
# ]
# docs = [WebBaseLoader(url).load() for url in urls]
# docs_list = [item for sublist in docs for item in sublist]
# text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1500, chunk_overlap = 200)
# doc_splits = text_splitter.split_documents(docs_list)

# # 2. Convert documents to Embeddings and store them
# vectorstore = Chroma. from_documents (
#     documents=doc_splits,
#     collection_name= "rag-chroma",
#     embedding=OllamaEmbeddings(model='nomic-embed-text'),
# )

# retriever = vectorstore.as_retriever()

# # # 3. Before RAG
# # print ("Before RAG\n" )
# # before_rag_template = "What are {topic}"
# # before_rag_prompt = ChatPromptTemplate.from_template (before_rag_template)
# # before_rag_chain = before_rag_prompt | model_local | StrOutputParser ()
# # print(before_rag_chain.invoke({"topic": "least cost courses"}))

# # 4. After RAG
# print("\n########\nAfter RAG\n")
# after_rag_template = """Answer the question based only on the following context:
# {context}
# Question: {question}
# """
# after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
# after_rag_chain = (
# {"context": retriever, "question": RunnablePassthrough ()}
#     | after_rag_prompt 
#     | model_local
#     | StrOutputParser ()
# )

# print(after_rag_chain.invoke("What are the courses with least cost?"))
