from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from crawler import Crawler
from tqdm.auto import tqdm
import torch
import asyncio
import json
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🚀 GPU is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️ No GPU found. Running on CPU.")

def log_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

async def main():
    try:
        print("\n🌐 Starting web crawl...")
        url = "https://kashimantram.in/"
        with tqdm(total=1, desc="Crawling website") as pbar:
            text_data = await Crawler(url)
            pbar.update(1)
        
        if not text_data:
            print("❌ No data retrieved from crawler!")
            return

        print("\n✂️ Processing documents...")
        documents = [Document(page_content=text) for text in text_data]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2400,
            chunk_overlap=600,
            length_function=len,
            is_separator_regex=False
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks.")

        print("\n🔧 Initializing Nomic embeddings...")
        embedding_model = OllamaEmbeddings(model="nomic-embed-text")
        
        print("\n🛠️ Creating vector database...")
        with tqdm(total=len(chunks), desc="Embedding documents") as pbar:
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                collection_name="simple_rag",
                persist_directory="./chroma_db_nomic"
            )
            pbar.update(len(chunks))
        print("✅ Vector database ready!")

        print("\n🧠 Initializing LLM...")
        llm = ChatOllama(model="gemma3:1b")
        
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five
                different versions of the given user question to retrieve relevant documents from
                a vector database. By generating multiple perspectives on the user question, your
                goal is to help the user overcome some of the limitations of the distance-based
                similarity search. Provide these alternative questions separated by newlines.
                Original question: {question}"""
                    )

        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(search_kwargs={"k": 3}),
            llm,
            prompt=QUERY_PROMPT
        )

        template = """Extract relevant information from this context ONLY:
        {context}
        Question: {question}
        
        Respond in this format:
        {{
            "answer": "concise answer",
            "sources": ["source1", "source2"]
        }}"""
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        print("\n💬 Chat ready! Type 'exit' to quit")
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                    
                print("\nProcessing...")
                with tqdm(total=3, desc="Steps") as pbar:
                    pbar.set_postfix_str("Retrieving context")
                    response = chain.invoke({"question": user_input})
                    pbar.update(3)
                
                try:
                    response_data = json.loads(response.strip())
                    print("\n📋 Response:")
                    print(f"Answer: {response_data.get('answer', 'No answer found')}")
                    if "sources" in response_data:
                        print("Sources:")
                        for src in response_data["sources"]:
                            print(f"- {src}")
                except json.JSONDecodeError:
                    print("\n📋 Response:")
                    print(response)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")

    except Exception as e:
        print(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())