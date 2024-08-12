from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS #导入向量数据库
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

#对请求进行封装,包含用户输入的api key，记忆（需要从外部传入）
def qa_agent(openai_api_key, memory, uploaded_file, question):
    model = ChatOpenAI(model="gpt-3.5-turbo",
                       openai_api_key=openai_api_key,
                       openai_api_base = "https://api.aigc369.com/v1")
    file_content = uploaded_file.read() #对用户上传的文档内容进行读取生成二进制内容
    temp_file_path = "temp.pdf"   #新建储存PDF文件的临时文件路径
    with open(temp_file_path, "wb") as temp_file:  #把二进制内容进行写入
        temp_file.write(file_content)
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    #实例化一个分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n","。","!","?",",","，","、"]
    )
    texts = text_splitter.split_documents(docs)
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key,openai_api_base = "https://api.aigc369.com/v1")    #创建嵌入模型的实例
    db = FAISS.from_documents(texts, embeddings_model)
    retriever = db.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )
    response = qa.invoke({"chat_history":memory, "question":question})
    return response