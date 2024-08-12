import streamlit as st

from langchain.memory import ConversationBufferMemory
from utils import qa_agent


st.title("📃克隆GPT 智能PFD问答器💡")

with st.sidebar:
    openai_api_key = st.text_input("请输入OpenAI API密钥：", type="password")
    st.markdown("[获取OpneAI API密钥](https://platform.openai.com/account/api-keys)")

#只有当对话框里没有memory键的时候才初始化记忆
if "memory" not in st.session_state:
    # 创建记忆的实例
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )

uploaded_file = st.file_uploader("上传你的PDF文件：",type="pdf")
question = st.text_input("对PDF的内容进行提问", disabled=not uploaded_file)

if uploaded_file and question and not openai_api_key:
    st.info("请输入你的OpenAI API密钥")

if uploaded_file and question and openai_api_key:
    #添加加载组件
    with st.spinner("NETA CPT正在思考中，请稍等..."):
        response = qa_agent(openai_api_key, st.session_state["memory"],
                            uploaded_file, question)
    st.write("### 答案")
    st.write(response["answer"])
    #把历史对话储存到对话状态里
    st.session_state["chat_history"]= response["chat_history"]

if "chat_history" in st.session_state:
    #历史消息的折叠组件
    with st.expander("历史消息"):
        for i in range(0, len(st.session_state["chat_history"]),2):
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i+1]
            st.write(human_message.content)
            st.write(ai_message.content)
            if i < len(st.session_state["chat_history"]) - 2:
                st.divider()

