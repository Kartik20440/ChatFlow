import chatbot 
import streamlit as st
from streamlit_extras.mention import mention

def main():
    st.set_page_config(page_title="ChatBot", page_icon="🦉", layout="centered", initial_sidebar_state="expanded")
    st.markdown(""" <style>
	#MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
	</style> """, unsafe_allow_html=True)

    st.sidebar.title("ChatBot🦉")
    menu = ["Home","Chat","Creator"]
    choice = st.sidebar.radio(" ",menu)

    if choice == "Home":
        st.title("ChatBot🦉")
        st.header("Welcome to the ChatBot!!")
        my_expander1 = st.expander("**What is ChatBot?**")
        my_expander1.write("ChatBot is a conversational agent that can answer your queries related to the content of the website. It uses a Knowledge Graph to understand the context of the question and provide the most relevant answer.")
        my_expander2 = st.expander("**How to use ChatBot?**")
        my_expander2.write("To use the ChatBot, you can select the Chat option from the sidebar and ask your query. The ChatBot will provide you with the most relevant answer based on the content of the website.")


    elif choice == "Chat":        
        st.title("Chat with the ChatBot🦉")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        query = st.chat_input("Enter your query here:")
        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            with st.chat_message("assistant"):
                response = st.write(chatbot.helper(query))
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.write("Please enter a query to get a response.")


    elif choice == "Creator":
        st.header("Kartik Jain")
        st.write("I am a 4th year Computer Science and Economics Undergraduate at IIIT Delhi. I am an academically goal-driven individual who has strong problem-solving skills. I am open to new experiences and opportunities.")
        st.write("Socials:")
        mention(label="Github",icon="github",  url="https://github.com/Kartik20440")
        mention(label="Linkedin",icon="https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/LinkedIn_logo_initials.png/900px-LinkedIn_logo_initials.png?20140125013055",  url="https://www.linkedin.com/in/kartikxjain/")
        st.write("kartik20440@iiitd.ac.in")


if __name__ == '__main__':
    main()
