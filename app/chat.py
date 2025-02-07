import streamlit as st
import google.generativeai as genai

# Configure your Google API key
GOOGLE_API_KEY = "AIzaSyCxvcEQdd8gON0uigqaA7WG-N1IJrkejos"
genai.configure(api_key=GOOGLE_API_KEY)

def get_default_response(query: str) -> str:
    """Provide default responses when no data is available"""
    try:
        model = genai.GenerativeModel("gemini-pro")
        prompt = f"""
        You are a business analytics assistant. Answer the following question briefly and professionally.
        Keep answers short (1-2 sentences). Don't mention anything about data availability.
        
        Question: {query}
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return "I apologize, but I'm having trouble processing your request at the moment."

def chat_with_context(query: str, context: str = None) -> str:
    """Enhanced chatbot function that works with or without context"""
    if not context:
        return get_default_response(query)
        
    prompt = f"""
You are a data analysis assistant. Analyze and answer questions concisely.
Your responses must be:
1. Short (1-2 sentences maximum)
2. Professional and direct
3. Never mention data availability or context
4. Focus on insights and answers

Context: {context}
Question: {query}
"""
    
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return "I apologize, but I'm having trouble processing your request at the moment."

def clear_chat():
    """Clear the chat history"""
    if 'chat_history' in st.session_state:
        st.session_state.chat_history = []

def display_chat(context: str = None):
    """
    Enhanced chat interface with dynamic updates and better UX
    """
    # Add clear chat button at the top
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            clear_chat()
            st.rerun()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    prompt = st.chat_input("Ask me anything about business analytics...", key="chat_input")

    if prompt:
        # Clear previous user input
        st.session_state.user_input = ""
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Show typing indicator
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_with_context(prompt, context)

        # Display assistant response
        with st.chat_message("assistant"):
            st.write(response)

        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Force a rerun to update the UI
        st.rerun()

