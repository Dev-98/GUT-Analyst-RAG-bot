import streamlit as st
from utils import get_gemini_response, get_context_new


def display_chat_messages() -> None:
    """Print message history
    @returns None
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


st.title("🦙GUT Analyst")

with st.expander("The GUT Analyst for three industry titans: Google, Tesla, and Uber."):
    st.subheader("Project Overview")
    st.markdown(
        """
        The GUT Analyst is a chatbot designed to analyze and answer your questions about three industry titans: Google, Tesla, and Uber. 
        It leverages the power of 2023 data, including annual reports and future plans, to provide you with insights, identify trends, and assist with research.
        """
    )
    

st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.greetings = False

# Display chat messages from history on app rerun
display_chat_messages()

# Greet user
if not st.session_state.greetings:
    with st.chat_message("assistant"):
        intro = "Hi there! I'm the GTU Analyst, your one-stop shop for researching Google, Tesla, and Uber. Let's unlock the secrets of these tech giants together!"
        st.markdown(intro)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": intro})
        st.session_state.greetings = True

# Example prompts
example_prompts = [
    "What is the total revenue for Google Search?",
    "What are the risk factors associated with Google and Tesla?",
    "What are the differences in the business of Google and Uber?",
]

example_prompts_help = [
    "Look for a specific company",
    "Search for profits and loss",
    "Difference beetween business.",
]

button_cols = st.columns(3)

button_pressed = ""

if button_cols[0].button(example_prompts[0], help=example_prompts_help[0]):
    button_pressed = example_prompts[0]
elif button_cols[1].button(example_prompts[1], help=example_prompts_help[1]):
    button_pressed = example_prompts[1]
elif button_cols[2].button(example_prompts[2], help=example_prompts_help[2]):
    button_pressed = example_prompts[2]


if prompt := (st.chat_input("What is up?") or button_pressed):

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    context = ""
    if "google" in prompt.lower():
        context += get_context_new(prompt, 'google')
    if "uber" in prompt.lower():
        context += get_context_new(prompt, 'uber')
    if "tesla" in prompt.lower():
        context += get_context_new(prompt, 'tesla')
        
    response = get_gemini_response(context, prompt)
    
    with st.chat_message("assistant"):
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
