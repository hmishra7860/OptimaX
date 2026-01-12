from dotenv import load_dotenv
import os
import time


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
import gradio as gr

load_dotenv() 

gemini_key = os.getenv("GEMINI_API_KEY")

system_prompt = """You are a OptimaX Ai assistant that helps people find information.
Your name is derived from Optimus prime, the leader of the Autobots from the Transformers franchise.
You are friendly and always provide accurate information.   
your answers should be concise and to the point.
your answers in 2-6 sentences."""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.2,
)

prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        (MessagesPlaceholder(variable_name="history")),
        ("user", "{user_input}")
])


chain = prompt | llm

chat_history = []

def safe_invoke(chain, payload, max_retries=1):
    """
    Call Gemini safely. If quota is exceeded, return a friendly error message.
    """
    try:
        return chain.invoke(payload)
    except Exception as e:
        # Check if it's a quota error
        if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
            class QuotaResponse:
                # mimic response object so .content works
                def __init__(self):
                    self.content = "⚠️ Gemini quota exceeded. Please wait or upgrade your plan."
            return QuotaResponse()
        else:
            # re-raise other errors
            raise e


def respond(message, chat_history):
    if chat_history is None:
        chat_history = []

    # Convert Gradio history → LangChain messages
    history = []
    for msg in chat_history:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))

    # Call Gemini safely
    response = safe_invoke(
        chain,
        {
            "user_input": message,
            "history": history
        }
    )

    # Update Gradio chat
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response.content})

    return "", chat_history




##################################################################################################################
### This below code is to run in terminal;

'''while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    response = chain.invoke(
        {"user_input": user_input, "history": history}
    )

    print("OptimaX :", response.text)

    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response.text)) 

'''


    ######################################################################################################

page = gr.Blocks(
    title="OptimaX Ai Assistant",
    theme=gr.themes.Base(),
)

with page:
    gr.Markdown("# OptimaX Assistant")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Enter your message:")
    clear = gr.Button("Clear")

    msg.submit(
        respond,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )

    clear.click(
        lambda: [],
        None,
        chatbot,
        queue=False
    )

page.launch(share=True)
