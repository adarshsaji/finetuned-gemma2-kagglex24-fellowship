import keras_nlp
import tensorflow as tf
from typing import List, Tuple
import gradio as gr
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Load the model
causal_lm = keras_nlp.models.BartCausalLM.from_preset("kaggle://adarshsaji/gemma/keras/gemma-startup-engine")

class ChatInterface:
    def __init__(self):
        self.model = causal_lm
        self.chat_history: List[Tuple[str, str]] = []
        self.system_prompt = """
You are an AI Tech Consultant, skilled in simplifying technical processes for non-technical users. Guide them in turning ideas into products by offering practical, clear advice on:

- Feasibility: Help assess idea feasibility, complexity, and time commitment.
- Product Development: Outline the step-by-step journey from idea to launch, including prototyping, testing, and iteration.
- Tech Basics: Simplify front-end, back-end, full-stack, and data technologies, explaining their roles in simple terms.
- Cost Estimation: Set realistic expectations for costs, covering team size, tech stack, time, and hidden factors.
- Tech Recommendations: Suggest suitable technologies and best practices for their idea (e.g., frameworks, cloud solutions).
- Collaboration: Advise on team roles (e.g., front-end developers, back-end engineers) and effective communication strategies.
- Clarity: Avoid jargon; use analogies and simple language to build user confidence.
- If a question is asked out of the above said context respond I can't answer this question.

Think step by step, use careful reasoning. 

"""
        self.MAX_HISTORY = 5  # Maximum number of previous exchanges to keep
        self.MIN_CHARS = 10  # Minimum characters required for submission

    async def process_message(self, message: str) -> str:
        # Format messages for the model
        formatted_history = self.format_history()
        prompt = f"{self.system_prompt}\n\n{formatted_history}Instruction:\n{message}\n\nResponse:\n"
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Update chat history
        self.update_history(message, response)
        
        return response

    def format_history(self) -> str:
        return "".join(f"Instruction:\n{human}\n\nResponse:\n{assistant}\n\n" 
                       for human, assistant in self.chat_history[-self.MAX_HISTORY:])

    def generate_response(self, prompt: str) -> str:
        template = "Instruction:\n{Instruction}\n\nResponse:\n{Response}"
        input_text = template.format(Instruction=prompt, Response="")

        outputs = self.model.generate(input_text, max_length=1000, strip_prompt=True)
        
        return outputs

    def update_history(self, message: str, response: str):
        self.chat_history.append((message, response))
        if len(self.chat_history) > self.MAX_HISTORY:
            self.chat_history = self.chat_history[-self.MAX_HISTORY:]

    def create_interface(self):
        with gr.Blocks(css="footer {visibility: hidden}") as interface:
            gr.Markdown("<h1 style='text-align: center;'>🤖 Startup Buddy</h1>")
            gr.Markdown("<p style='text-align: center;'>I can answer your startup/tech world related questions!</p>")
            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
                container=True,
                scale=1,
                value=[[None, "Here's what I can do to help you on your product development journey:\n\n" +
                       "* Answer recent tech/start-up related questions precisely between October 2023 - October 2024. \n" +
                       "* Try `What is the current status of Waymo's commercial robotaxi operations in California?` or `What is the market size and potential for healthcare billing systems, and how can startups capitalize on this opportunity?` * \n" +
                       "* Understand your market: I can analyze trends, funding patterns, and competitor activity to help you figure out if your product idea has a good chance of success.\n" +
                       "* Map out your path: I'll guide you through the entire process, from brainstorming to launch, with clear steps and actionable advice.\n" +
                       "* Know your competition: I can identify your potential rivals, highlight their strengths and weaknesses, and suggest ways to make your product stand out.\n" +
                       "* Attract investors: I can share insights on recent funding rounds, investor preferences, and valuation trends to help you prepare for fundraising.\n" +
                       "* Choose the right tech: I'll recommend technologies and practices that are popular and effective in the startup world.\n" +
                       "* Anticipate challenges: I can point out potential roadblocks based on common startup struggles and suggest strategies to overcome them.\n" +
                       "* Plan for growth: I can share successful growth tactics used by other startups to help you scale your business.\n\n" +
                       "Essentially, I'm here to make the complex world of product development more understandable and manageable for you.\n\n" +
                       "What's your product idea? Let's explore it together!"]],
                show_copy_button=True,
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Type your message here...",
                    placeholder="Type your message here...",
                    show_label=False,
                    container=True,
                )
            with gr.Row():      
                submit_btn = gr.Button(
                    "Submit", 
                    size="lg",                  
                )
                clear = gr.Button(
                    "Clear Chat",
                    size="lg"
                )

            async def respond(message, chat_history):
                if len(message) < self.MIN_CHARS:
                    return message, chat_history
                
                # Clear the initial message when the first user message is sent
                if len(chat_history) == 1 and chat_history[0][0] is None:
                    chat_history = []
                bot_message = await self.process_message(message)
                chat_history.append((message, bot_message))
                return "", chat_history

            def check_input_length(message):
                return gr.update(interactive=len(message) >= self.MIN_CHARS)

            msg.submit(
                respond,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot],
                api_name="submit"
            ).then(
                lambda: gr.update(interactive=False),
                outputs=submit_btn
            )
            
            submit_btn.click(
                respond,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot],
                api_name="submit_btn"
            ).then(
                lambda: gr.update(interactive=False),
                outputs=submit_btn
            )
            
            msg.change(
                check_input_length,
                inputs=msg,
                outputs=submit_btn
            )
            
            clear.click(lambda: None, None, chatbot, queue=False)


        return interface

def main():
    chat_interface = ChatInterface()
    interface = chat_interface.create_interface()
    interface.launch(share=True)

if __name__ == "__main__":
    main()