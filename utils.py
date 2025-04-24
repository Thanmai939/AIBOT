#here we will configure cohere API keys and use 
#that func in app.py
import cohere
import os
import base64
from AI_main import research_and_extract
#import streamlit as st

# Initialize Cohere client with API key
import cohere

#co = cohere.Client("411cZFLCIh5YraBzYKBzChDzSzwVitAE5CscIa5W")  # Use env vars in production

def get_answer(messages):
    # Prepare chat history for Cohere format
    chat_history = []
    for msg in messages[:-1]:
        role = "USER" if msg["role"] == "user" else "CHATBOT"
        chat_history.append({"role": role, "message": msg["content"]})

    
    '''user_prompt =  "You are a helpful AI chatbot that answers questions from the User with briefly in detailed manner. " + messages[-1]["content"]
    # Call Cohere's chat API
    response = co.chat(
        message=user_prompt,
        model="command-r",
        chat_history=chat_history,
        temperature=0.7,  # Lower temperature for more deterministic and focused output (Experiment with values between 0 and 1)
        max_tokens=300,  # Slightly increase max_tokens if point-by-point answers might be a bit longer
        prompt_truncation='AUTO',
        # Additional parameters to consider for neater output:
        #stop_sequences=["\n\n", "."], # Try to encourage shorter, punctuated points
        #k=20, # Top-k sampling: Consider the top k most likely tokens (can increase focus)
        # p=0.75, # Nucleus sampling: Consider a nucleus of tokens with cumulative probability p (can increase focus)
    )'''
    response=research_and_extract(messages[-1]["content"])
    return response

    # Clean up response
    '''reply = response.text.strip()
    if reply.lower() in ["i don't know", "i'm not sure", "sorry, i don't know that", ""]:
        return "Sorry, I don't know that."

    # Ensure response is short
    words = reply.split()
    #return ' '.join(words[:50]) + "..." if len(words) > 50 else reply
    return ' '.join(words)'''
