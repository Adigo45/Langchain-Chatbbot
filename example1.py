import os
from constant import openai_key
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain



load_dotenv()

import streamlit as st


# streamlit framwork
st.title("Celebrity Search Results")
input_text = st.text_input("Search the topic you want")




# Prompt Templates
first_input_prompt = PromptTemplate(

    input_variables = ['name'],
    template = "Tell me about celebrity {name}"
)


### OpenAI llms
llm = OpenAI(temperature = 0.8)

chain = LLMChain(llm = llm, prompt=first_input_prompt,verbose = True, output_key = 'person')

## -----2nd prompt Templates

second_input_prompt = PromptTemplate(
    input_variables = ['person'],
    template =  "when was {person} born"

)

chain2 = LLMChain(llm = llm, prompt= second_input_prompt,verbose = True, output_key = 'dob')

## -----2nd prompt Templates

third_input_prompt = PromptTemplate(
    input_variables = ['dob'],
    template =  "Mention 5 major events happened around {dob} in the world"
)

chain3  = LLMChain(llm = llm, prompt = third_input_prompt, verbose = True, output_key = 'description')


parent_chain =  SequentialChain(chains=[chain,chain2,chain3],input_variables = ['name'], output_variables = ['person','dob','description'], verbose = True)



if input_text:
    st.write(parent_chain({'name':input_text}))


