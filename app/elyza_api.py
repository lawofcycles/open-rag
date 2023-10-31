# API import Section
from fastapi import FastAPI, Request
import asyncio
# LLM section import
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# IMPORTS FOR TEXT GENERATION PIPELINE CHAIN
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
import copy

app = FastAPI(
    title="Inference API for ELYZA",
    description="A simple API that use MBZUAI/LaMini-Flan-T5-77M as a chatbot",
    version="1.0",
)

@app.get('/')
async def hello():
    return {"hello" : "Medium enthusiast"}