from fastapi import FastAPI, Request
import asyncio
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
import copy
from langchain.chains.question_answering import load_qa_chain
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI(
    title="Inference API for ELYZA",
    description="A simple API that use elyza/ELYZA-japanese-Llama-2-7b-fast-instruct as a chatbot",
    version="1.0",
)

MODEL_NAME = "elyza/ELYZA-japanese-Llama-2-7b-fast-instruct"
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    do_sample=True,
    top_k=20,
    temperature=0.1,
    # device=device,
)
llm = HuggingFacePipeline(pipeline=pipe)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "参考情報を元に、ユーザーからの質問に簡潔に正確に答えてください。"
text = "{context}\nユーザからの質問は次のとおりです。{question}"
template = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
    bos_token=tokenizer.bos_token,
    b_inst=B_INST,
    system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
    prompt=text,
    e_inst=E_INST,
)
rag_prompt_custom = PromptTemplate(
    template=template, input_variables=["context", "question"]
)
chain = load_qa_chain(llm, chain_type="stuff", prompt=rag_prompt_custom)

@app.get('/model')
async def model():
    inputs = {"input_documents": "", "question": ""}
    res = chain.run(inputs)
    result = copy.deepcopy(res)
    return {"result" : result}