from fastapi import FastAPI, Request
import asyncio
import torch
import time
from transformers import pipeline
from transformers import AutoTokenizer
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate
import copy
from langchain.chains.question_answering import load_qa_chain
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

app = FastAPI(
    title="Inference API for ELYZA",
    description="A simple API that use elyza/ELYZA-japanese-Llama-2-7b-fast-instruct as a chatbot",
    version="1.0",
)

# embed model
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

db = FAISS.load_local("faiss_index/mufgir", embeddings)
#db = FAISS.load_local("faiss_index/fiscguide", embeddings)

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
DEFAULT_SYSTEM_PROMPT = "参考情報だけを元にして、ユーザーからの質問に答えてください。参考情報で答えられない質問には「参考情報に記載がないのでわかりません」と答えてください。"
text = "参考情報:{context}\nユーザからの質問は次のとおりです:{question}"
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

# チェーンの準備
chain = load_qa_chain(llm, chain_type="stuff", prompt=rag_prompt_custom)

@app.get('/model')
async def model(question : str):
    start = time.time()
    db = FAISS.load_local("faiss_index/mufgir", embeddings)
    docs = db.similarity_search(question, k=3)
    elapsed_time = time.time() - start
    print(f"検索処理時間[s]: {elapsed_time:.2f}")
    for i in range(len(docs)):
        print(docs[i])

    start = time.time()
    # ベクトル検索結果の上位3件と質問内容を入力として、elyzaで文章生成
    inputs = {"input_documents": docs, "question": question}
    output = chain.run(inputs)
    res = chain.run(inputs)
    result = copy.deepcopy(res)
    print(f"テキスト生成処理時間[s]: {elapsed_time:.2f}")
    for i in range(len(docs)):
        print(docs[i])
    return result.replace('\n\n', '')