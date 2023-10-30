import sys
import logging
import os
import re
import torch

logging.basicConfig(stream=sys.stdout, level=logging.WARNING, force=True)
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    download_loader,
    ServiceContext,
    LangchainEmbedding,
    SimpleKeywordTableIndex,
)
from llama_index.vector_stores import FaissVectorStore
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline,
)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# import QueryBundle
from llama_index import QueryBundle

# import NodeWithScore
from llama_index.schema import NodeWithScore

# Retrievers
from llama_index.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)

from typing import List

from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import torch
from llama_index.llms import HuggingFaceLLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain.llms import HuggingFacePipeline
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.prompts.base import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index import ServiceContext
from llama_index.callbacks import CBEventType
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch

from transformers import pipeline
from langchain.llms import HuggingFacePipeline

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding
from typing import Any, List

from typing                      import List, Union, Optional, Type
from pathlib                     import Path
from llama_index                 import download_loader, GPTVectorStoreIndex, ServiceContext, OpenAIEmbedding
from llama_index                 import PromptTemplate as LlamaIndexPromptTemplate
from transformers                import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline)
from langchain.chat_models       import ChatOpenAI
from langchain.llms              import OpenAI, HuggingFacePipeline
from langchain.tools             import BaseTool, StructuredTool, Tool, tool
from langchain.agents            import AgentType, AgentExecutor, Tool, LLMSingleActionAgent, AgentOutputParser
from langchain.schema            import OutputParserException
from langchain.schema.agent      import AgentAction, AgentFinish
from langchain.chains            import LLMChain, SequentialChain
from langchain.memory            import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts           import PromptTemplate, StringPromptTemplate
from langchain.embeddings        import HuggingFaceEmbeddings
from langchain.callbacks         import get_openai_callback
from langchain.callbacks.manager import (AsyncCallbackManagerForToolRun, CallbackManagerForToolRun)
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
from langchain.document_loaders import UnstructuredFileLoader

# data source
PERSIST_DIR = "./resource/211122_amlcft_guidelines.pdf"

loader = UnstructuredFileLoader(PERSIST_DIR)
documents = loader.load()
print(f"number of docs: {len(documents)}")
print("--------------------------------------------------")
print(documents[0].page_content)

# embed model
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"

# query付きのHuggingFaceEmbeddings
class HuggingFaceQueryEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return super().embed_documents(["query: " + text for text in texts])

    def embed_query(self, text: str) -> List[float]:
        return super().embed_query("query: " + text)

embed_model = LangchainEmbedding(
    HuggingFaceQueryEmbeddings(model_name=EMBED_MODEL_NAME)
)

# setup LLM
MODEL_NAME = "elyza/ELYZA-japanese-Llama-2-7b-fast-instruct"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    use_auth_token=False,
    quantization_config=quantization_config,
).eval()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=2000,
    pad_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.2,
)

TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        "あなたは世界中で信頼されているQAシステムです。\n"
        "事前知識ではなく、常に提供されたコンテキスト情報を使用してクエリに回答してください。\n"
    ),
    role=MessageRole.SYSTEM,
)

TEXT_QA_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "コンテキスト情報は以下のとおりです。\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "事前知識ではなくコンテキスト情報のみを考慮して、Queryに答えてください。\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]

CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)

llm = HuggingFacePipeline(pipeline=pipe)

text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=300,
    chunk_overlap=20,
    separators=["\n= ", "\n== ", "\n=== ", "\n\n",
                 "\n", "。", "「", "」", "！",
                 "？", "、", "『", "』", "(", ")"," ", ""],
)

splitted_texts = text_splitter.split_documents(documents)
print(f"チャンクの総数：{len(splitted_texts)}")
print(f"チャンクされた文章の確認（参考に7番目にチャンクされたデータを確認）：\n{splitted_texts[6]}")

node_parser = SimpleNodeParser(text_splitter=text_splitter)

# from llama_index.callbacks import CallbackManager, LlamaDebugHandler
# llama_debug_handler = LlamaDebugHandler(print_trace_on_end=True)
# callback_manager = CallbackManager([llama_debug_handler])

# # ServiceContextの準備
# service_context = ServiceContext.from_defaults(
#     embed_model=embed_model,
#     chunk_size=1024,
#     node_parser=node_parser,
#     llm=llm,
#     callback_manager=callback_manager
# )

# index = VectorStoreIndex.from_documents(
#     documents,
#     service_context=service_context,
# )

# query_engine = index.as_query_engine(
#     similarity_top_k=10,
#     text_qa_template=CHAT_TEXT_QA_PROMPT,
# )

# def query(question):
#     print(f"Q: {question}")
#     response = query_engine.query(question).response.strip()
#     print(f"A: {response}\n")
#     torch.cuda.empty_cache()

# query("マネロン・テロ資金供与対策におけるリスクベース・アプローチとは？")

# from llama_index.callbacks import CBEventType
# event_pairs = llama_debug_handler.get_event_pairs(CBEventType.CHUNKING)
# print(event_pairs[0][0].payload.keys())  # get first chunking start event
# print(event_pairs[0][1].payload.keys())  # get first chunking end event