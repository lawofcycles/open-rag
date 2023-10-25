import os

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    download_loader,
    ServiceContext,
    LangchainEmbedding,
)
import faiss
from llama_index.vector_stores import FaissVectorStore
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate


persist_dir = "./resource/211122_amlcft_guidelines.pdf"

CJKPDFReader = download_loader("CJKPDFReader")

loader = CJKPDFReader()
documents = loader.load_data(file=persist_dir)

# 埋め込みモデルの準備
embed_model = HuggingFaceBgeEmbeddings(
    model_name="intfloat/multilingual-e5-large"
    )

import torch
from llama_index.llms import HuggingFaceLLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain.llms import HuggingFacePipeline
import torch

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"

model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,device_map="auto")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=2094,
    temperature=0.1,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    repetition_penalty=1.2,
)
llm = HuggingFacePipeline(pipeline=pipe)

# # ServiceContextの準備
# service_context = ServiceContext.from_defaults(
#     embed_model=embed_model,
#     chunk_size=1024,
#     llm=llm,
# )

# # dimensions of text-ada-embedding-002
# index = faiss.IndexFlatL2(10)
# # コサイン類似度
# faiss_index = faiss.IndexFlatIP(faiss_index=index)
# vector_store = FaissVectorStore(faiss_index=faiss_index)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)

from llama_index.callbacks import CallbackManager, LlamaDebugHandler
llama_debug_handler = LlamaDebugHandler()
callback_manager = CallbackManager([llama_debug_handler])

from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.prompts.base import ChatPromptTemplate

# QAシステムプロンプト
TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        "あなたは世界中で信頼されているQAシステムです。\n"
        "事前知識ではなく、常に提供されたコンテキスト情報を使用してクエリに回答してください。\n"
        "従うべきいくつかのルール:\n"
        "1. 回答内で指定されたコンテキストを直接参照しないでください。\n"
        "2. 「コンテキストに基づいて、...」や「コンテキスト情報は...」、またはそれに類するような記述は避けてください。"
    ),
    role=MessageRole.SYSTEM,
)

# QAプロンプトテンプレートメッセージ
TEXT_QA_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "コンテキスト情報は以下のとおりです。\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "事前知識ではなくコンテキスト情報を考慮して、クエリに答えます。\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]

# チャットQAプロンプト
CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)

# チャットRefineプロンプトテンプレートメッセージ
CHAT_REFINE_PROMPT_TMPL_MSGS = [
    ChatMessage(
        content=(
            "あなたは、既存の回答を改良する際に2つのモードで厳密に動作するQAシステムのエキスパートです。\n"
            "1. 新しいコンテキストを使用して元の回答を**書き直す**。\n"
            "2. 新しいコンテキストが役に立たない場合は、元の回答を**繰り返す**。\n"
            "回答内で元の回答やコンテキストを直接参照しないでください。\n"
            "疑問がある場合は、元の答えを繰り返してください。"
            "New Context: {context_msg}\n"
            "Query: {query_str}\n"
            "Original Answer: {existing_answer}\n"
            "New Answer: "
        ),
        role=MessageRole.USER,
    )
]

# チャットRefineプロンプト
CHAT_REFINE_PROMPT = ChatPromptTemplate(message_templates=CHAT_REFINE_PROMPT_TMPL_MSGS)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index import ServiceContext

text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=4096-3,
    chunk_overlap=20,  # オーバーラップの最大トークン数
    separators=["\n= ", "\n== ", "\n=== ", "\n\n", "\n", "。", "「", "」", "！", "？", "、", "『", "』", "(", ")"," ", ""],
)
node_parser = SimpleNodeParser(text_splitter=text_splitter)

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    node_parser=node_parser,
    callback_manager=callback_manager,
)

index = VectorStoreIndex.from_documents(documents,
                                     service_context=service_context,
                                    #  storage_context=storage_context
                                     )
# クエリエンジンの準備
query_engine = index.as_query_engine(
    similarity_top_k=3,
    text_qa_template=CHAT_TEXT_QA_PROMPT,
    refine_template=CHAT_REFINE_PROMPT,
)

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.WARNING, force=True)
import torch

def query(question):
    print(f"Q: {question}")
    response = query_engine.query(question).response.strip()
    print(f"A: {response}\n")
    torch.cuda.empty_cache()

query("マネロン・テロ資金供与対策におけるリスクベース・アプローチとは？")

from llama_index.callbacks import CBEventType
llama_debug_handler.get_event_pairs(CBEventType.LLM)[0][1].payload

# system_prompt = """以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"""

# # This will wrap the default prompts that are internal to llama-index
# prompt_string = """\n\n### 指示: \n{query_str}: \n\n\n### 応答"""
# query_wrapper_prompt = PromptTemplate.from_template(prompt_string)

# llm = HuggingFaceLLM(
#     context_window=1024,
#     max_new_tokens=256,
#     generate_kwargs={"temperature": 0.7, "do_sample": False},
#     system_prompt=system_prompt,
#     query_wrapper_prompt=query_wrapper_prompt,
#     tokenizer_name="novelai/nerdstash-tokenizer-v1",
#     model_name="stabilityai/japanese-stablelm-instruct-alpha-7b-v2",
#     device_map="auto",
#     stopping_ids=[50278, 50279, 50277, 1, 0],
#     tokenizer_kwargs={"max_length": 4096},
#     # uncomment this if using CUDA to reduce memory usage
#     # model_kwargs={"torch_dtype": torch.float16}
# )


# # クエリエンジンの作成
# query_engine = index.as_query_engine(
#     similarity_top_k=3  # 取得するチャンク数 (default:2)
# )

# response = query_engine.query("リスクベースのアプローチとは？")
# print(response)

# if not os.path.exists(persist_dir):
#     os.mkdir(persist_dir)
# documents = SimpleDirectoryReader("data").load_data()
# index = GPTVectorStoreIndex.from_documents(documents)
# index.storage_context.persist(persist_dir)



# from langchain.embeddings import HuggingFaceEmbeddings
# from llama_index import GPTVectorStoreIndex, ServiceContext, LangchainEmbedding

# # 埋め込みモデルの準備
# embed_model = LangchainEmbedding(HuggingFaceEmbeddings(
#     model_name="intfloat/multilingual-e5-large"
# ))

# # ServiceContextの準備
# service_context = ServiceContext.from_defaults(
#     embed_model=embed_model
# )

# # インデックスの生成
# index = GPTVectorStoreIndex.from_documents(
#     documents, # ドキュメント
#     service_context=service_context, # ServiceContext
# )




# app = Flask(__name__)
# tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
# model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')

# @app.route("/embeddings", methods=["POST"])
# def get_embeddings():
#     content = request.json
#     input_texts = content["text"]
#     # Tokenize the input texts
#     batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

#     outputs = model(**batch_dict)
#     embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

#     # normalize embeddings
#     embeddings = F.normalize(embeddings, p=2, dim=1)
#     scores = (embeddings[:2] @ embeddings[2:].T) * 100
#     return jsonify({"embeddings": scores.tolist()})

# def average_pool(last_hidden_states: Tensor,
#                  attention_mask: Tensor) -> Tensor:
#     last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
#     return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# if __name__ == "__main__":
#     app.run(debug=True)
