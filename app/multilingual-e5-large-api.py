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
    temperature=0,
    pad_token_id=tokenizer.eos_token_id,
    top_p=1,
    do_sample=True,
    repetition_penalty=1.2,
)
llm = HuggingFacePipeline(pipeline=pipe)

# ServiceContextの準備
service_context = ServiceContext.from_defaults(
    embed_model=embed_model,
    chunk_size=1024,
    llm=llm,
)

# # dimensions of text-ada-embedding-002
# index = faiss.IndexFlatL2(10)
# # コサイン類似度
# faiss_index = faiss.IndexFlatIP(faiss_index=index)
# vector_store = FaissVectorStore(faiss_index=faiss_index)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)

# APIを実行しFaissのベクター検索ができるようにする
index = VectorStoreIndex.from_documents(documents,
                                     service_context=service_context,
                                    #  storage_context=storage_context
                                     )

query_engine = index.as_query_engine(
    similarity_top_k=10  # 取得するチャンク数 (default:2)
)

response = query_engine.query("リスクベースのアプローチとは？")
print(response)


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
