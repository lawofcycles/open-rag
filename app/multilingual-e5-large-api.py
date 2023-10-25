import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.WARNING, force=True)
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    download_loader,
    ServiceContext,
    LangchainEmbedding,
)
from llama_index.vector_stores import FaissVectorStore
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline,
)

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
# llama_debug_handler.get_event_pairs(CBEventType.LLM)[0][1].payload


persist_dir = "./resource/211122_amlcft_guidelines.pdf"

CJKPDFReader = download_loader("CJKPDFReader")

loader = CJKPDFReader()
documents = loader.load_data(file=persist_dir)

embed_model_name = "intfloat/multilingual-e5-large"
embed_model = HuggingFaceBgeEmbeddings(
    model_name=embed_model_name,
    )

model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,device_map="auto")

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT =  "あなたは世界中で信頼されているQAシステムです。\n"
"事前知識ではなく、常に提供されたコンテキスト情報を使用してクエリに回答してください。\n"
text = "{context}\nユーザからの質問は次のとおりです。{question}"
template = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
    bos_token=tokenizer.bos_token,
    b_inst=B_INST,
    system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
    prompt=text,
    e_inst=E_INST,
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # max_length=2094,
    # temperature=0.1,
    pad_token_id=tokenizer.eos_token_id,
    # do_sample=True,
    repetition_penalty=1.2,
)
PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"],
    template_format="f-string"
)

chain_type_kwargs = {"prompt": PROMPT}

llm = HuggingFacePipeline(pipeline=pipe)

# llama_debug_handler = LlamaDebugHandler()
# callback_manager = CallbackManager([llama_debug_handler])

index = VectorStoreIndex.from_documents(documents,
                                    #  service_context=service_context,
                                    #  storage_context=storage_context
                                     )
retriever = index.as_retriever(search_kwargs={"k": 3})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
    verbose=True,
)

qa("リスクベースのアプローチとは？")

# service_context = ServiceContext.from_defaults(
#     llm=llm,
#     embed_model=embed_model,
#     node_parser=node_parser,
#     callback_manager=callback_manager,
# )

# # クエリエンジンの準備
# query_engine = index.as_query_engine(
#     similarity_top_k=3,
#     text_qa_template=CHAT_TEXT_QA_PROMPT,
#     refine_template=CHAT_REFINE_PROMPT,
# )


# def query(question):
#     print(f"Q: {question}")
#     response = query_engine.query(question).response.strip()
#     print(f"A: {response}\n")
#     torch.cuda.empty_cache()




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
