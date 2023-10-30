from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain


# data source
PERSIST_DIR = "./resource/211122_amlcft_guidelines.pdf"

loader = UnstructuredFileLoader(PERSIST_DIR)
documents = loader.load()
print(f"number of docs: {len(documents)}")
print("--------------------------------------------------")
print(documents[0].page_content)

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=600,
    chunk_overlap=20,
)

# text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
#     tokenizer,
#     chunk_size=300,
#     chunk_overlap=20,
#     # separators=["\n= ", "\n== ", "\n=== ", "\n\n",
#     #              "\n", "。", "「", "」", "！",
#     #              "？", "、", "『", "』", "(", ")"," ", ""],
# )

splitted_texts = text_splitter.split_documents(documents)
print(f"チャンクの総数：{len(splitted_texts)}")
print(f"チャンクされた文章の確認（20番目にチャンクされたデータ）：\n{splitted_texts[20]}")

# embed model
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

db = FAISS.from_documents(splitted_texts, embeddings)

question = "リスクベースのアプローチとはなんですか。"

start = time.time()
# 質問に対して、データベース中の類似度上位3件を抽出。質問の文章はこの関数でベクトル化され利用される
docs = db.similarity_search(question, k=3)
elapsed_time = time.time() - start
print(f"処理時間[s]: {elapsed_time:.2f}")
for i in range(len(docs)):
    print(docs[i])

# setup LLM
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
    max_new_tokens=512,
    do_sample=True,
    top_k=20,
    temperature=0.7,
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

# チェーンの準備
chain = load_qa_chain(llm, chain_type="stuff", prompt=rag_prompt_custom)

# RAG ありの場合
start = time.time()
# ベクトル検索結果の上位3件と質問内容を入力として、elyzaで文章生成
inputs = {"input_documents": docs, "question": question}
output = chain.run(inputs)
elapsed_time = time.time() - start
print("RAGあり")
print(f"処理時間[s]: {elapsed_time:.2f}")
print(f"出力内容：\n{output}")
print(f"トークン数: {llm.get_num_tokens(output)}")

###################################################
# メモリの解放

del model, tokenizer, pipe, llm, chain
torch.cuda.empty_cache()

# # query付きのHuggingFaceEmbeddings
# class HuggingFaceQueryEmbeddings(HuggingFaceEmbeddings):
#     def __init__(self, **kwargs: Any):
#         super().__init__(**kwargs)

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         return super().embed_documents(["query: " + text for text in texts])

#     def embed_query(self, text: str) -> List[float]:
#         return super().embed_query("query: " + text)

# embed_model = LangchainEmbedding(
#     HuggingFaceQueryEmbeddings(model_name=EMBED_MODEL_NAME)
# )

# # setup LLM
# MODEL_NAME = "elyza/ELYZA-japanese-Llama-2-7b-fast-instruct"

# # Tokenizer
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

# # Model
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     device_map="auto",
#     use_auth_token=False,
#     quantization_config=quantization_config,
# ).eval()

# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=2000,
#     pad_token_id=tokenizer.eos_token_id,
#     repetition_penalty=1.2,
# )

# TEXT_QA_SYSTEM_PROMPT = ChatMessage(
#     content=(
#         "あなたは世界中で信頼されているQAシステムです。\n"
#         "事前知識ではなく、常に提供されたコンテキスト情報を使用してクエリに回答してください。\n"
#     ),
#     role=MessageRole.SYSTEM,
# )

# TEXT_QA_PROMPT_TMPL_MSGS = [
#     TEXT_QA_SYSTEM_PROMPT,
#     ChatMessage(
#         content=(
#             "コンテキスト情報は以下のとおりです。\n"
#             "---------------------\n"
#             "{context_str}\n"
#             "---------------------\n"
#             "事前知識ではなくコンテキスト情報のみを考慮して、Queryに答えてください。\n"
#             "Query: {query_str}\n"
#             "Answer: "
#         ),
#         role=MessageRole.USER,
#     ),
# ]

# CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)

# llm = HuggingFacePipeline(pipeline=pipe)





# node_parser = SimpleNodeParser(text_splitter=text_splitter)

# # from llama_index.callbacks import CallbackManager, LlamaDebugHandler
# # llama_debug_handler = LlamaDebugHandler(print_trace_on_end=True)
# # callback_manager = CallbackManager([llama_debug_handler])

# # # ServiceContextの準備
# # service_context = ServiceContext.from_defaults(
# #     embed_model=embed_model,
# #     chunk_size=1024,
# #     node_parser=node_parser,
# #     llm=llm,
# #     callback_manager=callback_manager
# # )

# # index = VectorStoreIndex.from_documents(
# #     documents,
# #     service_context=service_context,
# # )

# # query_engine = index.as_query_engine(
# #     similarity_top_k=10,
# #     text_qa_template=CHAT_TEXT_QA_PROMPT,
# # )

# # def query(question):
# #     print(f"Q: {question}")
# #     response = query_engine.query(question).response.strip()
# #     print(f"A: {response}\n")
# #     torch.cuda.empty_cache()

# # query("マネロン・テロ資金供与対策におけるリスクベース・アプローチとは？")

# # from llama_index.callbacks import CBEventType
# # event_pairs = llama_debug_handler.get_event_pairs(CBEventType.CHUNKING)
# # print(event_pairs[0][0].payload.keys())  # get first chunking start event
# # print(event_pairs[0][1].payload.keys())  # get first chunking end event