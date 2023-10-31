from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# data source
PERSIST_DIR = "./resource/211122_amlcft_guidelines.pdf"

loader = UnstructuredFileLoader(PERSIST_DIR)
documents = loader.load()
print(f"number of docs: {len(documents)}")
print("--------------------------------------------------")

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
db.save_local("faiss_index")