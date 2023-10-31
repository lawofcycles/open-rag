from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import argparse

# Parserの作成
parser = argparse.ArgumentParser(description='引数のPDFのパスを読み込んでFAISSのインデックスを作成する')

# 引数の追加
parser.add_argument('arg1', type=str, help='pdfのパス')
args = parser.parse_args()

# data source
PERSIST_DIR = "./resource/211122_amlcft_guidelines.pdf"

loader = UnstructuredFileLoader(args.arg1)
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
db.save_local("faiss_index/" + args.arg1)
