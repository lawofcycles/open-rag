import os

from llama_index import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    download_loader,
    ServiceContext,
    LangchainEmbedding,

)
from langchain.embeddings import HuggingFaceEmbeddings

persist_dir = "./resource/211122_amlcft_guidelines.pdf"

CJKPDFReader = download_loader("CJKPDFReader")

loader = CJKPDFReader()
documents = loader.load_data(file=persist_dir)

# 埋め込みモデルの準備
embed_model = LangchainEmbedding(HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large"
))

# ServiceContextの準備
service_context = ServiceContext.from_defaults(
    embed_model=embed_model
)

# インデックスの生成
index = GPTVectorStoreIndex.from_documents(
    documents, # ドキュメント
    service_context=service_context, # ServiceContext
)

# クエリエンジンの作成
query_engine = index.as_query_engine(
    similarity_top_k=3  # 取得するチャンク数 (default:2)
)

response = query_engine.query("リスクベースのアプローチとは？")
print(response)

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
