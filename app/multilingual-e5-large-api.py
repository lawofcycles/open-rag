import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from flask import Flask, request, jsonify
from torch import Tensor

app = Flask(__name__)
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')

@app.route("/embeddings", methods=["POST"])
def get_embeddings():
    content = request.json
    input_texts = content["text"]
    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings[:2] @ embeddings[2:].T) * 100
    return jsonify({"embeddings": scores.tolist()})

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

if __name__ == "__main__":
    app.run(debug=True)
