
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
import torch

tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)

# initialize with RagRetriever to do everything in one forward call
input_dict = tokenizer.prepare_seq2seq_batch("How many people live in Paris?", "In Paris, there are 10 million people.", return_tensors="pt")
input_ids = input_dict["input_ids"]

# model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)
# outputs = model(input_ids=input_ids, labels=input_dict["labels"])


# or use retriever separately
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", use_dummy_dataset=True)

# 1. Encode
question_hidden_states = model.question_encoder(input_ids)[0]
# 2. Retrieve
docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
doc_scores = torch.bmm(question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)).squeeze(1)

print(docs_dict.keys())
print("doc_score: ", doc_scores.size())

context_input_ids = docs_dict["context_input_ids"][0]
generated_string = tokenizer.batch_decode(context_input_ids, skip_special_tokens=True)
print("context_input_ids: ", generated_string)

# 3. Forward to generator
outputs = model(context_input_ids=docs_dict["context_input_ids"], context_attention_mask=docs_dict["context_attention_mask"], doc_scores=doc_scores, decoder_input_ids=input_dict["labels"])
# or directly generate
generated = model.generate(context_input_ids=docs_dict["context_input_ids"], context_attention_mask=docs_dict["context_attention_mask"], doc_scores=doc_scores)
generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)
print(generated_string)
