from pytorch_pretrained_bert import OpenAIGPTModel
import pickle
from pytorch_pretrained_bert import OpenAIGPTTokenizer

#read corpus data first

tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

dic={}
for index,i in enumerate(corpus.dictionary.idx2word):
    text =i
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    dic[index]=[indexed_tokens[0],indexed_tokens[-1]]

with open('GPT_index.pkl', 'wb') as handle:
    pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
model = OpenAIGPTModel.from_pretrained('openai-gpt')

with open('GPT_index.pkl', 'rb') as handle:
    dic = pickle.load(handle)

def get_hidden(input_tensor,dic,gpt_model,use_cuda=args.cuda):
    indexed_tokens = [sum([dic[x.item()] for x in y],[]) for y in input_tensor]
    tokens_tensor = torch.tensor([indexed_tokens])
#     model.eval()
    # If you have a GPU, put everything on cuda
    if use_cuda:
        tokens_tensor = tokens_tensor.to('cuda')
        model.to('cuda')
    # Predict hidden states features for each layer
    with torch.no_grad():
        hidden_states = gpt_model(tokens_tensor)
    size=tuple(hidden_states.shape)
    return hidden_states.view(int(size[0]),int(size[1]),int(size[2] / 2), 2,int(size[-1])).sum(3).squeeze(0)

# data should be of size (data_size, batch_size)
get_hidden(data,dic,model)