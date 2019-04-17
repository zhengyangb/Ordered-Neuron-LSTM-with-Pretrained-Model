# Dev Branch
# NLU2019  
April 15   
run with `--cuda --mode GPT --learning_rate 1e-5 --lr 10 --batch_size 16 --dropoute 0.0 --dropout 0.45 --dropouth 0.3 --dropouti 0.0 --wdrop 0.45 --chunk_size 10 --seed 141 --epoch 15`  
- Distinguish GPT optimizer 
- Return GPT LM result
- Weighted loss
- Hyper-parameter
  - Turn off embedding drop out
  - decrease learning rate  

April 8  
tools/id2gptid  
util.get_batch_gpt  
GPT_model  
main_gpt  
  
TODO:  
Resume setting
Check accuracy   
Inprove efficiency  
Experiments design  

