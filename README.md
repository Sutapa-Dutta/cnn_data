https://github.com/francoisstamant/Fine-tuning-for-text-summarization/blob/main/text_summarization.ipynb


https://github.com/karndeepsingh/Custom_Text_Summarizer/blob/main/TRAIN_CUSTOM_NLP_TEXT_SUMMARIZER.ipynb

https://github.com/francoisstamant/Fine-tuning-for-text-summarization

https://github.com/priya-dwivedi/Deep-Learning/blob/master/wikihow-fine-tuning-T5/Tune_T5_WikiHow-Github.ipynb

https://github.com/CurationCorp/curation-corpus/blob/master/examples/bart/finetuning-bart.ipynb
https://github.com/CurationCorp/curation-corpus/blob/master/examples/bart/finetuning-bart.ipynb

#code
============================
#edited

class T5SummDataset:
    def __init__(self, 
                 text, 
                 summarized_text,
                 corrupted_text,
                 tokenizer, 
                 max_length,
                ):
        
        self.text = text
        self.summarized_text = summarized_text
        self.corrupted_text=corrupted_text
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        
        text = self.text[item]
        summarized_text = self.summarized_text[item]
        corrupted_text = self.corrupted_text[item]


        input_text = 'summarization: %s' % (text)
        
        inputs = self.tokenizer(input_text, 
                                 None, 
                                 add_special_tokens=True,
                                 max_length=self.max_length,
                                 truncation = True,
                                 padding = 'max_length',
                                 return_tensors = 'pt'
                                )
        
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        
        outputs = self.tokenizer(summarized_text, 
                                 None, 
                                 add_special_tokens=True,
                                 max_length=100,
                                 truncation = True,
                                 padding = 'max_length',
                                 return_tensors = 'pt'
                                )
        
        output_ids = outputs["input_ids"].squeeze(0)
        
        outputs_corrupted = self.tokenizer(corrupted_text, 
                                 None, 
                                 add_special_tokens=True,
                                 max_length=100,
                                 truncation = True,
                                 padding = 'max_length',
                                 return_tensors = 'pt'
                                )
        
        output_ids_corrupted = outputs_corrupted["input_ids"].squeeze(0)

        
        return {
            'text':input_text,
            'summarized_text':summarized_text,
            'corrupted_text':corrupted_text,
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'output_ids': torch.tensor(output_ids, dtype=torch.long),
            'output_ids_corrupted': torch.tensor(output_ids_corrupted, dtype=torch.long)
        }



import pandas as pd
df_corr=pd.read_csv('../input/noisy-summaries/noisy.csv')

df_corr_train=df_corr[0:80]
df_corr_val=df_corr[80:]
df_corr_val=df_corr_val.reset_index(drop=True)

train_dataset = T5SummDataset(text=df_train['article'].values,
                              summarized_text=df_train['highlights'].values,
                              corrupted_text=df_corr_train['noisy'].values,
                              tokenizer=tokenizer,
                              max_length=max_length,
                             )

val_dataset = T5SummDataset(text=df_val['article'].values,
                            summarized_text=df_val['highlights'].values,
                            corrupted_text=df_corr_val['noisy'].values,
                            tokenizer=tokenizer,
                            max_length=max_length,
                             )



df_train=df_train[0:80]
df_val=df_val[0:20]




lo=[]
la=[]
alpha=0.01
from torch.nn import CrossEntropyLoss
loss_fct=nn.CrossEntropyLoss()


def validation_loop(val_dataloader, model):
    model.eval() 
    val_loss=[]
    bleu = []
    with torch.no_grad():
        for bi, d in enumerate(val_dataloader):
            text = d["text"]
            input_ids = d["input_ids"]
            attention_mask = d["attention_mask"]
            output_ids = d["output_ids"]

            input_ids = input_ids.to(device, dtype=torch.long)
            attention_mask = attention_mask.to(device, dtype=torch.long)
            output_ids = output_ids.to(device, dtype=torch.long)
            
            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            labels=output_ids
                           ) 
            loss = outputs.loss
            val_loss.append(loss.item())
            
            preds_ids = model.generate(input_ids, max_length=128, num_return_sequences=1)
            
            for i in range(preds_ids.shape[0]):
                preds = tokenizer.decode(preds_ids[i].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
                score = bleu_metric.compute([preds], [text])['bleu']
                bleu.append(score)
            
    return model, mean(val_loss), mean(bleu) 
    
    
    
    
    
    
    
    
    
    


