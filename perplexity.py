
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2TokenizerFast,AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from datasets import load_dataset
import json

device = 'cuda'


def calc_ppl_norm_per_tokem(model, tokenizer, texts):
    total=[]
    token_length=0
    for i,text in enumerate(texts):
        input_ids = tokenizer(text, return_tensors='pt').input_ids
        max_length = 400 #model.config.n_positions
        token_length+=input_ids.shape[1]
        stride = 400

        nlls = []
        for i in tqdm(range(0, input_ids.size(1), stride)):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_id = input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_id.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_id, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len
            nlls.append(neg_log_likelihood)

        ppl = torch.stack(nlls).sum() / end_loc
        if neg_log_likelihood.isnan():
            print(text)
            print(i)
        else:
            total.append(ppl)

    print(token_length)
    return torch.exp(sum(total)/len(total))

def calc_ppl_seq_no_norm(model, tokenizer, texts):
    total=[]
    token_length=0
    for i,text in enumerate(texts):
        input_ids = tokenizer(text, return_tensors='pt').input_ids
        max_length = 400 #model.config.n_positions
        token_length+=input_ids.shape[1]
        stride = 5

        nlls = []
        for i in tqdm(range(0, input_ids.size(1), stride)):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_id = input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_id.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_id, labels=target_ids)
                neg_log_likelihood = outputs[0]
            nlls.append(neg_log_likelihood)

        ppl = torch.stack(nlls).sum()
        if neg_log_likelihood.isnan():
            print(text)
            print(i)
        else:
            total.append(ppl)
    print(token_length)
    return torch.exp(sum(total)/len(total))


def calc_ppl_seq_per_char(model, tokenizer, texts):
    total=[]
    token_length=0
    for i,text in enumerate(texts):
        input_ids = tokenizer(text, return_tensors='pt').input_ids
        max_length = 400#model.config.n_positions
        token_length+=input_ids.shape[1]
        stride = 5

        nlls = []
        for i in tqdm(range(0, input_ids.size(1), stride)):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_id = input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_id.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_id, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len
            nlls.append(neg_log_likelihood)

        ppl = torch.stack(nlls).sum() / len(text)
        if neg_log_likelihood.isnan():
            print(text)
            print(i)
        else:
            total.append(ppl)

    print(token_length)
    return torch.exp(sum(total)/len(total))


def main():
    #GPT-neo
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B",revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)


    #Tobias
    #tokenizer = GPT2TokenizerFast.from_pretrained("../../model_Tobias")
    #model = GPT2LMHeadModel.from_pretrained("../../model_Tobias").to(device)

    #GPT
    #model = GPT2LMHeadModel.from_pretrained('gpt2-xl').to(device)
    #tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')

    # gpt-swe
    #tokenizer = GPT2Tokenizer.from_pretrained("../../GPT-SWE/gpt-swe-model")
    #model = GPT2LMHeadModel.from_pretrained("../../GPT-SWE/gpt-swe-model").to(device)

  #  print(calc_ppl_seq(model, tokenizer, ["bröd och frukost"]))

    """
    data = load_dataset("bertin-project/mc4-sampling", "sv", split="validation", streaming=True,
                        sampling_method="random", factor=0.5)
    result=[]
    for j in range(1):
        counter = 0
        samples = []
        for sample in data:
            samples.append(sample["text"])
            if len(set(samples)) == 1050:
                break
        texts = list(set(samples))
        json.dump({"samples":texts},open('mc4.json','w'))
        break
        ppl = calc_ppl_seq(model, tokenizer, texts)
        result.append(ppl)
        print("ppl "+str(ppl))

    print(sum(result)/len(result))
    """


    data = json.load(open("mc4_en.json"))
    print(calc_ppl_seq_per_char(model,tokenizer,data["samples"][:50]))

    #data = json.load(open('test.json'))
    #print(calc_ppl_seq(model, tokenizer, [data["text"]]))
if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
