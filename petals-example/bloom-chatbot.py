#!/usr/bin/env python3

# based on https://colab.research.google.com/drive/1Ervk6HPNS6AYVr3xVdQnY5a-TjjmLCdQ?usp=sharing
# network status: http://health.petals.ml/

import torch
from transformers import BloomTokenizerFast 
from petals import DistributedBloomForCausalLM

MODEL_NAME = "bigscience/bloom-petals"
print("init tokenizer")
tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
print("init model")
model = DistributedBloomForCausalLM.from_pretrained(MODEL_NAME)
#model = model.cuda()

print("init session")
with model.inference_session(max_length=512) as sess:
    while True:
        prompt = input('Human: ')
        if prompt == "":
            break
        prefix = f"Human: {prompt}\nFriendly AI:"
        prefix = tokenizer(prefix, return_tensors="pt")["input_ids"]#.cuda()
        print("Friendly AI:", end="", flush=True)
        
        while True:
            outputs = model.generate(
                prefix, max_new_tokens=1, do_sample=True, top_p=0.9, temperature=0.75, session=sess
            )
            outputs = tokenizer.decode(outputs[0, -1:])
            print(outputs, end="", flush=True)
            if "\n" in outputs:
                break
            prefix = None  # Prefix is passed only for the 1st token of the bot's response