from tqdm import tqdm
from pprint import pprint
from rouge import Rouge

import argparse
import os
from bs4 import BeautifulSoup
from googlesearch import search
import numpy as np
import requests
from transformers import GPT2Config, GPT2LMHeadModel
import torch
from tqdm import tnrange, tqdm_notebook
from dataset import GPT21024Dataset
from utils import add_special_tokens, beam_search, generate_beam_sample, generate_sample, sample_seq, set_seed, top_k_top_p_filtering


def load_model():
	"""
	Create model and load weights
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed",default=42, type=int,  help="seed to replicate results")
	parser.add_argument("--num_workers",default=4, type=int,  help="num of cpus available")
	parser.add_argument("--device",default=torch.device('cuda'), help="torch.device object")
	parser.add_argument("--output_dir",default='./output', type=str,  help="path to save evaluation results")
	parser.add_argument("--model_dir",default='./weights', type=str,  help="path to save trained model")
	parser.add_argument("--root_dir",default='./CNN/gpt2_1024_data', type=str, help="location of json dataset.")
	parser.add_argument("--ids_file",default='./CNN/ids.json', type=str, help="location of train, valid and test file indexes")
	args = parser.parse_args([])
	print(args)

	# using the same validation and training data as during training
	tokenizer = add_special_tokens()
	# train_data = GPT21024Dataset(args.root_dir,args.ids_file,mode='train',length=3000)
	# valid_data = GPT21024Dataset(args.root_dir,args.ids_file,mode='valid',length=500)
	test_data = GPT21024Dataset(args.root_dir,args.ids_file,mode='test',length=500)
	print(len(test_data))

	# path to model and config files
	model_file = "gpt2_models/final_model.bin"
	config_file = "gpt2_models/final_config.json"

	config = GPT2Config.from_json_file(config_file)
	model = GPT2LMHeadModel(config)
	state_dict = torch.load(model_file, map_location=torch.device('cpu'))
	model.load_state_dict(state_dict)
	model.eval()
	model.to(args.device)

	#result = generate_sample(test_data, tokenizer, model, num=2, length=100, temperature=1, top_k=10, top_p=0.5, device=args.device)

	#rouge = Rouge()

	#for res in result:
	#	article = res[0]
	#	generated_summary = res[1]
	#	reference_summary = res[2].split("<|pad|>")[0]

	#	print(article)
	#	print("-----------")
	#	print(generated_summary)
	#	print("-----------")
	#	print(reference_summary)
	#	print("-----------")

	#	scores = rouge.get_scores(generated_summary, reference_summary)
	#	print(scores)
	#	print("**************")
	#	input()
	#return result

	return model, tokenizer



def hugging_face_decode(model, tokenizer):
	from datasets import load_dataset

	dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")

	for data in tqdm(dataset):
		article = data["article"]
		highlight = data["highlights"]
		data_id = data["id"]
		generated_summary_path = f"generated_summaries/{data_id}"
		reference_summary_path = f"reference_summaries/{data_id}"
		article_path = f"article/{data_id}"


		# try:
    # generate summary
		article_encoded = tokenizer.encode(article)[:900]
		generated_text = sample_seq(model, article_encoded, 50, torch.device('cuda'), temperature=1, top_k=10, top_p=0.5)
		generated_text = generated_text[0, len(article_encoded):].tolist()
		text = tokenizer.convert_ids_to_tokens(generated_text,skip_special_tokens=True)
		text = tokenizer.convert_tokens_to_string(text)


		# save article
		with open(article_path, "w") as f:
			print(article)
			f.write(article)

		# save reference_summary
		with open(reference_summary_path, "w") as f:
			f.write(highlight)

		# save generated summary
		with open(generated_summary_path, "w") as f:
			f.write(text)

		# except Exception as e:
		# 	print(f"[ERROR] on {article_path}")
		# 	print(e)
		# 	input()





def small_test_decode(model, tokenizer):
  test_data = GPT21024Dataset("./CNN/gpt2_1024_data", "./CNN/ids.json", mode='test', length=500)
  print(f"[LOG] Loaded test dataset, #articles: {len(test_data)}")



if __name__ == "__main__":
	model, tokenizer = load_model()
	hugging_face_decode(model, tokenizer)
