import argparse
import os

import jsonlines
import json
from transformers import pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--task",type=str, default='text-classification',
                    help="text-classification,token-classification,image-classification")
parser.add_argument('--model_name_or_path', type=str, default='',
                    help='initial weights path')
parser.add_argument("--predict_file",type=str,default="")
parser.add_argument("--out", type=str, default="", help="output dir")

opt = parser.parse_args()

# 使用情绪分析流水线
classifier = pipeline(opt.task, model=opt.model_name_or_path)
assert opt.predict_file.endswith("json")
assert os.path.isdir(opt.out)
assert opt.task in ["text-classification","token-classification","image-classification"]
result = []

if opt.task == "image-classification":
    key = "path"
else:
    key = "text"

with jsonlines.open(opt.predict_file) as reader:
    for line in reader:
        input = line[key]
        out = classifier(input)
        print(f"input:{input}\tprediction:{out[0]['label']}")
        result.append({'input':input,"predict":out})

with open(os.path.join(opt.out,"prediction.json"),"w",encoding="utf-8") as writer:
    json.dump(result,writer,ensure_ascii=False,indent=2)