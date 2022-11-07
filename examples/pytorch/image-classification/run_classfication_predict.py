import argparse
import os

import jsonlines
import json
from transformers import pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default='/data/wufan/algos/data/transformers_data/cat2dog',
                    help='initial weights path')
parser.add_argument("--predict_file", type=str, default="/data/wufan/algos/data/transformers_data/cat_vs_dog_test_auto_dl.json", help="the file of predicting")
parser.add_argument("--out", type=str, default="/data/wufan/algos/data/transformers_data/cat2dog/output", help="output dir")

opt = parser.parse_args()

# 使用情绪分析流水线
classifier = pipeline('image-classification', model=opt.model_name_or_path)

result = []
with jsonlines.open(opt.predict_file) as reader:
    for line in reader:
        input = line["path"]
        out = classifier(input)
        print(f"input:{input}\tprediction:{out[0]['label']}")
        result.append({'input':input,"predict":out})

with open(os.path.join(opt.out,"prediction.json"),"w",encoding="utf-8") as writer:
    json.dump(result,writer,ensure_ascii=False,indent=2)