import argparse
import os

import jsonlines
import json
from transformers import pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default='/data/wufan/algos/data/transformers_data/cluener/cluener/checkpoint-6500',
                    help='initial weights path')
parser.add_argument("--predict_file", type=str, default="", help="the file of predicting")
parser.add_argument("--out", type=str, default="", help="output dir")

opt = parser.parse_args()

# 使用情绪分析流水线
assert opt.predict_file.endswith("json")
assert os.path.isdir(opt.out)

classifier = pipeline('token-classification', model=opt.model_name_or_path)
result = []
with jsonlines.open(opt.predict_file) as reader:
    for line in reader:
        input = line["text"]
        out = classifier(input)
        # print(out)
        for line in out:
            print(line)
            line["score"] = float(line["score"])
        print(f"input:{input}\tprediction:{out}")
        result.append({'input':input,"predict":out})

with open(os.path.join(opt.out,"prediction.json"),"w",encoding="utf-8") as writer:
    json.dump(result,writer,ensure_ascii=False,indent=2)
