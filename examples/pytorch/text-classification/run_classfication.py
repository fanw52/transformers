import argparse
import os

import jsonlines
import json
from transformers import pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default='',
                    help='initial weights path')
parser.add_argument("--predict_file", type=str, default="", help="the file of predicting")
parser.add_argument("--out", type=str, default="", help="output dir")

opt = parser.parse_args()

# 使用情绪分析流水线
classifier = pipeline('text-classification', model=opt.model_name_or_path)
out = classifier('愤怒！！！')
print(out)

assert opt.predict_file.endswith("json")
assert os.path.isdir(opt.out)

result = []
with jsonlines.open(opt.predict_file) as reader:
    for line in reader:
        text = line["text"]
        out = classifier(text)
        print(f"text:{text}\tprediction:{out[0]['label']}")
        result.append({'text':text,"predict":out[0]['label']})
with open(os.path.join(opt.out,"prediction.json"),"w",encoding="utf-8") as writer:
    json.dump(result,writer,ensure_ascii=False,indent=2)