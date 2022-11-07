#
# img_dir = "/data/wufan/algos/data/transformers_data/cat_vs_dog/train"
# import os
# import jsonlines
# result = []
# for filename in os.listdir(img_dir):
#     # tmp = {"image":{'bytes': None,"path": os.path.join("/data/delta/猫狗大战/train", filename)}}
#     tmp = {"path": os.path.join("/data/delta/猫狗大战/train", filename)}
#
#     if "cat" in filename:
#         tmp["labels"] = "cat"
#     else:
#         tmp["labels"] = "dog"
#     result.append(tmp)
# print(len(result))
# with jsonlines.open("cat_vs_dog_test_auto_dl.json","w") as writer:
#     for line in result[:100]:
#         writer.write(line)
#
#
# # from datasets import load_dataset
# # data_files = "./cat_vs_dog_train.json"
# # ds = load_dataset(
# #     "json",
# #     data_files=data_files,
# #     split="train")
# # labels = ds.features["label"].names
# # label2id, id2label = dict(), dict()
# # for i, label in enumerate(labels):
# #     label2id[label] = str(i)
# #     id2label[str(i)] = label
# # print(id2label)
import jsonlines
path = "/data/wufan/algos/data/transformers_data/cluener/CLUENER2020_test.json"

result = []
with jsonlines.open(path) as reader:
    for line in reader:
        print(line)
        tokens = line["tokens"]
        text = "".join(tokens)
        print(text)
        result.append({"text":text})

with jsonlines.open("cluener_test.json","w") as writer:
    for line in result[:100]:
        writer.write(line)