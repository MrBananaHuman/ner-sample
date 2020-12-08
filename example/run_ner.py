import argparse
import os
import json
from collections import OrderedDict

from tokenization_kocharelectra import KoCharElectraTokenizer
from transformers import ElectraForTokenClassification, TokenClassificationPipeline
import kss


parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, required=True, default=None, help="Filename of input corpus")
parser.add_argument("--model_name_or_path", type=str, default="monologg/kocharelectra-base-modu-ner-all")
parser.add_argument("--input_dir", default="data", type=str)
parser.add_argument("--output_dir", default="result", type=str)
parser.add_argument("--device", default=-1, type=int, help="Device Num (-1 for cpu)")
args = parser.parse_args()


model = ElectraForTokenClassification.from_pretrained(args.model_name_or_path)
tokenizer = KoCharElectraTokenizer.from_pretrained(args.model_name_or_path)

ner = TokenClassificationPipeline(
    model=model, tokenizer=tokenizer, ignore_labels=["O"], grouped_entities=True, device=args.device
)

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)


instance_lst = []

with open(os.path.join(args.input_dir, args.filename), "r", encoding="utf-8") as f:
    for context in f:
        context = context.strip()
        tag_lst = dict()
        for sent in kss.split_sentences(context):
            result = ner(sent)
            for entity in result:
                word = entity["word"]
                tag = entity["entity_group"]
                score = entity["score"]
                if len(word) > 1 and "[UNK]" not in word:
                    if word in tag_lst:
                        if tag_lst[word][1] < score:
                            tag_lst[word] = [tag, score]
                    else:
                        tag_lst[word] = [tag, score]

        tag_lst = dict(sorted(tag_lst.items(), key=lambda item: -item[1][-1]))
        for k, v in tag_lst.items():
            instance = dict()
            instance["context"] = context
            instance["answer"] = k
            instance["entity"] = v[0]
            instance["score"] = v[1]
            instance["question"] = None
            instance_lst.append(instance)
        instance_lst.append(None)  # Add empty line for each context


output_filename = os.path.splitext(args.filename)[0] + ".jsonl"

with open(os.path.join(args.output_dir, output_filename), "w", encoding="utf-8") as f:
    for instance in instance_lst:
        if instance:
            f.write(f"{json.dumps(instance, ensure_ascii=False)}\n")
        else:
            f.write("\n")
