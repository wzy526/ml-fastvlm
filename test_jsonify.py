from llava.train.train_dat import TrainingArguments
import json

t1 = TrainingArguments(output_dir='./')
with open("./test.json", "w") as f:
    json.dump(t1, f)