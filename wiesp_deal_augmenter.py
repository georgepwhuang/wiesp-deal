from datasets import Dataset
import json
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, BertTokenizer, BertTokenizerFast
from torch.utils.data import DataLoader
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters, PunktLanguageVars
from nltk.tokenize import WhitespaceTokenizer
import pandas as pd
from tqdm import tqdm
from functools import partial
import torch
from pytorch_lightning.utilities.seed import seed_everything

seed_everything(42)

nltk.download('punkt')

with open("./WIESP2022-NER/ner_tags.json") as fp:
    ner_tags = json.load(fp)
ner_tags = [item[1] for item in sorted([(v, k) for k, v in ner_tags.items()])]

tk = WhitespaceTokenizer()
punkt_param = PunktParameters()
abbreviation = ['al', 'fig', 'tab', 'i.e', 'no', 'etal', ]
punkt_param.abbrev_types = set(abbreviation)

class SpacedLangVars(PunktLanguageVars):
    _period_context_fmt = r"""
#         %(SentEndChars)s
        (?=(?P<after_tok>
            (((%(NonWord)s)+\s+)
            |
            \s+)(?P<next_tok>\S+)
        ))"""

pt = PunktSentenceTokenizer(lang_vars=SpacedLangVars(), train_text=punkt_param)

def reconstruct_dataset(filename):
    with open(filename) as fp:
        json_dict = [json.loads(jline) for jline in fp.read().splitlines()]
    data = []
    for item in tqdm(json_dict):
        sentences = " ".join(item["tokens"])
        sentence_list = pt.tokenize(sentences)
        counter = 0
        deduct = 0
        for idx, sentence in enumerate(sentence_list):
            prev_counter = counter
            counter += (len(sentence.strip().split(" ")))
            entry = {}
            entry["tokens"] = item["tokens"][prev_counter: counter]
            entry["bibcode"] = item["bibcode"]
            entry["label_studio_id"] = item["label_studio_id"]
            entry["section"] = item["section"]
            entry["unique_id"] = item["unique_id"]
            entry["part"] = idx
            if "ner_ids" in item and "ner_tags" in item:
                entry["ner_ids"] = item["ner_ids"][prev_counter: counter]
                entry["ner_tags"] = item["ner_tags"][prev_counter: counter]
            data.append(entry)
        if counter != len(item['tokens']):
            assert counter == len(item['tokens'])
    data = pd.DataFrame(data)
    return data

train_dataset = Dataset.from_pandas(reconstruct_dataset("./WIESP2022-NER/WIESP2022-NER-TRAINING.jsonl"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

augment_models = ["roberta-base"]
tokenizers = {}
models = {}
for model_name in augment_models:
    tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
    models[model_name] = AutoModelForMaskedLM.from_pretrained(model_name).to(DEVICE)

def convert_to_features(example_batch, tokenizer):
    texts = example_batch["tokens"]
    texts = [" ".join(text) for text in texts]
    features = tokenizer.batch_encode_plus(
        texts, max_length=384,
        padding=True,
        truncation=True
    )
    return features

loader_columns = [
    "datasets_idx",
    "input_ids",
    "token_type_ids",
    "attention_mask",
    "start_positions",
    "end_positions",
    "labels",
    "word_ids"
]

def align_subwords(prediction, augment_string, mask_token_indexes, tokenizer):
    result = augment_string.copy()
    for idx in mask_token_indexes:
        if not prediction[idx].startswith(" ") ^ augment_string[idx - 1].startswith("Ġ"):
            result[idx-1] = prediction[idx]
        elif isinstance(tokenizer, BertTokenizerFast) or isinstance(tokenizer, BertTokenizer):
            if not prediction[idx].startswith("##") ^ augment_string[idx - 1].startswith("##"):
                result[idx-1] = prediction[idx]
    return result

def get_augment_tokens(augment_string, tokenizer):
    return "".join(augment_string).replace('Ġ', ' ').split(" ")

model_name = "roberta-base"
tokenizer = tokenizers[model_name]
model = models[model_name].to(DEVICE)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.3, return_tensors='pt')
masked_dataset = train_dataset.map(
    partial(convert_to_features, tokenizer=tokenizer),
    batched=True,
    batch_size=32)
columns = [c for c in masked_dataset.column_names if c in loader_columns]
masked_dataset.set_format(type="torch", columns=columns)
data_loader = DataLoader(masked_dataset, batch_size=32, collate_fn=data_collator)
counter = 0
augment_data = []
for sample in tqdm(data_loader):
    sample_d = sample.to(DEVICE)
    with torch.no_grad():
        result = model(**sample_d)
    logits = result.logits
    labels = sample.labels
    input_ids = sample.input_ids
    tokens = sample.tokens
    for idx in range(logits.size()[0]):
        augment_string = tokenizer.tokenize(" ".join(train_dataset[counter]["tokens"]))
        mask_token_indexes = (input_ids[idx] == tokenizer.mask_token_id).nonzero().squeeze(1)
        predicted_token_id = logits[idx].argmax(axis=-1)
        prediction = tokenizer.batch_decode(predicted_token_id)
        mask_token_indexes = mask_token_indexes.detach().cpu().numpy().tolist()
        augment_string = align_subwords(prediction, augment_string, mask_token_indexes, tokenizer)
        augment_string = get_augment_tokens(augment_string, tokenizer)
        assert len(augment_string) == len(train_dataset[counter]["tokens"])
        augment_data.append(augment_string)
        counter += 1
train_dataset = train_dataset.add_column("augment", augment_data)

train_dataset.to_json("WIESP2022-NER/WIESP2022-NER-AUGMENT.jsonl")

