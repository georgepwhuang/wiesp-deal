
# %%
from datasets import Dataset
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import json
from transformers import AutoTokenizer, AutoAdapterModel, get_linear_schedule_with_warmup
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import warnings
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters, PunktLanguageVars
from nltk.tokenize import WhitespaceTokenizer
import pandas as pd
from tqdm.auto import tqdm
from seqeval.metrics import classification_report, accuracy_score
from sklearn.metrics import matthews_corrcoef
from seqeval.scheme import IOB2
import numpy as np
from pytorch_lightning.utilities.seed import seed_everything
import itertools
import torch.nn.functional as F
from pytorch_lightning import loggers as pl_loggers

# %%
seed_everything(42)

# %%
nltk.download('punkt')

# %%
with open("./WIESP2022-NER/ner_tags.json") as fp:
    ner_tags = json.load(fp)
ner_tags = [item[1] for item in sorted([(v, k) for k, v in ner_tags.items()])]

# %%
tk = WhitespaceTokenizer()
punkt_param = PunktParameters()
abbreviation = ['al', 'fig', 'tab', 'i.e', 'no', 'etal', ]
punkt_param.abbrev_types = set(abbreviation)
class SpacedLangVars(PunktLanguageVars):
    _period_context_fmt = r"""
        %(SentEndChars)s             # a potential sentence ending
        (?=(?P<after_tok>
            (((%(NonWord)s)+\s+)            # either other punctuation
            |
            \s+)(?P<next_tok>\S+)     # or whitespace and some other token
        ))"""
pt = PunktSentenceTokenizer(lang_vars = SpacedLangVars(), train_text = punkt_param)

# %%
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

# %%
class DEALDataModule(LightningDataModule):
    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "aug_input_ids",
        "aug_token_type_ids",
        "aug_attention_mask",
        "aug_start_positions",
        "aug_end_positions",
        "labels",
        "word_ids"
    ]

    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.id_map = {}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

    def setup(self, stage: str = None):
        self.dataset = {}
        self.dataset["train"] = Dataset.from_json("/kaggle/input/wiespaugment/WIESP2022-NER-AUGMENT-SCIBERT.jsonl")
        self.dataset["val"] = Dataset.from_pandas(reconstruct_dataset("./WIESP2022-NER/WIESP2022-NER-DEV.jsonl"))
        self.dataset["test"] = Dataset.from_pandas(reconstruct_dataset("./WIESP2022-NER/WIESP2022-NER-VALIDATION-NO-LABELS.jsonl"))
        self.dataset["pred"] = Dataset.from_pandas(reconstruct_dataset("./WIESP2022-NER/WIESP2022-NER-TESTING-NO-LABELS.jsonl"))
        
        counter = 0
        for split in ["train", "val", "test", "pred"]:
            self.dataset[split+"_process"] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                batch_size=-1
            )
            self.columns = [c for c in self.dataset[split+"_process"].column_names if c in self.loader_columns]
            self.dataset[split+"_process"].set_format(type="torch", columns=self.columns)

    def train_dataloader(self):
        return DataLoader(self.dataset["train_process"], batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset["val_process"], batch_size=self.eval_batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset["test_process"], batch_size=self.eval_batch_size)


    def predict_dataloader(self):
        return DataLoader(self.dataset["pred_process"], batch_size=self.eval_batch_size)

    def get_label_idx(self, label, idx):
        if idx == None:
            return len(ner_tags)-1
        else:
            return label[idx]

    def convert_to_features(self, example_batch, indices=None):
        texts = example_batch["tokens"]
        
        if "augment" in example_batch:
            features = self.tokenizer.batch_encode_plus(
                texts, max_length=self.max_seq_length, 
                padding="max_length", 
                truncation=True,
                is_split_into_words=True
            )
            
            augmented = example_batch["augment"]
            aug_features = self.tokenizer.batch_encode_plus(
                augmented, max_length=self.max_seq_length, 
                padding="max_length", 
                truncation=True,
                is_split_into_words=True
            )
            for key in aug_features:
                features["aug_"+key] = aug_features[key]
                
        else: 
            features = self.tokenizer.batch_encode_plus(
                texts,
                padding=True,
                truncation=True,
                is_split_into_words=True
            )

        features["word_ids"] = [list(map(lambda x: -1 if x is None else x, features.word_ids(idx))) for idx in range(len(example_batch["unique_id"]))]
        if "ner_ids" in example_batch:
            features["labels"] = [[self.get_label_idx(label,i) for i in features.word_ids(idx)] for idx, label in enumerate(example_batch["ner_ids"])]
        return features

# %%
def compute_seqeval_jsonl(labels, prediction, mode="train"):
    report = classification_report(y_true=labels, y_pred=prediction, 
                                   scheme=IOB2, output_dict=True)
    
    # extract values we care about
    report.pop("macro avg")
    report.pop("weighted avg")
    overall_score = report.pop("micro avg")

    seqeval_results = {
        type_name: {
            f"{mode}_precision": score["precision"],
            f"{mode}_recall": score["recall"],
            f"{mode}_f1": score["f1-score"],
            f"{mode}_suport": score["support"],
        }
        for type_name, score in report.items()
    }
    seqeval_results[f"{mode}_precision"] = overall_score["precision"]
    seqeval_results[f"{mode}_recall"] = overall_score["recall"]
    seqeval_results[f"{mode}_f1"] = overall_score["f1-score"]
    seqeval_results[f"{mode}_accuracy"] = accuracy_score(y_true=labels, y_pred=prediction) 
    
    labels = np.concatenate(np.array(labels))  
    prediction = np.concatenate(np.array(prediction))

    mcc = matthews_corrcoef(y_true=labels, y_pred=prediction)   
    
    return seqeval_results, mcc

# %%
class DEALTransformer(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        learning_rate: float = 3e-4,
        warmup_steps: int = 0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.model = AutoAdapterModel.from_pretrained(model_name_or_path)
        self.model.add_tagging_head("ner", num_labels=num_labels)
        self.model.add_adapter("ner")
        self.model.train_adapter("ner")
        self.model.set_active_adapters("ner")

    def forward(self, **inputs):
        return self.model(**inputs)
    
    def on_train_start(self) -> None:
        self.cur_step = 0

    def training_step(self, batch, batch_idx):
        outputs = self(input_ids=batch["input_ids"], 
                       attention_mask=batch["attention_mask"], 
                       labels=batch["labels"])
        
        cross_entropy_loss = outputs.loss
        
        logits = outputs.logits
        
        probability = F.softmax(logits.view(-1, self.hparams.num_labels))

        self.cur_step += 1

        augmented = self(input_ids=batch["aug_input_ids"], 
                          attention_mask=batch["aug_attention_mask"])
        
        aug_logits = augmented.logits

        aug_prob = F.log_softmax(aug_logits.view(-1, self.hparams.num_labels))
        
        consistency_loss = F.kl_div(aug_prob, 
                                    probability.detach(),
                                    reduction="batchmean")
        
        loss = cross_entropy_loss + consistency_loss

        preds = torch.argmax(logits, axis=-1)

        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, axis=-1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        preds = preds.detach().cpu().numpy().tolist()
        labels = batch["labels"].detach().cpu().numpy().tolist()
        word_ids = batch["word_ids"].detach().cpu().numpy().tolist()
        pred_result = []
        ref_result = []
        for pred_item, label_item, word_id_item in zip(preds, labels, word_ids):
            counter = 0
            label_list = []
            pred_list = []
            for pred, label, word_id in zip(pred_item, label_item, word_id_item):
                if word_id == counter:
                    label_list.append(ner_tags[label])
                    pred_list.append(ner_tags[pred])
                    counter += 1
            ref_result.append(label_list)
            pred_result.append(pred_list)
        seq_eval, mcc = compute_seqeval_jsonl(ref_result, pred_result, "train")
        self.log("train_cross_entropy_loss", cross_entropy_loss)
        self.log("train_consistency_loss", consistency_loss)
        self.log("train_loss", loss)
        self.log_dict(seq_eval)
        self.log("train_mcc", mcc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(input_ids = batch["input_ids"], 
                       attention_mask=batch["attention_mask"], 
                       labels=batch["labels"])
        val_loss = outputs.loss
        logits = outputs.logits

        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, axis=-1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        return {"loss": val_loss, "preds": preds, "labels": batch["labels"], 
                "word_ids": batch["word_ids"]}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy().tolist()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy().tolist()
        word_ids = torch.cat([x["word_ids"] for x in outputs]).detach().cpu().numpy().tolist()
        pred_result = []
        ref_result = []
        for pred_item, label_item, word_id_item in zip(preds, labels, word_ids):
            counter = 0
            label_list = []
            pred_list = []
            for pred, label, word_id in zip(pred_item, label_item, word_id_item):
                if word_id == counter:
                    label_list.append(ner_tags[label])
                    pred_list.append(ner_tags[pred])
                    counter += 1
            ref_result.append(label_list)
            pred_result.append(pred_list)

        loss = torch.stack([x["loss"] for x in outputs]).mean()
        seq_eval, mcc = compute_seqeval_jsonl(ref_result, pred_result, "val")
        self.log("val_loss",loss)
        self.log_dict(seq_eval)
        self.log("val_mcc", mcc)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(input_ids = batch["input_ids"], 
                       attention_mask=batch["attention_mask"])
        logits = outputs.logits

        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, axis=-1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        return {"preds": preds, "word_ids": batch["word_ids"]}

    
    def test_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy().tolist()
        word_ids = torch.cat([x["word_ids"] for x in outputs]).detach().cpu().numpy().tolist()
        pred_result = []
        for pred_item, word_id_item in zip(preds, word_ids):
            counter = 0
            pred_list = []
            for pred, word_id in zip(pred_item, word_id_item):
                if word_id == counter:
                    pred_list.append(ner_tags[pred])
                    counter += 1
            pred_result.append(pred_list)
        
        entry_list = {}
        for entry, pred in zip(self.trainer.datamodule.dataset["test"], pred_result):
            if entry["unique_id"] not in entry_list:
                entry_list[entry["unique_id"]] = {}
            entry_list[entry["unique_id"]][entry["part"]] = entry
            entry_list[entry["unique_id"]][entry["part"]]["pred_ner_tags"] = pred
        
        output_list = []
        for entry in entry_list.values():
            final_entry = {}
            entry = [x[1] for x in sorted(entry.items())]
            final_entry["bibcode"] = entry[0]["bibcode"]
            final_entry["label_studio_id"] = entry[0]["label_studio_id"]
            final_entry["section"] = entry[0]["section"]
            final_entry["unique_id"] = entry[0]["unique_id"]
            final_entry["tokens"] = list(itertools.chain(*[item["tokens"] for item in entry]))
            final_entry["pred_ner_tags"] = list(itertools.chain(*[item["pred_ner_tags"] for item in entry]))
            if len(final_entry["tokens"]) > len(final_entry["pred_ner_tags"]): 
                final_entry["pred_ner_tags"].extend(["O"] * (len(final_entry["tokens"]) - len(final_entry["pred_ner_tags"])))
            assert len(final_entry["tokens"]) == len(final_entry["pred_ner_tags"])
            output_list.append(final_entry)
        with open('/kaggle/working/WIESP2022-NER-VALIDATION-sample-predictions.jsonl', 'w') as outfile:
            for output in output_list:
                json.dump(output, outfile)
                outfile.write("\n")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(input_ids = batch["input_ids"], 
                       attention_mask=batch["attention_mask"])
        logits = outputs.logits

        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, axis=-1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        return {"preds": preds, "word_ids": batch["word_ids"]}

    
    def on_predict_epoch_end(self, outputs):
        outputs = outputs[0]
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy().tolist()
        word_ids = torch.cat([x["word_ids"] for x in outputs]).detach().cpu().numpy().tolist()
        pred_result = []
        for pred_item, word_id_item in zip(preds, word_ids):
            counter = 0
            pred_list = []
            for pred, word_id in zip(pred_item, word_id_item):
                if word_id == counter:
                    pred_list.append(ner_tags[pred])
                    counter += 1
            pred_result.append(pred_list)
        
        entry_list = {}
        for entry, pred in zip(self.trainer.datamodule.dataset["pred"], pred_result):
            if entry["unique_id"] not in entry_list:
                entry_list[entry["unique_id"]] = {}
            entry_list[entry["unique_id"]][entry["part"]] = entry
            entry_list[entry["unique_id"]][entry["part"]]["pred_ner_tags"] = pred
        
        output_list = []
        for entry in entry_list.values():
            final_entry = {}
            entry = [x[1] for x in sorted(entry.items())]
            final_entry["bibcode"] = entry[0]["bibcode"]
            final_entry["label_studio_id"] = entry[0]["label_studio_id"]
            final_entry["section"] = entry[0]["section"]
            final_entry["unique_id"] = entry[0]["unique_id"]
            final_entry["tokens"] = list(itertools.chain(*[item["tokens"] for item in entry]))
            final_entry["pred_ner_tags"] = list(itertools.chain(*[item["pred_ner_tags"] for item in entry]))
            if len(final_entry["tokens"]) > len(final_entry["pred_ner_tags"]): 
                final_entry["pred_ner_tags"].extend(["O"] * (len(final_entry["tokens"]) - len(final_entry["pred_ner_tags"])))
            assert len(final_entry["tokens"]) == len(final_entry["pred_ner_tags"])
            output_list.append(final_entry)
        with open('/kaggle/working/WIESP2022-NER-TESTING-sample-predictions.jsonl', 'w') as outfile:
            for output in output_list:
                json.dump(output, outfile)
                outfile.write("\n")

    def setup(self, stage=None) -> None:
        if stage == "fit":
            train_loader = self.trainer.datamodule.train_dataloader()

            # Calculate total steps
            tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
            ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
            self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.learning_rate)
        return {"optimizer": optimizer,
                "lr_scheduler": get_linear_schedule_with_warmup(optimizer,
                                                                num_warmup_steps=0,
                                                                num_training_steps=self.total_steps)}

# %%
data = DEALDataModule(model_name_or_path="microsoft/deberta-v3-large", max_seq_length = 384, train_batch_size = 4, eval_batch_size = 4)
model = DEALTransformer(model_name_or_path="microsoft/deberta-v3-large", num_labels=len(ner_tags), train_batch_size = 4, eval_batch_size = 4)

# %%
warnings.filterwarnings("ignore")
early_stop_callback = EarlyStopping(monitor="val_f1", min_delta=0.00, patience=2, verbose=False, mode="max")
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val_f1",
    mode="max",
    dirpath='/kaggle/working',
    filename="wiesp-deal-deberta-scibert-swa",
)
swa_callback = StochasticWeightAveraging(3e-4, annealing_epochs=1, device=None)
tb_logger = pl_loggers.TensorBoardLogger(save_dir="/kaggle/working/logs")
trainer = Trainer(accelerator="auto", accumulate_grad_batches=4, precision=16, callbacks=[early_stop_callback, checkpoint_callback, swa_callback], max_epochs=5)

# %%
trainer.fit(model, data)

# %%
trainer.test(model, data, ckpt_path="best")

# %%
trainer.predict(model, data, ckpt_path="best")

# %%



