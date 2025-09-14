import argparse
import csv
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.preprocessing import LabelEncoder
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

# Minimal, standalone implementation of Curriculum Spotter for text classification.
# Dependencies: numpy, torch, scikit-learn, scipy, transformers.


RANDOM_STATE = 13


@dataclass
class CSConfig:
    model_name: str = "bert-base-uncased"
    max_epochs: int = 8
    batch_size: int = 16
    eval_batch_size: int = 64
    verbose: bool = True
    cache_dir: str | None = None


class CurriculumDataset(Dataset):
    def __init__(self, tokenized: Dict[str, np.ndarray], labels: Sequence[int]):
        self.tokenized = tokenized
        self.labels = list(labels)
        # mapping from new index -> original index; allows dynamic shrinking/growing
        self._mapping: Dict[int, int] = {i: i for i in range(len(self.labels))}

    def __len__(self) -> int:
        return len(self._mapping)

    def __getitem__(self, idx: int):
        orig = self._mapping[idx]
        item = {
            k: torch.as_tensor(v[orig], dtype=torch.long).clone().detach() for k, v in self.tokenized.items()
        }
        item["labels"] = torch.as_tensor(self.labels[orig], dtype=torch.long).clone().detach()
        return item

    @property
    def true_len(self) -> int:
        return len(self.labels)

    def update_mapping(self, mask: np.ndarray):
        # mask is a boolean array indicating membership in the next epoch's dataset
        mapping = {}
        c = 0
        for i, keep in enumerate(mask):
            if bool(keep):
                mapping[c] = i
                c += 1
        self._mapping = mapping


class CSCallback(TrainerCallback):
    def __init__(self, num_labels: int, tokenizer: AutoTokenizer):
        self.scores = None
        self._num_labels = num_labels
        self._tokenizer = tokenizer

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        train_dl: DataLoader = kwargs["train_dataloader"]
        ds: CurriculumDataset = train_dl.dataset
        self.scores = np.zeros(ds.true_len, dtype=float)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        train_dl: DataLoader = kwargs["train_dataloader"]
        logging.info(f"[CS] dataset size at epoch start: {len(train_dl.dataset)}")

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model: PreTrainedModel = kwargs["model"]
        train_dl: DataLoader = kwargs["train_dataloader"]
        ds: CurriculumDataset = train_dl.dataset

        preds, losses = _predict_and_losses(model, ds, args.per_device_eval_batch_size)

        lambda_ = _lambda_for_correct(ds, preds, losses)

        easy_mask = losses <= lambda_
        # sample delta percent (linearly increasing with epoch) of the easiest among hard ones
        delta = float(int(state.epoch) / max(1, int(state.num_train_epochs)))
        n = len(losses)
        k = int(n * delta)
        hard_mask = losses > lambda_
        hard_indices_sorted = [i for i in np.argsort(losses) if hard_mask[i]][:k]
        next_mask = easy_mask.copy()
        next_mask[hard_indices_sorted] = True

        # Update dataset mapping and accumulate scores for hard examples
        ds.update_mapping(next_mask)

        num_hard = max(1, int(hard_mask.sum()))
        self.scores += (hard_mask * (losses + 1.0 / num_hard))

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model: PreTrainedModel = kwargs["model"]
        train_dl: DataLoader = kwargs["train_dataloader"]
        ds: CurriculumDataset = train_dl.dataset
        _, losses = _predict_and_losses(model, ds, args.per_device_eval_batch_size)
        # break ties for never-hard by adding loss
        self.scores += (self.scores == 0) * losses


def _predict_and_losses(model: PreTrainedModel, ds: CurriculumDataset, eval_bs: int) -> Tuple[np.ndarray, np.ndarray]:
    with tempfile.TemporaryDirectory() as tmpdir:
        targs = TrainingArguments(
            output_dir=tmpdir,
            per_device_eval_batch_size=eval_bs,
            seed=RANDOM_STATE,
            disable_tqdm=True,
            report_to="none",
        )
        trainer = Trainer(model=model, args=targs)
        out = trainer.predict(ds)

    probas = softmax(out.predictions, axis=1)
    preds = np.argmax(probas, axis=1)

    logits = torch.from_numpy(out.predictions)
    labels = torch.as_tensor([ds.labels[ds._mapping[i]] for i in range(len(ds))], dtype=torch.long)
    loss_fct = CrossEntropyLoss(reduction="none")
    losses = loss_fct(logits.view(-1, model.num_labels), labels.view(-1)).cpu().numpy()

    return preds, losses


def _lambda_for_correct(y_gold: np.ndarray, y_pred: np.ndarray, losses: np.ndarray) -> float:
    correct = y_gold == y_pred
    return float(losses[correct].mean()) if correct.any() else float(losses.mean())


def curriculum_spotter_scores(texts: Sequence[str], labels: Sequence[str], cfg: CSConfig = CSConfig()) -> np.ndarray:
    # Use an explicit cache directory if provided to persist downloads
    tok_kwargs = {"cache_dir": cfg.cache_dir} if cfg.cache_dir else {}
    mdl_kwargs = {"cache_dir": cfg.cache_dir} if cfg.cache_dir else {}
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, **tok_kwargs)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(list(labels))

    tokenized = tokenizer(list(texts), truncation=True, padding=True)

    # Model
    config = AutoConfig.from_pretrained(
        cfg.model_name,
        num_labels=len(le.classes_),
        classifier_dropout=0.25,
        **mdl_kwargs,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        config=config,
        **mdl_kwargs,
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # Dataset
    ds = CurriculumDataset({k: np.array(v) for k, v in tokenized.items()}, y)

    # Training
    with tempfile.TemporaryDirectory() as tmp:
        targs = TrainingArguments(
            output_dir=tmp,
            num_train_epochs=cfg.max_epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.eval_batch_size,
            
            save_strategy="no",
            logging_strategy="epoch" if cfg.verbose else "no",
            disable_tqdm=not cfg.verbose,
            seed=RANDOM_STATE,
            report_to="none",
        )
        callback = CSCallback(num_labels=len(le.classes_), tokenizer=tokenizer)
        trainer = Trainer(model=model, args=targs, train_dataset=ds, eval_dataset=ds, callbacks=[callback])
        trainer.train()
        scores = callback.scores
    return scores


def _read_table(path: str, text_col: str, label_col: str, delimiter: str) -> Tuple[List[str], List[str]]:
    texts, labels = [], []
    df = pd.read_csv(path, delimiter=delimiter, encoding='latin-1')
    for _, row in df.iterrows():
        texts.append(row[text_col])
        labels.append(row[label_col])
    return texts, labels


def main():
    ap = argparse.ArgumentParser(description="Minimal Curriculum Spotter for text classification")
    ap.add_argument("data", help="Path to CSV/TSV with headers")
    ap.add_argument("--text-col", default="text", help="Text column name")
    ap.add_argument("--label-col", default="label", help="Label column name")
    ap.add_argument("--tsv", action="store_true", help="Input is TSV (default CSV)")
    ap.add_argument("--model", default="bert-base-uncased", help="HF model name or path")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE"),
        help="Directory to cache models/tokenizers (mount a host volume)",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if not args.quiet else logging.ERROR)

    texts, labels = _read_table(args.data, args.text_col, args.label_col, "\t" if args.tsv else ",")
    # If a cache dir is provided, normalize env so both Transformers and HF Hub use it
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = args.cache_dir
        os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(args.cache_dir, "transformers"))

    cfg = CSConfig(
        model_name=args.model,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        verbose=not args.quiet,
        cache_dir=args.cache_dir,
    )
    scores = curriculum_spotter_scores(texts, labels, cfg)
    for s in scores:
        print(float(s))


if __name__ == "__main__":
    main()
