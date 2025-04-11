from bs4 import BeautifulSoup
from datasets import DatasetDict, Dataset

from sentence_transformers.cross_encoder import losses


from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments


from sentence_transformers.training_args import BatchSamplers
from transformers.utils import logging
from sklearn.model_selection import train_test_split
import pandas as pd
import multiprocessing
import warnings

logging.set_verbosity_error()
warnings.filterwarnings("ignore")
SEED = 42
num_workers = multiprocessing.cpu_count()

df = pd.read_csv("train.csv")[["anchor","positive"]]

df = df.dropna(subset=["anchor", "positive"])


train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
dataset_dict = DatasetDict({"train": Dataset.from_pandas(train_df), "valid": Dataset.from_pandas(valid_df)})

columns_to_remove = [col for col in dataset_dict["train"].column_names if col.startswith("__")]

dataset_dict = DatasetDict({
    split: ds.remove_columns(columns_to_remove)
    for split, ds in dataset_dict.items()
})

print(dataset_dict)

model = CrossEncoder(
    "Alibaba-NLP/gte-reranker-modernbert-base",
    trust_remote_code=True
)
loss = losses.CachedMultipleNegativesRankingLoss(model)

steps_per_epoch = len(dataset_dict["train"]) // 16

args = CrossEncoderTrainingArguments(
    output_dir=f"checkpoints/cross-encoder",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    learning_rate=4e-6,
    warmup_ratio=0.15,
    weight_decay=0.2,
    lr_scheduler_type='cosine',
    max_grad_norm=4.7,
    bf16=True,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps= steps_per_epoch // 2,
    eval_steps= steps_per_epoch // 2,
    save_total_limit=2,
    logging_steps=50,
    run_name="crossencoder",
    seed=SEED,
    logging_dir="./runs/crossencoder",
    report_to=["tensorboard"],
    disable_tqdm=False,
    load_best_model_at_end=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES  # IMPORTANT
)

trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["valid"],
    loss=loss
)

trainer.train()
trainer.save_model(f"crossencoder")
