import json
import logging
import sys

import datasets
import evaluate
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)

import utils

logger = logging.getLogger(__name__)


def main():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = HfArgumentParser(
        (
            utils.ModelArguments,
            utils.DataTrainingArguments,
            TrainingArguments,
            utils.FineTuningArguments,
        )
    )

    model_args, data_args, training_args, ft_args = parser.parse_args_into_dataclasses()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    train_df = pd.read_csv("train_1.tsv", sep="\t")
    val_df = pd.read_csv("val_1.tsv", sep="\t")

    # Loading trustpilot files
    raw_datasets = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_pandas(train_df),
            "validation": datasets.Dataset.from_pandas(val_df),
        }
    )
    col_to_remove = list()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.num_labels = 2
    config.label2id = {"sexist": 1, "not sexist": 0}
    config.id2label = {v: k for k, v in config.label2id.items()}

    def tokenize_function(examples):
        # Remove empty lines
        examples[ft_args.text_column_name] = [
            line for line in examples["text"] if len(line) > 0 and not line.isspace()
        ]

        item = tokenizer(
            examples["text"],
            max_length=data_args.max_seq_length,
            padding=False,
            truncation=True,
            return_special_tokens_mask=True,
        )

        # We predict the sentiment
        item["labels"] = [config.label2id[l] for l in examples[ft_args.target]]

        return item

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=col_to_remove,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset line_by_line",
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    data_collator = DataCollatorWithPadding(
        tokenizer, max_length=data_args.max_seq_length, pad_to_multiple_of=True
    )

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("f1")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        preds = preds[mask]
        return metric.compute(predictions=preds, references=labels, average="macro")

    ## HPARAM OPTIM
    def optuna_hp_space(trial):
        reg_params = dict()
        if ft_args.regularization == "ear":
            reg_params["reg_strength"] = trial.suggest_float(
                "reg_strength", 0.0001, 1, log=True
            )
        if ft_args.regularization == "r3f":
            reg_params["r3f_lambda"] = trial.suggest_float("r3f_lambda", 1e-2, 10)

        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0001, 0.1, log=True),
            **reg_params,
        }

    def model_init(trial):
        return AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        )

    callbacks = [EarlyStoppingCallback(ft_args.patience, 1e-5)]

    if ft_args.hparam_search:

        import torch.nn.functional as F
        from torch import nn

        import ear

        def get_symm_kl(noised_logits, input_logits):
            return (
                F.kl_div(
                    F.log_softmax(noised_logits, dim=-1, dtype=torch.float32),
                    F.softmax(input_logits, dim=-1, dtype=torch.float32),
                    None,
                    None,
                    "sum",
                )
                + F.kl_div(
                    F.log_softmax(input_logits, dim=-1, dtype=torch.float32),
                    F.softmax(noised_logits, dim=-1, dtype=torch.float32),
                    None,
                    None,
                    "sum",
                )
            ) / noised_logits.size(0)

        class CustomTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):

                # R3F needs to be ran single-gpu
                labels = inputs.get("labels")
                input_ids = inputs.pop("input_ids")
                input_embeds = model.get_input_embeddings()(input_ids)
                inputs["inputs_embeds"] = input_embeds

                # forward pass
                outputs = model(**inputs, output_attentions=True)
                logits = outputs.get("logits")
                # compute custom loss (suppose one has 3 labels with different weights)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.model.config.num_labels), labels.view(-1)
                )

                trial = getattr(self, "_trial", None)
                if trial is None:
                    raise RuntimeError("Trial None")

                if ft_args.regularization == "ear":
                    reg_strength = trial.params["reg_strength"]
                    neg_entropy = ear.compute_negative_entropy(
                        inputs=outputs.attentions,
                        attention_mask=inputs["attention_mask"],
                    )
                    reg_loss = reg_strength * neg_entropy
                    loss = reg_loss + loss

                if ft_args.regularization == "r3f":

                    noise_sampler = torch.distributions.uniform.Uniform(
                        low=-1e-5, high=1e-5
                    )
                    noise = noise_sampler.sample(sample_shape=input_embeds.shape).to(
                        model.device
                    )
                    noised_embeddings = input_embeds.detach().clone() + noise
                    noised_logits = model(
                        inputs_embeds=noised_embeddings,
                        attention_mask=inputs["attention_mask"],
                    ).logits
                    symm_kl = get_symm_kl(noised_logits, logits)

                    # sample_size = batch["labels"].numel()
                    # symm_kl = symm_kl * sample_size
                    r3f_loss = trial.params["r3f_lambda"] * symm_kl

                    loss += r3f_loss

                return (loss, outputs) if return_outputs else loss

        trainer = CustomTrainer(
            model=None,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            data_collator=data_collator,
            callbacks=callbacks,
            compute_metrics=compute_metrics
            if training_args.do_eval and not is_torch_tpu_available()
            else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
            if training_args.do_eval and not is_torch_tpu_available()
            else None,
        )

        best_trial = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=optuna_hp_space,
            n_trials=ft_args.hparam_trials,
        )

        with open(f"{training_args.output_dir}/best_trial.txt", "w") as fp:
            fp.write(str(best_trial))

        # run training again with the best trial
        logger.info("Training with best hparam found!")
        del trainer

        training_args.learning_rate = best_trial.hyperparameters["learning_rate"]
        training_args.weight_decay = best_trial.hyperparameters["weight_decay"]
        training_args.load_best_model_at_end = True
        training_args.report_to = ["wandb"]

        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path, config=config
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            data_collator=data_collator,
            callbacks=callbacks,
            compute_metrics=compute_metrics
            if training_args.do_eval and not is_torch_tpu_available()
            else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
            if training_args.do_eval and not is_torch_tpu_available()
            else None,
        )

        train_result = trainer.train()
        trainer.save_model()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # test_result = trainer.predict(tokenized_datasets["test"])
        # trainer.log_metrics("test", test_result.metrics)
        # trainer.save_metrics("test", test_result.metrics)

        # import numpy as np

        # np.save(
        #     f"{training_args.output_dir}/custom_predictions.npy",
        #     test_result.predictions,
        # )

    else:
        raise NotImplementedError()


def run_baseline(name, train_df, val_df, test_df, ft_args, data_args, training_args):
    import json

    import optuna
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    if name == "lr":

        # 1. Define an objective function to be maximized.
        def objective(trial):
            max_df = trial.suggest_float("max_df", 0.8, 1, step=0.02)
            min_df = trial.suggest_float("min_df", 0, 0.2, step=0.02)
            max_features = trial.suggest_categorical(
                "max_features", [None, 5000, 10000]
            )

            tfidf = TfidfVectorizer(
                stop_words="english",
                max_df=max_df,
                min_df=min_df,
                max_features=max_features,
            )

            try:
                X_train = tfidf.fit_transform(train_df["text"])
                X_test = tfidf.transform(val_df["text"])
            except:
                return 0.0

            # 2. Suggest values for the hyperparameters using a trial object.
            # classifier = trial.suggest_categorical("classifier", ["lr", "gb"])
            classifier = "lr"

            if classifier == "lr":
                clf = LogisticRegression(
                    penalty=trial.suggest_categorical("penalty", ["none", "l2"]),
                    C=trial.suggest_float("C", 0.1, 10, log=True),
                    class_weight=trial.suggest_categorical(
                        "class_weight", [None, "balanced"]
                    ),
                    max_iter=500,
                    n_jobs=6,
                )
            else:
                learning_rate = trial.suggest_float("lr", 1e-3, 1, log=True)
                max_depth = trial.suggest_int("max_depth", 1, 7, step=1)
                clf = HistGradientBoostingClassifier(
                    learning_rate=learning_rate, max_depth=max_depth, n_iter_no_change=5
                )
                X_train = X_train.toarray()
                X_test = X_test.toarray()

            clf = clf.fit(X_train, train_df[ft_args.target])
            y_pred = clf.predict(X_test)
            f1_macro = f1_score(val_df[ft_args.target], y_pred, average="macro")

            return f1_macro

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100)

        logger.info("Best params found")
        logger.info(study.best_params)

        best_params = study.best_params
        clf = LogisticRegression(
            penalty=best_params["penalty"],
            C=best_params["C"],
            class_weight=best_params["class_weight"],
            max_iter=500,
            n_jobs=6,
        )


        vect = TfidfVectorizer(
            max_df=best_params["max_df"],
            min_df=best_params["min_df"],
            max_features=best_params["max_features"],
        )

        X_train = vect.fit_transform(train_df["text"])
        X_test = vect.transform(test_df["text"])
        clf = clf.fit(X_train, train_df[ft_args.target])
        y_pred = clf.predict(X_test)

    elif name == "dummy":
        tfidf = TfidfVectorizer()
        X_train = tfidf.fit_transform(train_df["text"])
        y_train = train_df[ft_args.target]
        X_test = tfidf.transform(test_df["text"])

        clf = DummyClassifier(strategy="stratified")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

    elif name == "sentence_embeddings":
        from sentence_transformers import SentenceTransformer

        logger.info("Running baseline model with Sentence Embeddings")

        se_model = "all-mpnet-base-v2"
        embedder = SentenceTransformer(se_model)
        X_train = embedder.encode(train_df["text"].tolist(), show_progress_bar=True)
        X_val = embedder.encode(val_df["text"].tolist(), show_progress_bar=True)
        X_test = embedder.encode(test_df["text"].tolist(), show_progress_bar=True)

        def objective(trial):

            clf = LogisticRegression(
                penalty=trial.suggest_categorical("penalty", ["none", "l2"]),
                C=trial.suggest_float("C", 0.1, 10, log=True),
                class_weight=trial.suggest_categorical(
                    "class_weight", [None, "balanced"]
                ),
                max_iter=500,
                n_jobs=6,
            )
            clf = clf.fit(X_train, train_df[ft_args.target])
            y_pred = clf.predict(X_val)
            f1_macro = f1_score(val_df[ft_args.target], y_pred, average="macro")

            return f1_macro

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)

        logger.info("Best params found")
        logger.info(study.best_params)

        best_params = study.best_params
        clf = LogisticRegression(
            penalty=best_params["penalty"],
            C=best_params["C"],
            class_weight=best_params["class_weight"],
            max_iter=500,
            n_jobs=6,
        )

        clf = clf.fit(X_train, train_df[ft_args.target])
        y_pred = clf.predict(X_test)

    results = {
        "baseline": name,
        "f1_macro": f1_score(test_df[ft_args.target], y_pred, average="macro"),
    }

    with open(f"{training_args.output_dir}/baseline_{name}.json", "w") as fp:
        json.dump(fp, results)


if __name__ == "__main__":
    main()
