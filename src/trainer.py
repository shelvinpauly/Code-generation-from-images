from dataloader import get_train_val_test_datasets_processed
from models import get_model
import argparse
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator
import wandb

WANDB_PROJECT = "Baseline"
WANDB_ENTITY = ""


def main(args):
    model, feature_extractor, tokenizer = get_model(
        args.image_encoder_model,
        args.text_decoder_model,
        args.max_length,
        args.saved_model_path,
    )
    processed_dataset = get_train_val_test_datasets_processed(
        model.tokenizer,
        model.feature_extractor,
        model.model,
        args.max_length,
        debug_amount=args.debug_amount,
    )
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        output_dir="./code_generation",
        report_to="wandb",
        num_train_epochs=args.epochs,
        save_strategy="epoch",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=feature_extractor,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        data_collator=default_data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_encoder_model", type=str, required=True, help="Image encoder model"
    )
    parser.add_argument(
        "--text_decoder_model", type=str, required=True, help="Text decoder model"
    )
    parser.add_argument(
        "--max_length", type=int, required=True, help="Maximum length for input"
    )
    parser.add_argument(
        "--saved_model_path",
        type=str,
        default=None,
        help="Path to saved model (optional)",
    )
    parser.add_argument(
        "--debug_amount",
        type=int,
        default=None,
        help="Amount of data for debugging (optional)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to train for"
    )
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Batch size")

    args = parser.parse_args()

    main(args)
