from transformers import VisionEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
from PIL import Image


def get_model(image_encoder_model, text_decoder_model, saved_model_path=None):
    if saved_model_path is not None:
        model = VisionEncoderDecoderModel.from_pretrained(saved_model_path)
    else:
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            image_encoder_model, text_decoder_model
        )

    feature_extractor = AutoFeatureExtractor.from_pretrained(image_encoder_model)
    tokenizer = AutoTokenizer.from_pretrained(text_decoder_model)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, feature_extractor, tokenizer
