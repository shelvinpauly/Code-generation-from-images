from transformers import VisionEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig, ResNetConfig, ResNetModel, BertGenerationDecoder, BertGenerationConfig
from transformers.modeling_outputs import BaseModelOutput
from PIL import Image
import torch

def get_model(image_encoder_model, text_decoder_model, saved_model_path=None):
    if saved_model_path is not None:
        model = VisionEncoderDecoderModel.from_pretrained(saved_model_path)
    else:
        # TODO put back in, although this needs to be rewritten for resnet encoder (CNNs encoders aren't supported out-of-box)

        if image_encoder_model == 'timber/resnet' and text_decoder_model == 'timber/bert-decoder':
            image_encoder_model = 'microsoft/resnet-50'
            text_decoder_model = 'facebook/bart-base'

            encoder_config = ResNetConfig(hidden_sizes=[64, 128, 256, 512], layer_type='basic')

            class TimberResNet(ResNetModel):
                def __init__(self, config):
                    super(TimberResNet, self).__init__(config)
                    self.config.hidden_size = config.hidden_sizes[3]*7*7 # this is a 224x224 image run through the model and flattened
                
                def forward(self, *args, **kwargs):
                    if 'output_attentions' in kwargs:
                        del kwargs['output_attentions']
                    output = super(TimberResNet, self).forward(*args, **kwargs)
                    output_dict = dict(output)
                    output_dict['last_hidden_state'] = torch.flatten(output_dict['last_hidden_state'], start_dim=1)
                    output_dict['attentions'] = None
                    del output_dict['pooler_output']
                    return BaseModelOutput(**output_dict)

            encoder = TimberResNet(encoder_config)

            decoder_config = AutoConfig.from_pretrained(text_decoder_model)
            decoder_config.num_hidden_layers = 3
            decoder_config.decoder_attention_heads = 4
            decoder_config.encoder_attention_heads = 4
            decoder_config.d_model = 256
            decoder_config.d_model = 256

            decoder = AutoModelForCausalLM.from_config(decoder_config)
            # decoder_config = BertGenerationConfig(
                # is_decoder=True
            # )
            # decoder = BertGenerationDecoder(decoder_config)
            model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
        else:
            model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
                image_encoder_model, text_decoder_model
            )

    feature_extractor = AutoFeatureExtractor.from_pretrained(image_encoder_model)
    tokenizer = AutoTokenizer.from_pretrained(text_decoder_model)
    # tokenizer.pad_token = tokenizer.eos_token
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, feature_extractor, tokenizer
