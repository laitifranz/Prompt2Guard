import torch
import torch.nn as nn

from models.clip import clip
from models.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.backbonename
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class PromptLearner(nn.Module):
    def __init__(self, cfg, clip_model, k):
        super().__init__()
        positional_embedding = clip_model.positional_embedding

        assert k >= 1, "K should be bigger than 0"

        self.K = k  # the number of prompt pair
        self.dtype = clip_model.dtype
        self.d_t = clip_model.ln_final.weight.shape[0]  # 512
        self.d_v = 768

        clip_imsize = clip_model.visual.input_resolution  # 224
        cfg_imsize = cfg.INPUTSIZE[0]  # (224, 224)[0]
        assert (
            cfg_imsize == clip_imsize
        ), f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self.initialization_token(clip_model)

    def initialization_token(self, clip_model):
        # text token initialization
        text_token = clip_model.token_embedding(torch.tensor([49407]))
        text_token = text_token.repeat(self.K, 1)
        text_noise = torch.randn(self.K, self.d_t)
        text_noise = text_noise / text_noise.norm(dim=-1, keepdim=True)
        text_token += 0.1 * text_noise
        text_token = text_token.type(self.dtype)
        self.text_prompt = nn.Parameter(text_token)

        # visual token initialization
        visual_token = clip_model.visual.class_embedding
        visual_token = visual_token.repeat(self.K, 1)
        visual_noise = torch.randn(self.K, self.d_v)
        visual_noise = visual_noise / visual_noise.norm(dim=-1, keepdim=True)
        visual_token += 0.1 * visual_noise
        visual_token = visual_token.type(self.dtype)
        self.img_prompt = nn.Parameter(visual_token)

    def forward(self):
        return self.text_prompt, self.img_prompt


class cfgc(object):
    backbonename = "ViT-B/16"
    INPUTSIZE = (224, 224)
