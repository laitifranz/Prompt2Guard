import torch
import torch.nn as nn
import copy
from einops import rearrange, reduce

from models.clip import clip
from models.clip.prompt_learner import cfgc, load_clip_to_cpu, PromptLearner
from utils.class_names import cddb_classnames

import logging


class SliNet(nn.Module):

    def __init__(self, args):
        super(SliNet, self).__init__()
        self.args = args
        self.cfg = cfgc()
        self.logging_cfg()

        # Load and configure CLIP model
        clip_model = load_clip_to_cpu(self.cfg)
        if args["precision"] == "fp32":
            clip_model.float()
        self.clip_model = clip_model

        # Set general parameters
        self.K = args["K"]
        self.device = "cuda"
        self.topk_classes = args["topk_classes"]

        # Set ensembling parameters for object classes, not the prediction ensembling (for that see the evaluation part)
        if self.topk_classes > 1:
            (
                self.ensemble_token_embedding,
                self.ensemble_before_cosine_sim,
                self.ensemble_after_cosine_sim,
                self.confidence_score_enable,
            ) = args["ensembling"]
        else:
            self.ensemble_token_embedding = self.ensemble_before_cosine_sim = self.ensemble_after_cosine_sim = self.confidence_score_enable = False

        # Set text encoder components
        self.token_embedding = clip_model.token_embedding
        self.text_pos_embedding = clip_model.positional_embedding
        self.text_transformers = clip_model.transformer
        self.text_ln_final = clip_model.ln_final
        self.text_proj = clip_model.text_projection

        # Set vision encoder components
        self.img_patch_embedding = clip_model.visual.conv1
        self.img_cls_embedding = clip_model.visual.class_embedding
        self.img_pos_embedding = clip_model.visual.positional_embedding
        self.img_pre_ln = clip_model.visual.ln_pre
        self.img_transformer = clip_model.visual.transformer
        self.img_post_ln = clip_model.visual.ln_post
        self.img_proj = clip_model.visual.proj

        # Set logit and dtype
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # Set continual learning parameters
        self.class_num = 1
        self.numtask = 0

        # Set up prompt learner and masks
        self.prompt_learner = nn.ModuleList()
        if args["dataset"] == "cddb":
            for i in range(len(args["task_name"])):
                self.prompt_learner.append(PromptLearner(self.cfg, clip_model, self.K))
            self.make_prompts(
                [
                    "a photo of a _ image.".replace("_", c)
                    for c in list(cddb_classnames.values())
                ]
            )
            self.class_num = 2
        else:
            raise ValueError("Unknown datasets: {}.".format(args["dataset"]))
        self.define_mask()

    def make_prompts(self, prompts):
        with torch.no_grad():
            self.text_tokenized = torch.cat([clip.tokenize(p) for p in prompts]).to(
                next(self.clip_model.parameters()).device
            )  # CLIP on CPU at the beginning, after in GPU
            self.text_x = self.token_embedding(self.text_tokenized).type(
                self.dtype
            ) + self.text_pos_embedding.type(self.dtype)
            self.len_prompts = self.text_tokenized.argmax(dim=-1) + 1

    def define_mask(self):
        len_max = 77
        attn_head = 8

        # text encoder mask
        num_masks = len(self.len_prompts) * attn_head
        text_mask = torch.full((num_masks, len_max, len_max), float("-inf"))

        for i, idx in enumerate(self.len_prompts):
            mask = torch.full((len_max, len_max), float("-inf"))
            mask.triu_(1)  # zero out the lower diagonal
            mask[:, idx:].fill_(float("-inf"))
            text_mask[i * attn_head : (i + 1) * attn_head] = mask

        self.text_mask = text_mask

        # image encoder mask
        att_size = 1 + 14 * 14 + self.K
        visual_mask = torch.zeros((att_size, att_size), dtype=self.dtype, requires_grad=False)
        visual_mask[:, -1 * self.K :] = float("-inf")
        self.visual_mask = visual_mask

    def get_none_attn_mask(self, att_size: int):  # correspond to a None attn_mask
        return torch.zeros((att_size, att_size), dtype=self.dtype, requires_grad=False)

    @property
    def feature_dim(self):
        return self.clip_model.visual.output_dim

    def extract_vector(self, image):
        # only image without prompts
        image_features = self.clip_model.visual(
            image.type(self.dtype), self.get_none_attn_mask(att_size=1 + 14 * 14)
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features

    def generate_prompts_from_input(self, object_labels):
        assert self.topk_classes <= 5  # maximum topk values from CLIP Zeroshot, hardcoded value based on our initial settings
        labels, scores = zip(*object_labels)
        labels_by_position_lists = [
            list(group) for group in zip(*labels[: self.topk_classes])
        ]

        if self.confidence_score_enable:
            self.score_weights_labels = (
                (torch.stack(scores[: self.topk_classes]) / 100)
                .t()
                .unsqueeze(1)
                .expand(-1, 2, -1)
                .to(self.device)
                .half()
            )
            self.score_weights_labels = (
                self.score_weights_labels
                / self.score_weights_labels.sum(dim=-1, keepdim=True)
            )  # normalize

        if self.topk_classes > 0:
            # Top1 object label to text
            if self.topk_classes == 1:
                prompts = [
                    f"a {type_image} photo of a {label[0]}."
                    for label in labels_by_position_lists
                    for type_image in cddb_classnames.values()
                ]  # * [N = B*2 = 256]
                self.make_prompts(prompts)
            # Topk object label to text
            else:
                prompts = [
                    f"a {type_image} photo of a {topk}."
                    for label in labels_by_position_lists
                    for type_image in cddb_classnames.values()
                    for topk in label
                ]
                if self.ensemble_token_embedding:
                    assert (
                        self.ensemble_before_cosine_sim == False
                        and self.ensemble_after_cosine_sim == False
                    )
                    with torch.no_grad():
                        self.text_tokenized = torch.cat(
                            [clip.tokenize(p) for p in prompts]
                        ).to(
                            next(self.clip_model.parameters()).device
                        )  # CLIP on CPU at the beginning, after in GPU
                        self.text_x = self.token_embedding(self.text_tokenized).type(
                            self.dtype
                        ) + self.text_pos_embedding.type(self.dtype)
                        self.len_prompts = torch.cat(
                            [
                                self.text_tokenized[i : i + self.topk_classes]
                                .argmax(dim=-1)
                                .max()
                                .unsqueeze(0)
                                + 1
                                for i in range(
                                    0, len(self.text_tokenized), self.topk_classes
                                )
                            ]
                        )
                    # * B = batch | L = label (real/fake) | O = object labels (topk) | M = len_max 77 | D = dimension 512 *#
                    self.text_x = rearrange(
                        self.text_x,
                        "(b l o) m d -> b l o m d",
                        b=len(labels_by_position_lists),
                        l=len(cddb_classnames.values()),
                        o=self.topk_classes,
                    )
                    self.text_x = reduce(self.text_x, "b l o m d -> b l m d", "mean")
                    self.text_x = rearrange(self.text_x, "b l m d -> (b l) m d")
                else:
                    self.make_prompts(prompts)

        # Real/fake image prompts without object labels
        else:
            # emulate top1 prompts generation, generate batch size numbers prompts * 2 (real/fake)
            prompts = [
                f"a photo of a {type_image} image."
                for i in range(len(object_labels[0][0]))
                for type_image in cddb_classnames.values()
            ]
            self.make_prompts(prompts)
        self.define_mask()

    def image_encoder(self, image, image_prompt):
        batch_size = image.shape[0]
        visual_mask = self.visual_mask

        # training and inference may have different image_prompt shape
        if image_prompt.dim() == 2:
            image_prompt = image_prompt.repeat(batch_size, 1, 1)

        # forward propagate image features with token concatenation
        image_embedding = self.img_patch_embedding(
            image.type(self.dtype)
        )  # (batch_size, h_dim, 7, 7)
        image_embedding = image_embedding.reshape(
            batch_size, image_embedding.shape[1], -1
        )
        image_embedding = image_embedding.permute(0, 2, 1)  # (batch_size, 49, h_dim)
        image_embedding = torch.cat(
            [
                self.img_cls_embedding.repeat(batch_size, 1, 1).type(self.dtype),
                image_embedding,
            ],
            dim=1,
        )  # 16 (batch_size, 50, h_dim)
        img_x = image_embedding + self.img_pos_embedding.type(self.dtype)  # (N,L,D)
        # concatenation the token on visual encoder
        img_x = torch.cat([img_x, image_prompt], dim=1)
        # image encoder
        img_x = self.img_pre_ln(img_x)
        img_x = img_x.permute(1, 0, 2)
        img_x = self.img_transformer(img_x, visual_mask)
        img_x = img_x.permute(1, 0, 2)
        img_f = self.img_post_ln(img_x[:, -1 * self.K :, :]) @ self.img_proj
        i_f = self.img_post_ln(img_x[:, 0, :]) @ self.img_proj

        """ 
            img_f: only K prompts
            i_f:   img fts without K prompts   
        """
        return img_f, i_f

    def text_encoder(self, text_prompt):
        text_x = self.text_x  # * [N, L = 77, D = 512]
        text_mask = self.text_mask  # * [N * ATTN_HEAD = 8, 77, 77]
        text_x = text_x.to(self.device)

        for i in range(self.K):
            text_x[torch.arange(text_x.shape[0]), self.len_prompts + i, :] = (
                text_prompt[i, :].repeat(text_x.shape[0], 1)
            )

        text_x = text_x.permute(1, 0, 2)  # * NLD -> LND
        text_x = self.text_transformers(text_x, text_mask)  # * [LND]
        text_x = text_x.permute(1, 0, 2)  # * [NLD]
        text_x = self.text_ln_final(text_x).type(self.dtype)

        text_f = torch.empty(
            text_x.shape[0], 0, 512, device=self.device, dtype=self.dtype
        )  # * [N0D]
        for i in range(self.K):
            idx = self.len_prompts + i
            x = text_x[torch.arange(text_x.shape[0]), idx]
            text_f = torch.cat([text_f, x[:, None, :]], dim=1)

        text_f = text_f @ self.text_proj  # * [NKD]
        t_f = None
        # t_f = text_x[torch.arange(text_x.shape[0]), self.text_tokenized.argmax(dim=-1)] @ self.text_proj  # [ND]

        if self.ensemble_before_cosine_sim:
            assert (
                self.ensemble_token_embedding == False
                and self.ensemble_after_cosine_sim == False
            )
            batch_size = self.text_x.shape[0] // (
                len(cddb_classnames.values()) * self.topk_classes
            )
            # * B = batch | L = label (real/fake) | O = object labels (topk) | K = k learnable prompts | D = dimension 512 *#
            text_f = rearrange(
                text_f,
                "(b l o) k d -> b l o k d",
                b=batch_size,
                l=len(cddb_classnames.values()),
                o=self.topk_classes,
            )
            text_f = reduce(text_f, "b l o k d -> b l k d", "mean")
            text_f = rearrange(text_f, "b l k d -> (b l) k d")

        """ 
            text_f: only K prompts
            t_f:    text fts without K prompts   
        """
        return text_f, t_f

    def forward(self, image, object_labels):
        ## * B = batch | N = B*2 = num prompts | D = text features | F = image features | P = prompt per image
        text_prompt, image_prompt = self.prompt_learner[self.numtask - 1]()  # * [KD], [KF]
        self.generate_prompts_from_input(object_labels)

        text_f, _ = self.text_encoder(text_prompt)  # * [NKD]
        img_f, _ = self.image_encoder(image, image_prompt)  # * [BKD]

        text_f = text_f / text_f.norm(dim=-1, keepdim=True)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)

        logits = self.training_cosine_similarity(text_f, img_f)

        return {"logits": logits}

    def training_cosine_similarity(self, text_f, img_f):
        if self.ensemble_after_cosine_sim:
            assert (
                self.ensemble_before_cosine_sim == False
                and self.ensemble_token_embedding == False
            )
            # * B = batch | L = label (real/fake) | O = object labels (topk) | K = k learnable prompts | D = dimension 512 *#
            text_f = rearrange(
                text_f,
                "(b l o) k d -> b l o k d",
                b=img_f.shape[0],
                l=len(cddb_classnames.values()),
                o=self.topk_classes,
            )
            logits = torch.zeros(
                img_f.shape[0], text_f.shape[1], device=self.device
            )  # * [BP]

            for i in range(self.K):
                i_img_f = img_f[:, i, :]  # * [BD]
                i_text_f = text_f[:, :, :, i, :]  # * [BLOD]
                logit = torch.einsum("bd,blod->blo", i_img_f, i_text_f)  # * [BLO]
                if self.confidence_score_enable:
                    logit = torch.einsum(
                        "blo,blo->bl", logit, self.score_weights_labels
                    )
                else:
                    logit = reduce(logit, "b l o -> b l", "mean")  # * [BL]
                logit = self.logit_scale.exp() * logit
                logits += logit
            logits /= self.K

        else:  # default case
            text_f = rearrange(
                text_f,
                "(b p) k d -> b p k d",
                b=img_f.shape[0],
                p=len(cddb_classnames.values()),
            )
            logits = torch.zeros(
                img_f.shape[0], text_f.shape[1], device=self.device
            )  # * [BP]

            for i in range(self.K):
                i_img_f = img_f[:, i, :]  # * [BD]
                i_text_f = text_f[:, :, i, :]  # * [BPD]
                logit = torch.einsum("bd,bpd->bp", i_img_f, i_text_f)  # * [BP]
                logit = self.logit_scale.exp() * logit
                logits += logit
            logits /= self.K

        return logits

    def interface(self, image, object_labels, total_tasks, keys_dict):
        ## * B = batch | N = B*2 = num prompts | D = text features | F = image features | P = prompt per image | K = k learnable prompt for each task | T = task
        self.total_tasks = total_tasks
        img_prompts = torch.cat(
            [
                learner.img_prompt
                for idx, learner in enumerate(self.prompt_learner)
                if idx < self.total_tasks
            ]
        )  # * [K*T,D]
        text_prompts = torch.cat(
            [
                learner.text_prompt
                for idx, learner in enumerate(self.prompt_learner)
                if idx < self.total_tasks
            ]
        )  # * [K*T,F]

        self.K = self.K * self.total_tasks  # make appropriate masks
        self.generate_prompts_from_input(object_labels)

        text_f, _ = self.text_encoder(text_prompts)  # * [N,K*T,D]
        img_f, i_f = self.image_encoder(image, img_prompts)  # * [B,K*T,D] , [B,D]

        prob_dist_dict = {
            "real_prob_dist": self.convert_to_prob_distribution(
                keys_dict["real_keys_one_cluster"], i_f
            ),
            "fake_prob_dist": self.convert_to_prob_distribution(
                keys_dict["fake_keys_one_cluster"], i_f
            ),
            "keys_prob_dist": self.convert_to_prob_distribution(
                keys_dict["all_keys_one_cluster"], i_f
            ),
            "upperbound_dist": keys_dict["upperbound"],
        }

        selection_mapping = {
            "fake": "fake_prob_dist",
            "real": "real_prob_dist",
            "all": "keys_prob_dist",
            "upperbound": "upperbound_dist",
        }

        self.prototype_selection = selection_mapping.get(keys_dict["prototype"], None)

        text_f = text_f / text_f.norm(dim=-1, keepdim=True)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)

        self.K = (
            self.K // self.total_tasks
        )  # restore K to original value for cosine similarity
        logits = self.inference_cosine_similarity(
            text_f, img_f, prob_dist_dict
        )  # * [B,T,P]
        logits = logits
        return logits

    def convert_to_prob_distribution(self, keys, i_f):
        domain_cls = torch.einsum("bd,td->bt", i_f, keys)
        domain_cls = nn.functional.softmax(domain_cls, dim=1)
        return domain_cls

    def inference_cosine_similarity(self, text_f, img_f, prob_dist_dict):
        if self.ensemble_after_cosine_sim:
            assert (
                self.ensemble_before_cosine_sim == False
                and self.ensemble_token_embedding == False
            )
            text_f = rearrange(
                text_f,
                "(b l o) k d -> b l o k d",
                b=img_f.shape[0],
                l=len(cddb_classnames.values()),
                o=self.topk_classes,
            )
            logits = []
            for t in range(self.total_tasks):
                logits_tmp = torch.zeros(img_f.shape[0], text_f.shape[1], device=self.device)  # * [B,P]

                t_img_domain_cls = prob_dist_dict[self.prototype_selection][:, t].unsqueeze(-1)  # * [B, 1]
                t_text_domain_cls = t_img_domain_cls.unsqueeze(-1).unsqueeze(-1)

                for k in range(self.K):
                    offset = k + t * self.K
                    i_img_f = img_f[:, offset, :] * t_img_domain_cls  # * [B,D]
                    i_text_f = (text_f[:, :, :, offset, :] * t_text_domain_cls)  # * [B,P,D]
                    logit = torch.einsum("bd,blod->blo", i_img_f, i_text_f)
                    if self.confidence_score_enable:
                        logit = torch.einsum("blo,blo->bl", logit, self.score_weights_labels)
                    else:
                        logit = reduce(logit, "b l o -> b l", "mean")  # * [B,P]
                    logit = self.logit_scale.exp() * logit
                    logits_tmp += logit
                logits_tmp /= self.K
                logits.append(logits_tmp)

        else:
            text_f = rearrange(
                text_f,
                "(b p) k d -> b p k d",
                b=img_f.shape[0],
                p=len(cddb_classnames.values()),
            )  # * [B,P,K*T,D]
            logits = []
            for t in range(self.total_tasks):
                logits_tmp = torch.zeros(img_f.shape[0], text_f.shape[1], device=self.device)  # * [B,P]

                t_img_domain_cls = prob_dist_dict[self.prototype_selection][:, t].unsqueeze(-1)  # * [B, 1]
                t_text_domain_cls = t_img_domain_cls.unsqueeze(-1)  # * [B, P, 1]
                # t_text_domain_cls = stack_real_fake_prob[:,:,t].unsqueeze(-1)                                     #* [B, P, 1]

                for k in range(self.K):
                    offset = k + t * self.K
                    i_img_f = img_f[:, offset, :] * t_img_domain_cls  # * [B,D]
                    i_text_f = text_f[:, :, offset, :] * t_text_domain_cls  # * [B,P,D]
                    logit = torch.einsum("bd,bpd->bp", i_img_f, i_text_f)  # * [B,P]
                    logit = self.logit_scale.exp() * logit  # * t_img_domain_cls
                    logits_tmp += logit
                logits_tmp /= self.K
                logits.append(logits_tmp)
        logits = torch.stack(logits)  # * [T,B,P]
        logits = rearrange(logits, "t b p -> b t p")  # * [B,T,P]
        return logits

    def update_fc(self):
        self.numtask += 1

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def logging_cfg(self):
        args = {
            attr: getattr(self.cfg, attr)
            for attr in dir(self.cfg)
            if not attr.startswith("_")
        }
        for key, value in args.items():
            logging.info("CFG -> {}: {}".format(key, value))
