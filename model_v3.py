from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from transformers.models.bert.modeling_bert import BertModel, BertForMaskedLM
from transformers import BertConfig
from transformers.models.bert.configuration_bert import BertConfig
# import torchvision.models as models
from transformers import ViTModel, ViTConfig
from collections import OrderedDict
# from loss import TripletLoss


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)
        # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)
        # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        att = self.dropout(torch.softmax(att, -1))
        att = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        # (b_s, nq, h*d_v)
        att = self.fc_o(att)  # (b_s, nq, d_model)
        return att


class Attention(nn.Module):
    def __init__(self, d_q, d_v, dropout=.1):
        super(Attention, self).__init__()
        self.fc_q = nn.Linear(d_q, d_q)
        self.fc_k = nn.Linear(d_q, d_q)  # d_k=d_q
        self.fc_v = nn.Linear(d_v, d_v)
        self.dropout = nn.Dropout(dropout)

        self.d_q = d_q
        self.d_v = d_v

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, dropout=0):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries)
        # (b_s_q, d_q)
        print(keys.size())
        k = self.fc_k(keys).permute(1, 0)
        # (d_q, b_s_k)
        v = self.fc_v(values)
        # (b_s_k, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_q)
        # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        if dropout > 0:
            att = self.dropout(torch.softmax(att, -1), p=dropout)
        att = torch.matmul(att, v)
        # (b_s, nq, d_v)
        return att


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)



class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length
        self.vit_config = ViTConfig.from_pretrained('/modelweights/vit224')
        self.visual = ViTModel(config=self.vit_config)

        # self.transformer = BertForMaskedLM.from_pretrained(
        #     "/usr/openv/zhuyi143/medCLIP/ClinicalBERT")  # BERT-Base (cased_L-12_H-768_A-12)
        self.transformer = BertModel.from_pretrained("/medCLIP/ClinicalBERT")
        # self.transformer = BertModel.from_pretrained('/usr/openv/zhuyi143/medCLIP/ChexBERT/save_binary_file')

        self.vocab_size = vocab_size
        # self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        # self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        # self.text_pro = nn.Linear(768, embed_dim)
        # self.image_pro = nn.Linear(768, 512)  # 512 for resnet, 768 for vit, 768*2 for CA
        self.imp2fid = nn.Linear(768, 768)
        self.fid2imp = nn.Linear(768, 768)
        self.f2l = nn.Linear(768, 768)
        self.l2f = nn.Linear(768, 768)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.merge_attenion = ScaledDotProductAttention(d_model=768, d_k=64, d_v=64, h=8)
        self.linear_matcher1 = nn.Linear(2 * 768, 300)
        self.linear_matcher2 = nn.Linear(300, 2)
        self.loss = nn.CrossEntropyLoss()

        # adaptive pooling
        # self.weights = nn.Parameter(torch.zeros(50))  # torch.ones
        # self.weights = nn.Parameter(torch.zeros(1))
        # for softmax(w1)*cls + softmax(w2)*patch
        self.weight1 = nn.Parameter(torch.zeros(1))
        self.weight2 = nn.Parameter(torch.zeros(1))

        # self.initialize_parameters()

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        output = self.visual(image.type(self.dtype))
        if isinstance(output, (tuple, list)):
            return output[0]
        else:
            return output

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        ret, x = self.transformer(x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        eot = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return eot

    def encode_image_v1(self, image):
        # for vit
        # global_feature = self.visual(image)
        # global_feature = self.image_pro(global_feature[0])
        # return (global_feature[:, 0, :], global_feature)

        # # for adaptive pooling
        # bs = image.size()[0]
        # weights = torch.sigmoid(weights)
        # weights = self.weights.repeat(bs, 1, 768)
        # global_feature = self.visual(image)
        # global_feature = global_feature[0]
        # img_feature = torch.mul(weights, global_feature)
        # return (torch.mean(img_feature, dim=1), global_feature)

        # # for w1*cls+(1-w1)*patch
        # bs = image.size()[0]
        # weights = torch.sigmoid(self.weights)
        # cls_weights = torch.tensor([1]).cuda() - weights
        # weights = weights.repeat(bs, 49, 768)
        # cls_weights = cls_weights.repeat(bs, 1, 768)
        # # cls_weights = torch.tensor([1]).repeat(bs, 1, 768).cuda() - weights
        # global_feature = self.visual(image)
        # global_feature = global_feature[0]
        #
        # img_feature = torch.mul(weights, global_feature[:, 1:, :])
        # img_feature = torch.mean(img_feature, dim=1)
        # img_feature = img_feature + torch.mul(cls_weights, global_feature[:, :1, :]).squeeze(1)
        # return (img_feature, global_feature)

        # for softmax(w1)*cls + softmax(w2)*patch
        bs = image.size()[0]
        p = torch.cat((self.weight1.unsqueeze(1), self.weight2.unsqueeze(1)), dim=0)
        weights = torch.softmax(p, dim=0)
        cls_weights = weights[0]
        weights = weights[1]
        weights = weights.repeat(bs, 49, 768)
        cls_weights = cls_weights.repeat(bs, 1, 768)
        # cls_weights = torch.tensor([1]).repeat(bs, 1, 768).cuda() - weights
        global_feature = self.visual(image)
        global_feature = global_feature[0]

        img_feature = torch.mul(weights, global_feature[:, 1:, :])
        img_feature = torch.mean(img_feature, dim=1)
        img_feature = img_feature + torch.mul(cls_weights, global_feature[:, :1, :]).squeeze(1)
        return (img_feature, global_feature)

    def encode_text_v1(self, input_ids, attention_mask, token_type_ids):
        # x = self.token_embedding(text).type(self.dtype)
        # # [batch_size, n_ctx, d_model]
        #
        # x = x + self.positional_embedding.type(self.dtype)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.transformer(x)
        #
        # x = x.permute(1, 0, 2)  # LND -> NLD
        # x = self.ln_final(x).type(self.dtype)
        #
        # # bs,len,512 ---> bs,len,512
        # eot = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        # x = x @ self.text_projection
        #
        # return (eot, x)
        text_output = self.transformer(input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,
                                       output_hidden_states=True,
                                       return_dict=True)
        text_output = text_output['hidden_states'][-1]   # bs, seqlength, 768->512 最后一层
        # print(text_output.size())
        # logits = text_output[1]  # bs, 768

        return (text_output[:, 0, :], text_output)

    def forward(self,
                image,
                input_ids,
                attention_mask,
                token_type_ids,
                labels1=None, labels2=None,
                train_nativecon=False,
                train_fl=False,
                train_matching=True
                ):

        if train_nativecon:
            input_ids_imp = input_ids[::2, :]
            input_ids_fid = input_ids[1::2, :]
            attention_mask_imp = attention_mask[::2, :]
            attention_mask_fid = attention_mask[1::2, :]
            token_type_ids = token_type_ids[::2, :]  # useless
            text_features_imp, _ = self.encode_text_v1(input_ids_imp, attention_mask_imp, token_type_ids)
            text_features_fid, _ = self.encode_text_v1(input_ids_fid, attention_mask_fid, token_type_ids)
            text_features_imp_pro = self.imp2fid(text_features_imp)
            text_features_fid_pro = self.fid2imp(text_features_fid)
            text_features_imp = text_features_imp / text_features_imp.norm(dim=-1, keepdim=True)
            text_features_fid = text_features_fid / text_features_fid.norm(dim=-1, keepdim=True)
            text_features_imp_pro = text_features_imp_pro / text_features_imp_pro.norm(dim=-1, keepdim=True)
            text_features_fid_pro = text_features_fid_pro / text_features_fid_pro.norm(dim=-1, keepdim=True)
            # loss = nativecon_loss(text_features)

            # text_output1 = self.transformer(input_ids_imp, attention_mask_imp, token_type_ids, labels=labels1)
            # text_output2 = self.transformer(input_ids_fid, attention_mask_fid, token_type_ids, labels=labels2)
            # mlm_loss = text_output1.loss + text_output2.loss
            return text_features_imp, text_features_fid, text_features_imp_pro, text_features_fid_pro
        if train_fl:
            img1 = image[::2, :, :, :]
            img2 = image[1::2, :, :, :]
            img_features_f, _ = self.encode_image_v1(img1)
            img_features_l, _ = self.encode_image_v1(img2)
            img_features_f_pro = self.f2l(img_features_f)
            img_features_l_pro = self.l2f(img_features_l)
            img_features_f = img_features_f / img_features_f.norm(dim=-1, keepdim=True)
            img_features_l = img_features_l / img_features_l.norm(dim=-1, keepdim=True)
            img_features_f_pro = img_features_f_pro / img_features_f_pro.norm(dim=-1, keepdim=True)
            img_features_l_pro = img_features_l_pro / img_features_l_pro.norm(dim=-1, keepdim=True)

            return img_features_f, img_features_l, img_features_f_pro, img_features_l_pro
        else:
            """
            image_features/image_features1 : bs,512
            text_features/text_features1 : bs,512
            latent:bs,50,512
            logits:bs,77,512
            """
            batchsize = image.size(0)
            image_features, latent = self.encode_image_v1(image)
            text_features, logits = self.encode_text_v1(input_ids, attention_mask, token_type_ids)

            image_features_ret = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features_ret = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()

            if not train_matching:
                return (image_features_ret, text_features_ret, logit_scale,)
            else:
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                with torch.no_grad():
                    weight_i2t = F.softmax(logits_per_image[:, 0:batchsize], dim=1) + 1e-4
                    weight_t2i = F.softmax(logits_per_text[:, 0:batchsize], dim=1) + 1e-4

                    weight_i2t.fill_diagonal_(0)
                    weight_t2i.fill_diagonal_(0)

                image_embeds_neg1 = []
                image_vecs_neg1 = []

                text_embeds_neg1 = []
                text_vecs_neg1 = []


                for bs in range(batchsize):
                    _, neg_idxs = weight_t2i[bs].topk(3, dim=0)

                    neg_idx = neg_idxs[0].item()
                    image_embeds_neg1.append(latent[neg_idx])
                    image_vecs_neg1.append(image_features[neg_idx])

                for bs in range(batchsize):
                    _, neg_idxs = weight_i2t[bs].topk(3, dim=0)

                    neg_idx = neg_idxs[0].item()
                    text_embeds_neg1.append(logits[neg_idx])
                    text_vecs_neg1.append(text_features[neg_idx])

                image_embeds_neg1 = torch.stack(image_embeds_neg1, dim=0)
                text_embeds_neg1 = torch.stack(text_embeds_neg1, dim=0)

                image_vecs_neg1 = torch.stack(image_vecs_neg1, dim=0)
                text_vecs_neg1 = torch.stack(text_vecs_neg1, dim=0)

                pos = torch.cat([latent, logits], dim=1)  # bs,50+77,512

                pos = torch.mean(self.merge_attenion(pos, pos, pos), dim=1)  # bs,512
                pos = torch.cat([pos, torch.abs(image_features - text_features)], dim=1)  # bs,512*2
                pos = self.linear_matcher1(pos)  # bs,300
                pos = self.linear_matcher2(pos)  # bs,2

                neg1 = torch.cat([latent, text_embeds_neg1], dim=1)
                neg1 = torch.mean(self.merge_attenion(neg1, neg1, neg1), dim=1)
                neg1 = torch.cat([neg1, torch.abs(image_features - text_vecs_neg1)], dim=1)
                neg1 = self.linear_matcher1(neg1)
                neg1 = self.linear_matcher2(neg1)

                neg2 = torch.cat([image_embeds_neg1, logits], dim=1)
                neg2 = torch.mean(self.merge_attenion(neg2, neg2, neg2), dim=1)
                neg2 = torch.cat([neg2, torch.abs(image_vecs_neg1 - text_features)], dim=1)
                neg2 = self.linear_matcher1(neg2)
                neg2 = self.linear_matcher2(neg2)

                pred = torch.cat([neg1, neg2, pos], dim=0)  # 3*bs,2
                true = torch.cat([
                    torch.zeros(2 * batchsize, dtype=torch.long),
                    torch.ones(batchsize, dtype=torch.long)
                ], dim=0).to(image.device)

                loss = self.loss(pred, true)

                return (loss, image_features_ret, text_features_ret, logit_scale,)


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, eval=False):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = 64 
        vision_patch_size = None
        image_resolution = 224

    embed_dim = 512 
    context_length = 125
    vocab_size = 49408 
    transformer_width = 512  
    transformer_heads = transformer_width // 64  
    transformer_layers = 12  

    model = CLIP(
        embed_dim, context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    if eval:
        model.load_state_dict(state_dict, strict=True)
        return model.eval()

    # for vit
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    state_dict = new_state_dict

    for key in ["input_resolution", "context_length", "vocab_size", "positional_embedding", "fc.weight", "fc.bias"]:
        if key in state_dict:
            del state_dict[key]

    # 增加visual前缀
    state_dict = {'visual.' + k: v for k, v in state_dict.items()}

    # convert_weights(model)
    model.load_state_dict(state_dict, strict=False)
    return model.eval()
