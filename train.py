import torch
import torch.nn as nn
from clip import clip
from torch.utils.data import DataLoader
from PIL import Image
from loader import Image_Text_Pair_Masked_DataSet
from torch.utils.data import RandomSampler, SequentialSampler
import os
from transformers.optimization import AdamW
import torch.nn.functional as F
import random
import pandas as pd
from transformers import ViTFeatureExtractor
import math

main_device = torch.device("cuda:0")
os.environ["CUDA_VIDIBLE_DEVICES"] = "0,1,2,3"
EPOCH = 100
BATCH_SIZE = 512
nativecon_BATCH_SIZE = 400
fl_BATCH_SIZE = 400


class MyData(Image_Text_Pair_Masked_DataSet):
    def __getitem__(self, item):

        if self.train_nativecon:
            text1 = self.data['impression'][item].lower()
            text2 = self.data['findings'][item].lower()
            text1 = self.aug_single_sentences(text1)
            text1 = self.clean_text(text1)
            text2 = self.aug_single_sentences(text2)
            text2 = self.clean_text(text2)
        elif self.train_fl:
            img1 = self.data['frontal'][item]
            img2 = self.data['lateral'][item]
            img1 = Image.open(img1).convert('RGB')
            img1 = self.base_transform(img1)
            img2 = Image.open(img2).convert('RGB')
            img2 = self.base_transform(img2)
        else:
            img_path = self.data['img'][item]
            img = Image.open(img_path).convert('RGB')
            img1 = self.base_transform(img)
            text = self.data['text'][item].lower()

            text1 = self.aug_single_sentences(text)
            text1 = self.clean_text(text1)

        if not self.pair:
            label = self.data['label'][item]

        if self.train_nativecon:
            ret_data = {
                "text1": text1,
                "text2": text2
            }
            return ret_data
        elif self.train_fl:
            ret_data = {
                "img1": img1,
                "img2": img2
            }
            return ret_data
        if not self.pair:
            ret_data = {
                "img1": img1,
                "text1": text1,
                'label': label
            }

            return ret_data

        ret_data = {
            "img1": img1,
            "text1": text1
        }

        return ret_data


class ContrastiveLoss(nn.Module):
    def __init__(self, bs=512, dev=torch.device("cpu"), t=0.5):
        super().__init__()
        self.bs = bs
        self.dev = dev
        self.register_buffer("temperature", torch.tensor(t))
        self.register_buffer("negatives_mask", (~torch.eye(bs * 2, bs * 2, dtype=bool)).float())

    def forward(self, embed_i, embed_j):
        z_i = F.normalize(embed_i, dim=1)
        z_j = F.normalize(embed_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)

        similarity = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        temperature = self.temperature.to(self.dev)
        negatives_mask = self.negatives_mask.to(self.dev)

        ij = torch.diag(similarity, self.bs)
        ji = torch.diag(similarity, -self.bs)
        pos = torch.cat([ij, ji], dim=0)

        nom = torch.exp(pos / temperature)
        denom = negatives_mask * torch.exp(similarity / temperature)

        loss = -torch.log(nom / torch.sum(denom, dim=1))
        loss = torch.sum(loss) / (2 * self.bs)

        return loss


# compute nativecon loss
def nativecon_loss(y_pred):
    idx = torch.arange(0, y_pred.shape[0], device=y_pred.device)
    y_true = idx + 1 - idx % 2 * 2  # 1, 0, 3, 2, 5, 4, 7, 6, 9, 8
    simi = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    simi = simi - torch.eye(y_pred.shape[0], device=y_pred.device)
    simi = simi / 0.05
    loss = F.cross_entropy(simi, y_true)
    return torch.mean(loss)

def knegative_contrast_loss(logits, k):
    # logits:bs, bs with gt in diagonal
    diag = torch.diag(logits).unsqueeze(1) #bs,1

    logits -= diag
    logits = logits.topk(k).values
    out = torch.cat((diag, logits), dim=1)
    gt = torch.ones(logits.size()[0], dtype=torch.long).cuda()
    # print(out.size())
    loss = nn.CrossEntropyLoss().cuda()
    return loss(out, gt)

# total
train_dataset2 = MyData(
    '/merge_cluster_100_image_notwosapces_new_address.csv',
    is_train=True
    )
train_sampler2 = SequentialSampler(train_dataset2)
train_dataloader2 = DataLoader(train_dataset2,
                               sampler=train_sampler2,
                               batch_size=BATCH_SIZE,
                               num_workers=16,
                               drop_last=True
                               )

train_dataset_shuffle = MyData('/datasets/MIMIC-CXR/no_Dr_annotation/mimic_iuxray_merge.csv',
                               is_train=True
                               )
train_sampler_shuffle = RandomSampler(train_dataset_shuffle)
train_dataloader_shuffle = DataLoader(train_dataset_shuffle,
                                      sampler=train_sampler_shuffle,
                                      batch_size=BATCH_SIZE,
                                      num_workers=16,
                                      drop_last=True
                                      )

# nativecon
train_dataset_nativecon = MyData('/datasets/MIMIC-CXR/impression_and_findings.csv',  # cluster
                                 train_nativecon=True
                                 )
train_sampler_nativecon = SequentialSampler(train_dataset_nativecon)
train_dataloder_nativecon = DataLoader(train_dataset_nativecon,
                                       sampler=train_sampler_nativecon,
                                       batch_size=nativecon_BATCH_SIZE,
                                       num_workers=16,
                                       drop_last=True
                                       )

# FL
train_dataset_fl = MyData('/datasets/MIMIC-CXR/frontal_and_lateral_new_address.csv',  # cluster
                                 train_fl=True
                                 )
train_sampler_fl = SequentialSampler(train_dataset_nativecon)
train_dataloder_fl = DataLoader(train_dataset_fl,
                                       sampler=train_sampler_fl,
                                       batch_size=fl_BATCH_SIZE,
                                       num_workers=16,
                                       drop_last=True
                                       )


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()
    return logits_per_x1, logits_per_x2


class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        ret, _ = self.model.encode_text_v1(text)
        return ret


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        ret, _ = self.model.encode_image_v1(image)
        return ret


def train():

    # model, _ = clip.load("/usr/ext_openv/zhuyi143/checkpoints/pretrained_vit_medical/ckpt_epoch_30.pth",
    #                      device=main_device, jit=False,
    #                      from_scratch=False, eval=False)  # /usr/openv/zhuyi143/medCLIP/C2L_res18_model.pt

    model, _ = clip.load(
        "/1.pt",
        device=main_device, jit=False,
        from_scratch=False, eval=False)

    loss_text = nn.CrossEntropyLoss()
    loss_img = nn.CrossEntropyLoss()
    linear_matcher_params, total_params = [], []
    for name, para in model.named_parameters():
        if "linear_matcher1" in name or "linear_matcher2" in name or "merge_attenion" in name:
            linear_matcher_params.append(para)
    for name, para in model.named_parameters():
        if "linear_matcher1" not in name and "linear_matcher2" not in name and "merge_attenion" not in name:
            total_params.append(para)
    total_optimizer = AdamW([
        {"params": total_params},
        {"params": linear_matcher_params, "lr": 8e-5}
    ], lr=5e-6, betas=(0.9, 0.98), eps=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(total_optimizer, T_max=30)
    nativecon_params, proj_params = [], []
    for name, para in model.named_parameters():
        if ("transformer" in name and "visual" not in name) or "ln_final" in name or "token_embedding" in name:
            nativecon_params.append(para)
    for name, para in model.named_parameters():
        if 'imp2fid' in name or 'fid2imp' in name:
            proj_params.append(para)
    optimizer_nativecon = AdamW([
        {"params": nativecon_params},
        {"params": proj_params, "lr": 8e-5}
    ], betas=(0.9, 0.98), lr=5e-6, eps=1e-6)
    fl_params, proj_fl_params = [], []
    for name, para in model.named_parameters():
        if "visual" in name and "transformer" not in name:
            fl_params.append(para)
    for name, para in model.named_parameters():
        if 'f2l' in name or 'l2f' in name:
            proj_fl_params.append(para)
    optimizer_fl = AdamW([
        {"params": fl_params},
        {"params": proj_fl_params, "lr": 8e-5}
    ], betas=(0.9, 0.98), lr=5e-6, eps=1e-6)

    model.to(main_device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(0, EPOCH):
        model.train()
        if epoch < 5:
            for idx, batch in enumerate(train_dataloder_nativecon):
                optimizer_nativecon.zero_grad()
                text1 = batch['text1']
                input_ids1, attention_mask1, token_type_ids1 = clip.tokenize(text1)

                text2 = batch['text2']
                input_ids2, attention_mask2, token_type_ids2 = clip.tokenize(text2)

                bs, lens = input_ids1.size()
                texts = torch.zeros((2 * bs, lens), dtype=input_ids1.dtype, device=input_ids1.device)
                token_type_ids = torch.zeros((2 * bs, lens), dtype=token_type_ids1.dtype, device=token_type_ids1.device)
                attention_mask = torch.zeros((2 * bs, lens), dtype=attention_mask1.dtype, device=attention_mask1.device)

                texts[::2, :] = input_ids1  # 步长为2
                texts[1::2, :] = input_ids2
                attention_mask[::2, :] = attention_mask1
                attention_mask[1::2, :] = attention_mask2

                text_features_imp, text_features_fid, text_features_imp_pro, text_features_fid_pro = model(None,
                                                                                                           texts.to(
                                                                                                               main_device),
                                                                                                           attention_mask.to(
                                                                                                               main_device),
                                                                                                           token_type_ids.to(
                                                                                                               main_device),
                                                                                                           train_nativecon=True)
                text_features1 = torch.zeros(2 * bs, 768)
                text_features2 = torch.zeros(2 * bs, 768)
                text_features1[::2, :] = text_features_imp
                text_features1[1::2, :] = text_features_fid_pro
                text_features2[::2, :] = text_features_fid
                text_features2[1::2, :] = text_features_imp_pro
                loss1 = nativecon_loss(text_features1).mean()
                loss2 = nativecon_loss(text_features2).mean()
                loss = (loss1 + loss2) / 2

                loss.backward()
                optimizer_nativecon.step()

                print("epoch: ", epoch + 1, "step: ", idx + 1, ", native contrast training loss: ",
                      round(loss.item(), 3))
        elif 4< epoch < 10:
            for idx, batch in enumerate(train_dataloder_fl):
                optimizer_fl.zero_grad()
                img1 = batch['img1']
                img2 = batch['img2']
                # print(img1.size())
                bs,c,h,w = img1.size()
                img = torch.zeros((2 * bs, c,h,w), dtype=img1.dtype, device=img1.device)

                img[::2, :, :, :] = img1  # 步长为2
                img[1::2, :, :, :] = img2

                img_features_f, img_features_l, img_features_f_pro, img_features_l_pro = model(img,None,None,None,train_fl=True)
                img_features1 = torch.zeros(2 * bs, 768)
                img_features2 = torch.zeros(2 * bs, 768)
                img_features1[::2, :] = img_features_f
                img_features1[1::2, :] = img_features_l_pro
                img_features2[::2, :] = img_features_l
                img_features2[1::2, :] = img_features_f_pro
                loss1 = nativecon_loss(img_features1).mean()
                loss2 = nativecon_loss(img_features2).mean()
                loss = (loss1 + loss2) / 2

                loss.backward()
                optimizer_fl.step()

                print("epoch: ", epoch + 1, "step: ", idx + 1, ", FL contrast training loss: ",
                      round(loss.item(), 3))
        else:
            for idx, batch in enumerate(train_dataloader2):
                total_optimizer.zero_grad()
                img1 = batch['img1']
                bs = img1.shape[0]

                images = img1.to(main_device)
                text1 = batch['text1']
                input_ids, attention_mask, token_type_ids = clip.tokenize(text1)

                with torch.cuda.amp.autocast():
                    loss, image_features, text_features, logit_scale = model(images, input_ids, attention_mask,
                                                                             token_type_ids)
                    # ground_truth = torch.arange(bs, dtype=torch.long).to(main_device)
                    logit_scale = logit_scale.mean()

                    # logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_image = image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()
                    # print(image_features[0])
                    # print(logits_per_image[0])
                    # align and uniformity
                    align_img = 0
                    for i in range(bs):
                        negatives = (logits_per_image[i].mean() * bs - logits_per_image[i][i]) / (bs - 1)
                        align_img += negatives
                    align_img = align_img / bs

                    align_txt = 0
                    for i in range(bs):
                        negatives = (logits_per_text[i].mean() * bs - logits_per_text[i][i]) / (bs - 1)
                        align_txt += negatives
                    align_txt = align_txt / bs

                    logits_per_image_uniform = logits_per_image
                    # for i in range(logits_per_image.size()[0]):
                    #     for j in range(logits_per_image.size()[1]):
                    #         logits_per_image_uniform[i][j] = math.exp(logits_per_image_uniform[i][j])
                    logits_per_image_uniform = torch.exp(logits_per_image_uniform)

                    uniform = torch.log(logits_per_image_uniform.mean())
                    K_hat_img = math.floor(bs * math.cos(math.pi / 8 * (2 + uniform - align_img)))
                    K_img = max(1, min(K_hat_img, bs - 1))
                    K_hat_txt = math.floor(bs * math.cos(math.pi / 8 * (2 + uniform - align_txt)))
                    K_txt = max(1, min(K_hat_txt, bs - 1))
                    if idx%30 ==0:
                        print(K_img, K_txt)

                    loss_contra_i2t = knegative_contrast_loss(logits_per_image, K_img)
                    loss_contra_t2i = knegative_contrast_loss(logits_per_text, K_txt)

                    loss = loss.mean()
                    loss += loss_contra_i2t + loss_contra_t2i

                scaler.scale(loss).backward()
                scaler.step(total_optimizer)
                scaler.update()

                print("epoch: ", epoch + 1, "step: ", idx + 1, ",clustered image total training loss: ",
                      round(loss.item(), 3))

            # for idx, batch in enumerate(train_dataloader_shuffle):
            #     total_optimizer.zero_grad()
            #     img1 = batch['img1']
            #     bs = img1.shape[0]
            #
            #     images = img1.to(main_device)
            #     text1 = batch['text1']
            #     input_ids, attention_mask, token_type_ids = clip.tokenize(text1)
            #
            #     with torch.cuda.amp.autocast():
            #         loss, image_features, text_features, logit_scale = model(images, input_ids, attention_mask,
            #                                                                  token_type_ids)
            #         ground_truth = torch.arange(bs, dtype=torch.long).to(main_device)
            #         logit_scale = logit_scale.mean()
            #
            #         logits_per_image = logit_scale * image_features @ text_features.t()
            #         logits_per_text = logits_per_image.t()
            #
            #         loss_contra_t2i = loss_img(logits_per_image, ground_truth)
            #         loss_contra_i2t = loss_text(logits_per_text, ground_truth)
            #
            #         loss = loss.mean()
            #         loss += loss_contra_i2t + loss_contra_t2i
            #
            #     scaler.scale(loss).backward()
            #     scaler.step(total_optimizer)
            #     scaler.update()
            #
            #     print("epoch: ", epoch + 1, "step: ", idx + 1, ",shuffle img and text total training loss: ",
            #           round(loss.item(), 3))

            scheduler.step()

            if epoch % 2 == 0 or epoch == EPOCH - 1:
                save_path = "/yourdir/epoch_" + str(epoch + 1) + "_" + str(round(loss.item(), 4)) + ".pt"
                torch.save(model.module.state_dict(), save_path)

        if epoch == 4:
            del optimizer_nativecon
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import gc
    train()