from pydoc import text
import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from train import MyData, TextCLIP, ImageCLIP
from clip import clip

main_device = torch.device("cuda:0")  # cuda:0
BATCH_SIZE = 256  # 64-128-256-512


def compute_similarity(image_features, text_features, bs=100):
    max_pairs = image_features.shape[0]
    similarity_scores = torch.zeros(max_pairs, max_pairs)
    for v in range(0, max_pairs, bs):
        for t in range(0, max_pairs, bs):
            batch_visual_emb = image_features[v:v + bs]
            batch_caption_emb = text_features[t:t + bs]

            logits = batch_visual_emb @ batch_caption_emb.t()
            similarity_scores[v:v + bs, t:t + bs] = logits

    print('Done similarity')
    return similarity_scores


def compute_retrieval(a2b_sims, return_ranks=False):
    npts = a2b_sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(a2b_sims[index])[::-1]
        where = np.where(inds == index)
        rank = where[0][0]
        ranks[index] = rank
        top1[index] = inds[0]
    r1 = round(100.00 * len(np.where(ranks < 1)[0]) / len(ranks), 4)
    r5 = round(100.00 *len(np.where(ranks < 5)[0]) / len(ranks), 4)
    r10 = round(100.00 * len(np.where(ranks < 10)[0]) / len(ranks), 4)
    r50 = round(100.00 * len(np.where(ranks < 50)[0]) / len(ranks), 4)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    precision = round(100.00 * sum([1 / (x + 1) for x in ranks]) / len(ranks), 4)

    report_dict = {"r1": r1, "r5": r5, "r10": r10, "r50": r50, "precision": precision, "medr": medr, "meanr": meanr, "sum": r1 + r5 + r10}

    if return_ranks:
        return report_dict, (ranks, top1)
    else:
        return report_dict


def evaluate(val_dataloder, model_image, model_text, model):
    model_image.eval()
    model_text.eval()
    model.eval()

    print(len(val_dataloder))
    image_feats = []
    text_feats = []
    image_feats_list = []
    text_feats_list = []

    for idx, batch in enumerate(val_dataloder):
        img1 = batch['img1']
        img1 = img1.to(main_device)

        text1 = batch['text1']
        images = img1
        input_ids, attention_mask, token_type_ids = clip.tokenize(text1)

        with torch.no_grad():
            image_embed, _ = model.encode_image_v1(images)
            text_embed, _ = model.encode_text_v1(input_ids.to(main_device), attention_mask.to(main_device), token_type_ids.to(main_device))

        text_feats.append(text_embed)
        image_feats.append(image_embed)

        # text_feats_list.append(text_feat)
        # image_feats_list.append(image_feat)

    image_feats = torch.cat(image_feats, 0)
    text_feats = torch.cat(text_feats, 0)

    image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    similarity = compute_similarity(image_feats, text_feats)
    print(similarity.shape)
    i2t_dict = compute_retrieval(similarity.numpy())
    t2i_dict = compute_retrieval(similarity.t().numpy())

    print("image2text: ", i2t_dict)
    print("text2image: ", t2i_dict)
    print()

    """
    latent_list = torch.cat(image_feats_list, 0)
    logits_list = torch.cat(text_feats_list, 0)
    similarity= similarity.t()
    score_matrix_i2t = torch.full((similarity.shape[0], similarity.shape[1]), -100.0).to(main_device)
    txt2img = []
    tok_idxs = []

    for i in range(similarity.shape[0]):
        sims = similarity[i]
        topk_sim, topk_idx = sims.topk(k=topk, dim=0)

        logits_output = logits_list[i].repeat(topk, 1, 1)  # topk,20,512
        latent_output = latent_list[topk_idx]  # topk,patches,512

        tok_idxs.append(topk_idx)
        txt2img.append(i)

        with torch.no_grad():
            output = model.cross_attenion_i2t(logits_output, latent_output, latent_output)
            output = output[torch.arange(logits_output.shape[0]), argmax_ids[i]]
            scores = model.text_matcher(output)[:, 1]  # topk,1

        scores = scores.type(score_matrix_i2t.dtype)
        score_matrix_i2t[i, topk_idx] = scores

    ranks = np.zeros(score_matrix_i2t.shape[0])
    # text ---> image
    for index, score in enumerate(score_matrix_i2t.cpu().numpy()):
        inds = np.argsort(score)[::-1]
        rank = np.where(inds == txt2img[index])[0][0]
        ranks[index] = rank

    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    print(ir1, ir5, ir10)

    img2txt = []
    tok_idxs = []
    similarity = similarity.t()
    score_matrix_t2i = torch.full((similarity.shape[0], similarity.shape[1]), -100.0).to(main_device)
    for i in range(similarity.shape[0]):
        sims = similarity[i]
        topk_sim, topk_idx = sims.topk(k=topk, dim=0)

        latent_output = latent_list[i].repeat(topk, 1, 1)  # topk,197,768
        logits_output = logits_list[topk_idx]  # topk,50,768


        tok_idxs.append(topk_idx)
        img2txt.append(i)

        with torch.no_grad():
            output = model.cross_attenion_t2i(latent_output, logits_output, logits_output)
            scores = model.image_matcher(output[:,0,:])[:, 1]  # topk,1

        scores = scores.type(score_matrix_t2i.dtype)
        score_matrix_t2i[i, topk_idx] = scores

    ranks = np.zeros(score_matrix_t2i.shape[0])
    # image ---> text
    for index, score in enumerate(score_matrix_t2i.cpu().numpy()):
        inds = np.argsort(score)[::-1]
        rank = np.where(inds == img2txt[index])[0][0]
        ranks[index] = rank

    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    print(tr1, tr5, tr10)
    """


##########################################################
val_dataset = MyData('/MIMIC-CXR/no_Dr_annotation/mimic_test_images_withnames_notwosapces.csv',  #  /usr/ext_openv/zhuyi143/medical/MIMIC-CXR/no_Dr_annotation/mimic_test_images_withnames.csv
                     is_train=False)
                     # /usr/ext_openv/zhuyi143/medical/iu_xray/iu_xray_test_withnames_1k.csv
val_dataloader = DataLoader(val_dataset,
                            shuffle=False,
                            batch_size=BATCH_SIZE,
                            num_workers=12,
                            drop_last=False
                            )
# for data in val_dataloader:
#     for i in data['text1']:
#         print(i)
#         if i == 4:
#             break

model, _ = clip.load("/yourdir/epoch_.pt"  # 72 68
                     , device=main_device, jit=False, from_scratch=False, eval=True)

model_text = TextCLIP(model)
model_image = ImageCLIP(model)

model_text = torch.nn.DataParallel(model_text, device_ids=[0, 1, 2, 3])
model_image = torch.nn.DataParallel(model_image, device_ids=[0, 1, 2, 3])
# model.to(main_device)

val1k = True

if not val1k:
    evaluate(val_dataloader, model_image, model_text, model)
else:
    random_seed = 42
    indices = list(range(25000))
    split = 1000
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    val1k_indices = indices[:split]
    # print(val1k_indices)
    val1k_sampler = SubsetRandomSampler(val1k_indices)
    val1k_dataloader = DataLoader(val_dataset,
                                  shuffle=False,
                                  batch_size=BATCH_SIZE,
                                  num_workers=12,
                                  drop_last=False,
                                  sampler=val1k_sampler
                                  )
    evaluate(val1k_dataloader, model_image, model_text, model)
