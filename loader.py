import torch, os
import PIL, random
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torchvision.transforms.functional import InterpolationMode
from eda import synonym_replacement, random_deletion, random_insertion, random_swap, get_only_chars
from eda import stop_words


def clean_text(sent):
    words = sent.split(" ")
    words = [w for w in words if w not in stop_words]
    if len(words) > 0:
        return " ".join(words)
    else:
        return sent


def build_transformer(is_train, is_train_mae=False, is_train_clip=False, input_size=224):
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]

    if is_train_mae:
        """mae image augmentation"""
        transform = transforms.Compose([
            transforms.CenterCrop(size=input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        return transform
    elif is_train:
        """ scimclr image augmentation"""
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=input_size, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
            # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            # transforms.RandomGrayscale(p=0.3),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        return transform
    elif is_train_clip:
        """clip image augmentation"""
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=input_size, scale=(0.65, 1.0), interpolation=InterpolationMode.BICUBIC),
            # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.6),
            # transforms.RandomGrayscale(p=0.3),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.3),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        return transform
    else:
        t = []
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1
        size = int(input_size / crop_pct)
        t.append(transforms.Resize(size, interpolation=PIL.Image.BICUBIC))
        t.append(transforms.CenterCrop(input_size))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)


class GaussianBlur(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(PIL.ImageFilter.GaussianBlur(sigma))

        return x


class Image_Text_Pair_Masked_DataSet(Dataset):
    def __init__(self,
                 csv_path,
                 max_length=20,
                 text_aug=False,
                 is_train=False,
                 train_mae=False,
                 train_nativecon=False,
                 train_simsam=False,
                 ratio=0.15,
                 fix_text=False,
                 multi_cap=False,
                 select_idx=-1,
                 shuffle=False,
                 pair=True,
                 is_train_clip=False
                 ):

        self.data = pd.read_csv(csv_path)
        # if shuffle and "category" in self.data:
        #     indexs = list(self.data.index)
        #     random.shuffle(indexs)
        #     self.data = self.data.iloc[indexs, :]
        #     self.data = self.data.sort_values('category')
        #     self.data.reset_index(inplace=True, drop=True)

        # self.data = self.data.sample(frac=0.025, replace=False)
        # self.data.reset_index(inplace=True,drop=True)

        self.select_idx = select_idx
        self.pair = pair

        self.base_transform = build_transformer(is_train, is_train_mae=train_mae, is_train_clip=is_train_clip)
        self.max_length = max_length
        self.text_aug = text_aug
        self.train_mae = train_mae
        self.train_nativecon = train_nativecon
        self.train_simsam = train_simsam
        self.fix_text = fix_text
        self.multi_cap = multi_cap
        self.ratio = ratio

    def word_repeatation(self, text, dup_ratio=0.3):
        """text augmentation for nativecon"""
        try:
            act_len = len(text.split(" "))
            dup_len = random.randint(0, b=max(2, int(act_len * dup_ratio)))
            dup_word_index = random.sample(list(range(0, act_len)), k=dup_len)

            dup_words = []
            for index, word in enumerate(text.split(" ")):
                dup_words.append(word)
                if index in dup_word_index:
                    dup_words.append(word)

            return ' '.join(dup_words)
        except:
            return text

    def clean_text(self, sent):
        words = sent.split(" ")
        words = [w for w in words if w not in stop_words]
        if len(words) > 0:
            words = words[:60]
            return " ".join(words)
        else:
            return sent

    def aug_single_sentences(self, sentence, alpha_sr=0.02, alpha_ri=0.02, alpha_rs=0.02, p_rd=0.01):
        """EAD text augmentation"""
        sentence = get_only_chars(sentence)
        words = sentence.split(' ')
        words = [word for word in words if word is not '']
        num_words = len(words)

        choice = random.randint(1, 5)
        try:
            if choice == 1:
                n_sr = max(1, int(alpha_sr * num_words))
                a_words = synonym_replacement(words, n_sr)
                return ' '.join(a_words)
            elif choice == 2:
                n_ri = max(1, int(alpha_ri * num_words))
                a_words = random_insertion(words, n_ri)
                return ' '.join(a_words)
            elif choice == 3:
                n_rs = max(1, int(alpha_rs * num_words))
                a_words = random_swap(words, n_rs)
                return ' '.join(a_words)
            elif choice == 4:
                a_words = random_deletion(words, p_rd)
                return ' '.join(a_words)
            else:
                return self.word_repeatation(sentence, dup_ratio=0.08)
        except:
            return sentence

    def __getitem__(self, item):
        img_path = os.path.join(self.base_path, self.data['img'][item])
        img = Image.open(img_path).convert('RGB')
        img1 = self.base_transform(img)
        text = self.data['text'][item].lower()

        if self.train_simsam:
            img2 = self.base_transform(img)

        if self.text_aug:
            text = self.aug_single_sentences(text)
        elif self.train_nativecon:
            text2 = self.word_repeatation(text)

        inputs1 = self.tokenizer.encode_plus(text,
                                             max_length=self.max_length,
                                             padding="max_length",
                                             truncation=True,
                                             add_special_tokens=True,
                                             return_token_type_ids=True,
                                             return_attention_mask=True,
                                             return_tensors="pt")

        inputs1['labels'] = inputs1.input_ids.detach().clone()
        rand1 = torch.rand(inputs1.input_ids.shape)
        mask_arr1 = (rand1 < self.ratio) * (inputs1.input_ids != 101) * (inputs1.input_ids != 102) * (
                inputs1.input_ids != 0)
        selection1 = torch.flatten(mask_arr1[0].nonzero()).tolist()
        inputs1.labels[0, selection1] = 103

        if self.train_nativecon:
            inputs2 = self.tokenizer.encode_plus(text2,
                                                 max_length=self.max_length,
                                                 padding="max_length",
                                                 truncation=True,
                                                 add_special_tokens=True,
                                                 return_token_type_ids=True,
                                                 return_attention_mask=True,
                                                 return_tensors="pt")

        if self.train_simsam:
            ret_data = {
                "img1": img1,
                "img2": img2,
                "input_ids1": inputs1.input_ids,
                "token_type_ids1": inputs1.token_type_ids,
                "attention_mask1": inputs1.attention_mask,
                "labels1": inputs1.labels,
            }
        elif self.train_nativecon:
            ret_data = {
                "img1": img1,

                "input_ids1": inputs1.input_ids,
                "token_type_ids1": inputs1.token_type_ids,
                "attention_mask1": inputs1.attention_mask,
                "labels1": inputs1.labels,

                "input_ids2": inputs2.input_ids,
                "token_type_ids2": inputs2.token_type_ids,
                "attention_mask2": inputs2.attention_mask,
            }
        else:
            ret_data = {
                "img1": img1,
                "input_ids1": inputs1.input_ids,
                "token_type_ids1": inputs1.token_type_ids,
                "attention_mask1": inputs1.attention_mask,
                "labels1": inputs1.labels,
            }
        return ret_data

    def __len__(self):
        return self.data.shape[0]


if __name__ == "__main__":
    dataset = Image_Text_Pair_Masked_DataSet()
