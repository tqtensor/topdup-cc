import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from bs4 import BeautifulSoup
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import Parameter
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
from vncorenlp import VnCoreNLP

# Load rdrsegmenter from VnCoreNLP
rdrsegmenter = VnCoreNLP(
    r"./nlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size="-Xmx500m"
)

# Load PhoBERT
model_name = "vinai/phobert-base"
phobert = AutoModel.from_pretrained(model_name)

# For transformers v4.x+:
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# Define Vietnamese valid regex
chars_vn = [
    "á",
    "à",
    "ả",
    "ã",
    "ạ",
    "â",
    "ấ",
    "ầ",
    "ẩ",
    "ẫ",
    "ậ",
    "ă",
    "ắ",
    "ằ",
    "ẳ",
    "ẵ",
    "ặ",
    "ó",
    "ò",
    "ỏ",
    "õ",
    "ọ",
    "ô",
    "ố",
    "ồ",
    "ổ",
    "ỗ",
    "ộ",
    "ơ",
    "ớ",
    "ờ",
    "ở",
    "ỡ",
    "ợ",
    "é",
    "è",
    "ẻ",
    "ẽ",
    "ẹ",
    "ê",
    "ế",
    "ề",
    "ể",
    "ễ",
    "ệ",
    "ú",
    "ù",
    "ủ",
    "ũ",
    "ụ",
    "ư",
    "ứ",
    "ừ",
    "ử",
    "ữ",
    "ự",
    "í",
    "ì",
    "ỉ",
    "ĩ",
    "ị",
    "ý",
    "ỳ",
    "ỷ",
    "ỹ",
    "ỵ",
    "đ",
]
chars_vn = "".join(chars_vn)
chars_vn = chars_vn + chars_vn.upper()
# regex pattern of invalid character
pattern_unchars_vn = f"[^a-zA-Z{chars_vn} \.]"

# GLU CNN
SCALE_WEIGHT = 0.5 ** 0.5


def html_to_text(filename):
    try:
        soup = BeautifulSoup(open(filename), "html.parser")
        contents = list()
        for part in soup.find_all("div", {"class": re.compile(r".*content.*")}):
            contents.append(part.get_text())
        contents = " ".join(contents)
    except:
        return

    article = re.sub("\.+", ".", contents)  # multi dots to one dot
    article = re.sub("[?!;…]", ".", article)  # alls to one dot
    article = re.sub(r" +", " ", article)  # remove white space

    sents = article.split(".")
    sents = [
        re.sub(pattern_unchars_vn, "", sent.strip()) for sent in sents
    ]  # remove invalid VN words
    norm_sents = [s for s in sents if s]
    return norm_sents


def segmentize(sents, threshold):
    seg_sents = list()
    for s in sents:
        try:
            word_segmented_text = rdrsegmenter.tokenize(s.lower())
            seg_sents.append(
                " ".join([" ".join([w for w in s]) for s in word_segmented_text])
            )
        except Exception as e:
            print(e)
            print(s)
    seg_sents = [s for s in seg_sents if len(s.split(" ")) >= threshold]
    return seg_sents


def embedding(seg_sents):
    embeddings = list()
    for sent in seg_sents:
        input_ids = torch.tensor([tokenizer.encode(sent)])

        with torch.no_grad():
            try:
                features = phobert(input_ids)  # Models outputs are now tuples
                embeddings.append(features[-1].cpu().detach().numpy())
            except:
                continue
    return np.array(embeddings)


def hungarian_algo(x, y, threshold):
    w, h = len(x), len(y)
    x, y = np.squeeze(x, axis=(1,)), np.squeeze(y, axis=(1,))
    sim_matrix = cosine_similarity(x, y)

    matched_data = []
    row_ind, col_ind = linear_sum_assignment(sim_matrix, maximize=True)
    for r, c in zip(row_ind, col_ind):
        if sim_matrix[r, c] >= threshold:
            matched_data.append(
                {"score": sim_matrix[r, c], "row_ind": r, "col_ind": c,}
            )
    score_list = [e["score"] for e in matched_data]
    row_ind = [e["row_ind"] for e in matched_data]
    col_ind = [e["col_ind"] for e in matched_data]
    summary_score = len(score_list) / (max(w, h))
    return summary_score, row_ind, col_ind


#############################
########## GLU CNN ##########
#############################


def shape_transform(x):
    """ Tranform the size of the tensors to fit for conv input. """
    return torch.unsqueeze(torch.transpose(x, 1, 2), 3)


class GatedConv(nn.Module):
    def __init__(self, input_size, width=3, dropout=0.2, nopad=False):
        super(GatedConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=input_size,
            out_channels=2 * input_size,
            kernel_size=(width, 1),
            stride=(1, 1),
            padding=(width // 2 * (1 - nopad), 0),
        )
        init.xavier_uniform_(self.conv.weight, gain=(4 * (1 - dropout)) ** 0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_var):
        x_var = self.dropout(x_var)
        x_var = self.conv(x_var)
        out, gate = x_var.split(int(x_var.size(1) / 2), 1)
        out = out * torch.sigmoid(gate)
        return out


class StackedCNN(nn.Module):
    def __init__(self, num_layers, input_size, cnn_kernel_width=3, dropout=0.2):
        super(StackedCNN, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GatedConv(input_size, cnn_kernel_width, dropout))

    def forward(self, x):
        for conv in self.layers:
            x = x + conv(x)
            x *= SCALE_WEIGHT
        return x


class GLU(nn.Module):
    #
    # Reference: https://arxiv.org/pdf/1711.04168.pdf
    def __init__(self, num_layers, hidden_size, cnn_kernel_width, dropout, input_size):
        super(GLU, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.cnn = StackedCNN(num_layers, hidden_size, cnn_kernel_width, dropout)

    def forward(self, emb):
        emb_reshape = emb.view(emb.size(0) * emb.size(1), -1)
        emb_remap = self.linear(emb_reshape)
        emb_remap = emb_remap.view(emb.size(0), emb.size(1), -1)

        emb_remap = shape_transform(emb_remap)
        out = self.cnn(emb_remap)
        return out.squeeze(3).contiguous()


def glu_pooling(embeddings):
    inputs = torch.from_numpy(embeddings)
    glu = GLU(3, inputs.shape[-1], 5, 0.2, inputs.shape[-1])
    return glu(inputs).cpu().detach().numpy()


if __name__ == "__main__":
    a = html_to_text("a.htm")
    b = html_to_text("b.htm")
    c = html_to_text("c.htm")

    a = segmentize(a, 10)
    b = segmentize(b, 10)
    c = segmentize(c, 10)

    a_embeddings = embedding(a)
    b_embeddings = embedding(b)
    c_embeddings = embedding(c)

    print("========== Hungarian Algorithm ==========")

    results = hungarian_algo(a_embeddings, b_embeddings, 0.85)
    # for i, j in zip(results[1], results[2]):
    #     print(a[i])
    #     print(b[j])
    #     print("============")
    #     print("\n")
    print("HA - a vs b:", results[0])
    results = hungarian_algo(a_embeddings, c_embeddings, 0.85)
    # for i, j in zip(results[1], results[2]):
    #     print(a[i])
    #     print(c[j])
    #     print("============")
    #     print("\n")
    print("HA - a vs c:", results[0])

    print("============= Naive Pooling =============")

    print(
        "MaxPooling - a vs b:",
        cosine_similarity(np.max(a_embeddings, axis=0), np.max(b_embeddings, axis=0)),
    )
    print(
        "MaxPooling - a vs c:",
        cosine_similarity(np.max(a_embeddings, axis=0), np.max(c_embeddings, axis=0)),
    )
    print("=========================================")
    print(
        "MeanPooling - a vs b:",
        cosine_similarity(np.mean(a_embeddings, axis=0), np.mean(b_embeddings, axis=0)),
    )
    print(
        "MeanPooling - a vs c:",
        cosine_similarity(np.mean(a_embeddings, axis=0), np.mean(c_embeddings, axis=0)),
    )

    print("=============  GLU Pooling  =============")
    glu_a = glu_pooling(a_embeddings)
    glu_b = glu_pooling(b_embeddings)
    glu_c = glu_pooling(c_embeddings)
    print(
        "GLUPooling - a vs b:",
        cosine_similarity(np.max(glu_a, axis=0).T, np.max(glu_b, axis=0).T),
    )
    print(
        "GLUPooling - a vs c:",
        cosine_similarity(np.max(glu_a, axis=0).T, np.max(glu_c, axis=0).T),
    )
