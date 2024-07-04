import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from batch_idx_sel import batched_index_select
from torchvision.models.resnet import ResNet18_Weights

# import torch_resnet #po co to było?
# from nystrom_attention import NystromAttention
# from monai.networks.nets import EfficientNetBN


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Attn_Net(nn.Module):
    # """
    #     Attention Network without Gating (2 fc layers)
    #     args:
    #         L: input feature dimension
    #         D: hidden layer dimension
    #         dropout: whether to use dropout (p = 0.25)
    #         n_classes: number of classes
    # """

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        A = self.module(x)
        return A, x  # N x n_classes


class Attn_Net_Gated(nn.Module):
    #   """
    #         Attention Network with Sigmoid Gating (3 fc layers)
    #         args:
    #             L: input feature dimension
    #             D: hidden layer dimension
    #             dropout: whether to use dropout (p = 0.25)
    #             n_classes: number of classes
    # """
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


# -----------------------GATED MIL---------------------------------
class AttentionMIL(nn.Module):

    def __init__(
                self,
                num_classes=1,
                gated=False,
                pretrained=True,
                dropout=False,
                size_arg="big"):

        super().__init__()
        self.size_dict = {"medium": [512, 256, 128], "big": [512, 256, 256],
                          "tiny": [128, 64, 64], "small": [256, 128, 64]}
        size = self.size_dict[size_arg]
        self.num_features = size[0]
        self.is_pe = False
        self.cat = False
        self.A = None

        # size big
        if size_arg == 'medium': # 11.2M params // feature extractor
            if pretrained:
                self.feature_extractor = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            else:
                self.feature_extractor = models.resnet18()
            # self.num_features = self.feature_extractor.fc.in_features
            self.feature_extractor.fc = Identity()
        # size small
        elif size_arg == 'small': # 2.2M params // feature extractor
        # use 6 blocks (output -1, 256, 3, 3 before adaptive avg pool for 33x33 input)
        # use 6 blocks (output -1, 256, 2, 2 before adaptive avg pool for 32x32 input)
            self.feature_extractor = nn.Sequential(*list(models.resnet18(weights=ResNet18_Weights.DEFAULT).children())[:-3],
                                                nn.AdaptiveAvgPool2d(1), nn.Flatten())
        elif size_arg == 'tiny': # 683k params // feature extractor
        # (output -1, 128, 5, 5 before adaptive avg pool for 33x33 input)
        # (output -1, 128, 4, 4 before adaptive avg pool for 32x32 input)
            self.feature_extractor = nn.Sequential(*list(models.resnet18(weights=ResNet18_Weights.DEFAULT).children())[:-4],
                                                nn.AdaptiveAvgPool2d(1), nn.Flatten())

        self.patch_extractor = nn.Sequential(nn.Linear(size[0], size[1]),
                                             nn.ReLU())
        self.positional_encoder = PositionalEncoding(d=self.num_features,
                                                     dropout=0., max_len=64)

        if self.cat and self.is_pe:
            self.L *= 2

        if gated:
            self.attention_net = Attn_Net_Gated(
                L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            self.attention_net = Attn_Net(
                L=size[1], D=size[2], dropout=dropout, n_classes=1)
        self.classifier = nn.Sequential(nn.Linear(size[1],
                                                  num_classes))

    def forward(self, x, position_coords=None):
        # x: bs x N x C x W x W
        bs, num_instances, ch, w, h = x.shape
        x = x.view(bs*num_instances, ch, w, h)  # x: N bs x C x W x W
        H = self.feature_extractor(x)  # x: N bs x C' x W' x W'
        H = self.patch_extractor(H)  # added ~dim Nbs
        H = H.view(bs, num_instances, -1)
        A, H = self.attention_net(H)
        A = torch.transpose(A, 2, 1)  # added ~dim bsxKxN 10
        A = F.softmax(A, dim=2)  # ~ensure weights sum up  to unity ~dim bsxKxN

        if self.is_pe:
            H = H.squeeze(0)
            H = self.positional_encoder(H, position_coords,
                                        self.cat)  # N x NUM_FEATS(2)
            H = H.unsqueeze(0)

        m = torch.matmul(A, H)  # added ~dim bsxKxN ~attention pooling

        Y = self.classifier(m)  # added
        # print(Y)
        self.A = A
        return Y#, A dodany self
# -----------------------------------------------------------------


# -------------------Dual Stream MIL-------------------------------
class DSMIL(nn.Module):

    def __init__(
                self,
                num_classes=1,
                pretrained=True,
                nonlinear=False,
                dropout=False,
                size_arg="small"):

        super().__init__()

        # self.D = self.num_features
        self.size_dict = {"small": [512, 256, 128], "big": [512, 256, 256]}
        size = self.size_dict[size_arg]
        self.Q_dim = 128
        self.is_pe = False
        self.cat = False

        self.feature_extractor = models.resnet18(pretrained=pretrained)
        self.num_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = Identity()

        self.positional_encoder = PositionalEncoding(d=self.num_features,
                                                     dropout=0., max_len=64)

        self.IC_fc = nn.Linear(size[0], num_classes)

        if nonlinear:
            self.lin = nn.Sequential(nn.Linear(size[0],
                                               size[1]),
                                     nn.ReLU())

            self.q = nn.Sequential(nn.Linear(size[1], size[2]),
                                   nn.Tanh())
        # else:
        #     self.lin = nn.Identity()
        #     self.q = nn.Linear(self.num_features, self.Q_dim)

        if dropout:
            d = 0.25
        else:
            d = 0.
        self.v = nn.Sequential(nn.Dropout(d),
                               nn.Linear(size[1], size[1]))

        self.fcc = nn.Conv1d(num_classes, num_classes,
                             kernel_size=size[1])

    def forward(self, x, position_coords=None):
        # bs-batch, N-num_instances, K-fts_per_inst CH-channels W/H-wdth/hght
        bs, num_instances, ch, h, w = x.shape
        device = x.device

        x = x.view(bs*num_instances, ch, h, w)  # x: N bs x CH x W x W
        feats = self.feature_extractor(x)  # feats: N bs x CH' x W' x W'
        feats = feats.view(bs, num_instances, -1)  # bs x N x num_feats
        c = self.IC_fc(feats)  # bs x N x C

        feats = self.lin(feats)  # bs x N x num_feats

        if self.is_pe:
            feats = feats.squeeze(0)  # N x num_feats
            feats = self.positional_encoder(feats, position_coords)  # N x n_ft
            feats = feats.unsqueeze(0)  # bs x N x num_feats

        V = self.v(feats)  # bs x N x V
        Q = self.q(feats).view(bs, num_instances, -1)  # bs x N x Q

        _, m_indices = torch.sort(c, 1, descending=True)  # bs x N x C

        # bs x C x K
        m_feats = batched_index_select(feats, dim=1, index=m_indices[:, 0, :])
        q_max = self.q(m_feats)  # bs x C x Q

        A = torch.matmul(Q, q_max.transpose(1, 2))  # bs x N x C
        # A = F.softmax(A, dim=1)
        A = F.softmax(A / torch.sqrt(torch.tensor(
            Q.shape[2], dtype=torch.float32, device=device)), dim=1)

        B = torch.matmul(A.transpose(1, 2), V)  # bs x C x V
        # B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V

        C = self.fcc(B)  # bs x C x 1
        C = C.view(bs, -1)  # bs x C

        Y = C

        return Y, A, B, c
# -----------------------------------------------------------------


# ----------------------SIMPLE MIL---------------------------------
class SimpleMIL(nn.Module):

    def __init__(
                self,
                num_instances=100,
                num_classes=1,
                pretrained=True):

        super().__init__()
        self.L = 512  # to ma z resneta wychodzić AMP2d
        self.D = 256  # att inner dim
        self.K = 1    # minimum patchy?
        self.num_instances = num_instances

        self.feature_extractor = models.resnet18(pretrained=pretrained)
        self.num_features = self.feature_extractor.fc.in_features  # selfdod
        self.feature_extractor.fc = Identity()
        # self.feature_extractor.avgpool = Identity()
        self.patch_extractor = nn.Sequential(  # nn.AdaptiveMaxPool2d(1),
                                             nn.Linear(self.num_features,
                                                       self.L),
                                             nn.ReLU())  # added

        self.attention = nn.Sequential(nn.Linear(self.L, self.D),
                                       nn.Tanh(),
                                       nn.Linear(self.D, self.K))  # added

        self.classifier = nn.Sequential(nn.Linear(self.L * self.K,
                                                  num_classes))  # added

    def forward(self, x):
        # x: bs x N x C x W x W
        bs, _, ch, w, h = x.shape
        x = x.view(bs*self.num_instances, ch, w, h)  # x: N bs x C x W x W

        H = self.feature_extractor(x)  # x: N bs x C' x W' x W'
        H = self.patch_extractor(H)  # added ~dim Nbs

        A = self.attention(H)  # added ~dim NbsxK
        A = torch.transpose(A, 1, 0)  # added ~dim KxN 10
        A = F.softmax(A, dim=1)  # ~ensure weights sum up  to unity ~dim KxN

        z = torch.mm(A, H)  # added ~dim KxN ~attention pooling
        Y = self.classifier(z)  # added
        return Y, A
# -----------------------------------------------------------------


# ------------------------MA-MIL-----------------------------------
class MultiAttentionMIL(nn.Module):
    def __init__(self, num_classes, pretrained,
                 use_dropout=False, n_dropout=0.4):

        super(MultiAttentionMIL, self).__init__()
        self.num_classes = num_classes
        self.use_dropout = use_dropout
        self.n_dropout = n_dropout

        self.D = 128

        self.feature_extractor = models.resnet18(pretrained=pretrained)
        self.num_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = Identity()

        self.fc1 = nn.Sequential(
            nn.Linear(self.num_features, self.D),
            nn.ReLU(),
        )
        self.attention1 = nn.Sequential(
            nn.Linear(self.D, self.D), nn.Tanh(), nn.Linear(self.D, 1)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.ReLU(),
        )
        self.attention2 = nn.Sequential(
            nn.Linear(self.D, self.D), nn.Tanh(), nn.Linear(self.D, 1)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.ReLU(),
        )
        self.attention3 = nn.Sequential(
            nn.Linear(self.D, self.D), nn.Tanh(), nn.Linear(self.D, 1)
        )

        self.fc4 = nn.Sequential(nn.Linear(self.D, self.num_classes))

    def forward(self, x):
        ################
        x1 = x.squeeze(0)
        x1 = self.feature_extractor(x1)
        x1 = self.fc1(x1)
        if self.use_dropout:
            x1 = nn.Dropout(self.n_dropout)(x1)
        # -------
        a1 = self.attention1(x1)
        a1 = torch.transpose(a1, 1, 0)
        a1 = nn.Softmax(dim=1)(a1)
        # -------
        m1 = torch.mm(a1, x1)
        m1 = m1.view(-1, 1 * self.D)

        ################
        x2 = self.fc2(x1)
        if self.use_dropout:
            x2 = nn.Dropout(self.n_dropout)(x2)
        # -------
        a2 = self.attention2(x2)
        a2 = torch.transpose(a2, 1, 0)
        a2 = nn.Softmax(dim=1)(a2)
        # -------
        m2 = torch.mm(a2, x2)
        m2 = m2.view(-1, 1 * self.D)
        m2 += m1

        ################
        x3 = self.fc3(x2)
        if self.use_dropout:
            x3 = nn.Dropout(self.n_dropout)(x3)
        # -------
        a3 = self.attention3(x3)
        a3 = torch.transpose(a3, 1, 0)
        a3 = nn.Softmax(dim=1)(a3)
        # -------
        m3 = torch.mm(a3, x3)
        m3 = m3.view(-1, 1 * self.D)
        m3 += m2

        result = self.fc4(m3)

        return result, a1, a2, a3
# -----------------------------------------------------------------


# ------------------------GMA-MIL----------------------------------
class GatedMultiAttentionMIL(nn.Module):
    def __init__(self, num_classes, pretrained,
                 use_dropout=True, n_dropout=0.4):

        super(GatedMultiAttentionMIL, self).__init__()
        self.num_classes = num_classes
        self.use_dropout = use_dropout
        self.n_dropout = n_dropout
        # self.n_dropout = 0.2

        self.D = 128
        # self.feature_extractor = EfficientNetBN("efficientnet-b0",
        #                                         num_classes=1)
        # self.num_features = self.feature_extractor._fc.in_features
        # self.feature_extractor._fc = Identity()

        self.feature_extractor = models.resnet18(pretrained=pretrained)
        self.num_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = Identity()

        self.fc1 = nn.Sequential(
            nn.Linear(self.num_features, self.D),
            nn.ReLU()
        )
        self.attention_V1 = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.Tanh()
        )
        self.attention_U1 = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.Sigmoid()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.ReLU()
        )
        self.attention_V2 = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.Tanh(),
        )
        self.attention_U2 = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.Sigmoid()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.ReLU(),
        )
        self.attention_V3 = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.Tanh(),
        )
        self.attention_U3 = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.Sigmoid()
        )

        self.attention_weights1 = nn.Linear(self.D, 1)
        self.attention_weights2 = nn.Linear(self.D, 1)
        self.attention_weights3 = nn.Linear(self.D, 1)

        self.fc4 = nn.Sequential(nn.Linear(self.D, self.num_classes))

    def forward(self, x):
        ################
        x1 = x.squeeze(0)
        x1 = self.feature_extractor(x1)
        x1 = self.fc1(x1)
        if self.use_dropout:
            x1 = nn.Dropout(self.n_dropout)(x1)
        # -------
        A_V1 = self.attention_V1(x1)  # N x D
        A_U1 = self.attention_U1(x1)  # N x D
        A1 = self.attention_weights1(torch.mul(A_V1, A_U1))
        A1 = torch.transpose(A1, 1, 0)
        A1 = F.softmax(A1, dim=1)  # ~ensure weights sum up  to unity
        m1 = torch.mm(A1, x1)  # ~attention pooling
        m1 = m1.view(-1, 1 * self.D)

        ################
        x2 = self.fc2(x1)
        if self.use_dropout:
            x2 = nn.Dropout(self.n_dropout)(x2)
        # -------
        A_V2 = self.attention_V2(x2)  # N x D
        A_U2 = self.attention_U2(x2)  # N x D
        A2 = self.attention_weights2(torch.mul(A_V2, A_U2))
        A2 = torch.transpose(A2, 1, 0)
        A2 = F.softmax(A2, dim=1)  # ~ensure weights sum up  to unity
        m2 = torch.mm(A2, x2)  # ~attention pooling
        m2 = m2.view(-1, 1 * self.D)
        m2 += m1
        ################
        x3 = self.fc3(x2)
        if self.use_dropout:
            x3 = nn.Dropout(self.n_dropout)(x3)
        # -------
        A_V3 = self.attention_V3(x3)  # N x D
        A_U3 = self.attention_U3(x3)  # N x D
        A3 = self.attention_weights3(torch.mul(A_V3, A_U3))
        A3 = torch.transpose(A3, 1, 0)
        A3 = F.softmax(A3, dim=1)  # ~ensure weights sum up  to unity
        m3 = torch.mm(A3, x3)  # ~attention pooling
        m3 = m3.view(-1, 1 * self.D)
        m3 += m2

        result = self.fc4(m3)

        return result, A1, A2, A3
# -----------------------------------------------------------------


# -----------------------TO DO---------------------------------
class PositionalEncoding(nn.Module):
    """Positional encoding.
       d - dimension of the output embedding space
       """
    def __init__(self, d, dropout, max_len=1000):
        super().__init__()
        self.d = d
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((max_len, max_len, self.d))

        # temperature is user-defined scalar
        # (set to 10k by authors of Attention is all You Need)
        temperature = 10_000

        # i,j is an integer in [0, d/4),
        # where d is the size of the ch dimension
        # half channels encoded with pos_x and second half with pos_y ??
        i = torch.arange(0, self.d, 4, dtype=torch.float32)
        j = torch.arange(0, self.d, 4, dtype=torch.float32)

        # x, y is an integer in [0, max_len],
        # where max_len is long enough
        # (x,y) is a point in 2d space
        x = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
        y = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)

        pos_x = x / torch.pow(temperature, i / d)
        pos_y = y / torch.pow(temperature, j / d)

        # P - matrix of postional encodings
        self.P[:, :, 0:self.d//2:2] = torch.sin(pos_x)
        self.P[:, :, 1:self.d//2:2] = torch.cos(pos_x)
        self.P[:, :, (0+self.d//2)::2] = torch.sin(pos_y)
        self.P[:, :, (1+self.d//2)::2] = torch.cos(pos_y)
        # self.register_buffer('P-enc', self.P)
        '''
        PE(x,y,2i) = sin(x/10000^(4i/D))
        PE(x,y,2i+1) = cos(x/10000^(4i/D))
        PE(x,y,2j+D/2) = sin(y/10000^(4j/D))
        PE(x,y,2j+1+D/2) = cos(y/10000^(4j/D))
        '''
    def forward(self, X, position_coords, cat):
        position_encodings = torch.zeros(X.shape[0], self.d)
        for i in range(X.shape[0]):
            position_coords = position_coords.squeeze(0)  # remove batch dim
            position_encodings[i] = self.P[position_coords[i][0],
                                           position_coords[i][1]]
        if cat:
            X = torch.cat((X, position_encodings.to(X.device)), dim=1)
        else:
            X = X + position_encodings.to(X.device)
        return self.dropout(X)

# -----------------------------------------------------------------


class APE_SAMIL(nn.Module):
    def __init__(self, num_classes, pretrained,
                 use_dropout=False, n_dropout=0.4):

        super(APE_SAMIL, self).__init__()
        self.num_classes = num_classes
        self.use_dropout = use_dropout
        self.n_dropout = n_dropout
        self.gated = True
        self.is_pe = False
        self.cat = False
        self.bias = False
        self.D = 128  # num hiddens
        self.K = 1

        self.feature_extractor = models.resnet18(pretrained=pretrained)
        self.num_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = Identity()

        # for larger images or more overlaps set max_len appropiately
        self.positional_encoder = PositionalEncoding(d=self.num_features,
                                                     dropout=0., max_len=64)
        if self.cat and self.is_pe:
            self.num_features *= 2
            # self.D *= 2

        self.fc_q = nn.Sequential(
            nn.Linear(self.num_features, self.D, self.bias),
            nn.ReLU()
        )
        self.fc_k = nn.Sequential(
            nn.Linear(self.num_features, self.D, self.bias),
            nn.ReLU()
        )
        self.fc_v = nn.Sequential(
            nn.Linear(self.num_features, self.D, self.bias),
            nn.ReLU()
        )

        if self.gated:
            self.attention_V = nn.Sequential(
                nn.Linear(self.D, self.D),
                nn.Tanh()
            )

            self.attention_U = nn.Sequential(
                nn.Linear(self.D, self.D),
                nn.Sigmoid()
            )
            self.attention_weights = nn.Linear(self.D, self.K)

        else:
            self.attention = nn.Sequential(nn.Linear(self.D, self.D),
                                           nn.Tanh(),
                                           nn.Linear(self.D, self.K)
                                           )

        self.classifier = nn.Linear(self.D*self.K, num_classes)

    def forward(self, x, position_coords=None):

        device = x.device
        # feature extraction
        x = x.squeeze(0)  # N x CH x H x W
        H = self.feature_extractor(x)  # N x NUM_FEATS
        # positional encoding
        if self.is_pe:
            H = self.positional_encoder(H, position_coords,
                                        self.cat)  # N x NUM_FEATS(2)

        # self-attention
        Q = self.fc_q(H)  # N x D
        K = self.fc_k(H)  # N x D
        V = self.fc_v(H)  # N x D

        # Q = self.positional_encoder(Q, position_coords,
        #                             self.cat)
        # K = self.positional_encoder(K, position_coords,
        #                             self.cat)
        # V = self.positional_encoder(V, position_coords,
        #                             self.cat)

        sA = torch.mm(Q, K.transpose(1, 0))  # N x N
        sA = F.softmax(sA / torch.sqrt(torch.tensor(
            Q.shape[1], dtype=torch.float32, device=device)), dim=1)  # N x N
        AV = torch.mm(sA, V)  # N x D

        # attention pooling
        if self.gated:
            A_V = self.attention_V(AV)  # N x D
            A_U = self.attention_U(AV)  # N x D
            A = self.attention_weights(torch.mul(A_V, A_U))  # N x K
        else:
            A = self.attention(AV)  # N x K

        A = A.transpose(1, 0)  # K x N
        A = F.softmax(A, dim=1)  # K x N
        M = torch.mm(A, AV)  # K x D
        M = M.view(1, -1)  # KD x 1

        # classifier
        y = self.classifier(M)  # 1 x 1
        return y, A, sA


# -----------------------------------------------------------------
# ---------------------------CLAM----------------------------------

class CLAM_SB(nn.Module):
    """
    args:
        gate: whether to use gated attention network
        size_arg: config for network size
        dropout: whether to use dropout
        k_sample: number of positive/neg patches to sample
            for instance-level training
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes
        instance_loss_fn: loss function to supervise instance-level training
        subtyping: whether it's a subtyping problem
    """
    def __init__(self, gate=True, size_arg="small", dropout=False,
                 k_sample=8, num_classes=2, pretrained=True,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=True):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [512, 256, 128], "big": [512, 256, 256]}
        size = self.size_dict[size_arg]

        self.feature_extractor = models.resnet18(pretrained=pretrained)
        self.num_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = Identity()

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(
                L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(
                L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], num_classes)
        instance_classifiers = [
            nn.Linear(size[1], 2) for i in range(num_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = num_classes
        self.subtyping = subtyping

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, x, label=None, instance_eval=False):
        return_features = False
        attention_only = False

        # x: bs x N x C x W x W
        x = x.squeeze(0)  # x: N x C x W x H
        H = self.feature_extractor(x)  # H: N x num_feats (Nx512)
        # H = H.view(bs, num_instances, -1)

        A, H = self.attention_net(H)  # A: N x 1   H: N x 512
        A = torch.transpose(A, 1, 0)  # 1 x N
        if attention_only:
            return A
        # A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(
                label.long(), num_classes=self.n_classes+1).squeeze()
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(
                        A, H, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(
                            A, H, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, H)
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {
                'instance_loss': total_inst_loss, 'inst_labels': np.array(
                    all_targets),
                'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A, results_dict


class CLAM_MB(CLAM_SB):
    def __init__(self, gate=True, size_arg="small", dropout=False,
                 k_sample=8, num_classes=2, pretrained=True,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        nn.Module.__init__(self)
        self.size_dict = {"small": [512, 256, 128], "big": [512, 256, 256]}
        size = self.size_dict[size_arg]

        self.feature_extractor = models.resnet18(pretrained=pretrained)
        self.num_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = Identity()
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(
                L=size[1], D=size[2], dropout=dropout, n_classes=num_classes)
        else:
            attention_net = Attn_Net(
                L=size[1], D=size[2], dropout=dropout, n_classes=num_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        # use an indepdent linear layer to predict each class
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(num_classes)]
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [
            nn.Linear(size[1], 2) for i in range(num_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = num_classes
        self.subtyping = subtyping

    def forward(self, x, label=None, instance_eval=False):
        return_features = False
        attention_only = False

        device = x.device
        # x: bs x N x C x W x W
        x = x.squeeze(0)  # x: N x C x W x W
        # bs, num_instances, ch, w, h = x.shape
        # x = x.view(bs*num_instances, ch, w, h)  # x: N bs x C x W x W
        H = self.feature_extractor(x)  # x: N bs x C' x W' x W'
        # H = H.view(bs, num_instances, -1)

        A, H = self.attention_net(H)

        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        # A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(
                label.long(), num_classes=self.n_classes+1).squeeze()
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(
                        A[i], H, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(
                            A[i], H, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
        M = torch.mm(A, H)

        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {
                'instance_loss': total_inst_loss, 'inst_labels': np.array(
                    all_targets),
                'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A, results_dict
