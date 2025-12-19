import logging
import torch
from torch import nn
from model.modelbase import Baseclip, Pre_Layer


class UncertaintyModule(nn.Module):
    def __init__(self, embedding_size):
        super(UncertaintyModule, self).__init__()
        self.fc_mu = nn.Linear(embedding_size, embedding_size)
        self.fc_log_sigma_sq = nn.Linear(embedding_size, embedding_size)
        self.batch_norm_mu = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.995)
        self.batch_norm_sigma = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.995, affine=False)

        self.gamma = nn.Parameter(torch.tensor(1e-4))
        self.beta = nn.Parameter(torch.tensor(-7.0))

    def forward(self, x):
        mu = self.batch_norm_mu(self.fc_mu(x))

        log_sigma_sq = self.batch_norm_sigma(self.fc_log_sigma_sq(x))
        log_sigma_sq = self.gamma * log_sigma_sq + self.beta
        sigma_sq = 1e-6 + torch.exp(log_sigma_sq)
        return mu, sigma_sq


class MDUaPH(Baseclip):

    def __init__(self,
                 outputDim=64,
                 num_classes=None,
                 uncer=True,
                 clipPath="./ViT-B-32.pt",
                 writer=None,
                 saveDir="./result/log",
                 logger: logging.Logger = None,
                 is_train=True):
        super(MDUaPH, self).__init__(outputDim=outputDim, clipPath=clipPath, writer=writer,
                                    saveDir=saveDir, logger=logger, is_train=is_train)
        if num_classes is not None:
            self.classifier = True
            self.image_pre = Pre_Layer(inputdim=outputDim, nb_class=num_classes)
            self.text_pre = Pre_Layer(inputdim=outputDim, nb_class=num_classes)
        else:
            self.classifier = False
        if uncer:
            self.uncertainty_module_i = UncertaintyModule(outputDim)
            self.uncertainty_module_t = UncertaintyModule(outputDim)
        self.uncer = uncer

    def encode_image(self, image):
        image_embed = self.clip.encode_image(image)
        image_hash = self.image_hash(image_embed)
        img_cls, mu_img, sigma_sq_img = None, None, None
        if self.classifier:
            img_cls = self.image_pre(image_hash)
        if self.uncer:
            mu_img, sigma_sq_img = self.uncertainty_module_i(image_hash)
        return image_hash, img_cls, mu_img, sigma_sq_img

    def encode_text(self, text):
        text_embed = self.clip.encode_text(text)
        text_hash = self.text_hash(text_embed)
        txt_cls, mu_txt, sigma_sq_txt = None, None, None
        if self.classifier:
            txt_cls = self.text_pre(text_hash)
        if self.uncer:
            mu_txt, sigma_sq_txt = self.uncertainty_module_t(text_hash)
        return text_hash, txt_cls, mu_txt, sigma_sq_txt

    def forward(self, image, text):
        image_hash, img_cls, mu_img, sigma_sq_img = self.encode_image(image)
        text_hash, txt_cls, mu_txt, sigma_sq_txt = self.encode_text(text)
        return image_hash, img_cls, mu_img, sigma_sq_img, text_hash, txt_cls, mu_txt, sigma_sq_txt
