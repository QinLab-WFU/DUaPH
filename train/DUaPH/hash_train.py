import os
import torch
import time
from tqdm import tqdm
from model.DUaPH import MDUaPH
from train.base import TrainBase
from model.base.optimization import BertAdam
from utils.calc_utils import calc_map_k_matrix as calc_map_k
from .Loss import Criterion
from .get_args import get_args
import numpy as np


def cross_modal_mls(mu_img, sigma_sq_img, mu_txt, sigma_sq_txt):

    distance_img_txt = torch.norm(mu_img - mu_txt, p=2, dim=1) ** 2
    sigma_sum = sigma_sq_img + sigma_sq_txt
    sigma_sum = sigma_sum.unsqueeze(1)
    loss_1 = (distance_img_txt.unsqueeze(-1) ** 2) / sigma_sum
    loss_2 = torch.log(sigma_sq_img + sigma_sq_img + 1)
    loss = loss_1 + loss_2
    return loss.mean()


class DUaPHTrainer(TrainBase):

    def __init__(self):
        args = get_args()
        super(DUaPHTrainer, self).__init__(args)
        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))
        self.run()

    def _init_model(self):
        self.logger.info("init model.")
        self.model = MDUaPH(outputDim=self.args.output_dim, num_classes=self.nclass,
                            clipPath=self.args.clip_path, writer=self.writer,
                            logger=self.logger, is_train=True).to(self.rank)

        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info("load pretrained model.")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"))

        self.model.float()

        to_optim = [
            {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},
            {'params': self.model.image_hash.parameters(), 'lr': self.args.lr},
            {'params': self.model.text_hash.parameters(), 'lr': self.args.lr},
            {'params': self.model.image_pre.parameters(), 'lr': self.args.lr},
            {'params': self.model.text_pre.parameters(), 'lr': self.args.lr},
            {'params': self.model.uncertainty_module_i.parameters(), 'lr': self.args.lr},
            {'params': self.model.uncertainty_module_t.parameters(), 'lr': self.args.lr},
        ]
        self.criterion = Criterion().to(self.rank)
        self.optimizer = BertAdam(to_optim, lr=self.args.lr, warmup=self.args.warmup_proportion,
                                  schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                                  t_total=len(self.train_loader) * self.args.epochs,
                                  weight_decay=self.args.weight_decay, max_grad_norm=1.0)

        self.total_time = 0.0
        print(self.model)

    def train_epoch(self, epoch):
        self.change_state(mode="train")
        self.logger.info(">>>>>> epochs: %d/%d" % (epoch, self.args.epochs))
        all_loss = 0
        for image, text, label, index in tqdm(self.train_loader):
            start_time = time.time()
            image.float()
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            label = label.to(self.rank, non_blocking=True).float()

            image_hash, pre_i, mu_img, sigma_sq_img, text_hash, pre_t, mu_txt, sigma_sq_txt = self.model(image, text)
            i_loss = self.criterion(image_hash, label)
            t_loss = self.criterion(text_hash, label)
            it_loss = self.criterion(image_hash, label, text_hash)
            ti_loss = self.criterion(text_hash, label, image_hash)
            loss = i_loss + t_loss + it_loss + ti_loss
            loss += cross_modal_mls(mu_img, sigma_sq_img, mu_txt, sigma_sq_txt)

            all_loss += loss.data
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.total_time += time.time() - start_time

        self.logger.info(
            f">>>>>> [{epoch}/{self.args.epochs}] loss: {all_loss.data / (len(self.train_loader))}, time: {self.total_time}")

    def valid(self, epoch):
        self.logger.info("Valid.")
        self.change_state(mode="valid")
        query_img, query_txt, q_encoder_time, q_i, q_t = self.get_code(self.query_loader, self.args.query_num)
        retrieval_img, retrieval_txt, r_encoder_time, d_i, d_t = self.get_code(self.retrieval_loader, self.retrieval_num)

        mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)

        if self.max_mapi2t < mAPi2t:
            self.best_epoch_i = epoch
            self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t")
        self.max_mapi2t = max(self.max_mapi2t, mAPi2t)
        if self.max_mapt2i < mAPt2i:
            self.best_epoch_t = epoch
            self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="t2i")
        self.max_mapt2i = max(self.max_mapt2i, mAPt2i)
        self.logger.info(
            f">>>>>> [{epoch}/{self.args.epochs}], MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}, \n \
                            MAX MAP(i->t): {self.best_epoch_i}:{self.max_mapi2t}, MAX MAP(t->i): {self.best_epoch_t}:{self.max_mapt2i}, \
                            query_encoder_time: {q_encoder_time}, retrieval_encoder_time: {r_encoder_time}")
        self.logger.info(
            f">>>>>>> Best epoch, I-T: {self.best_epoch_i}, mAP: {self.max_mapi2t}, T-I: {self.best_epoch_t}, mAP: {self.max_mapt2i}")

    def get_code(self, data_loader, length: int):
        img_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        text_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        encoder_time = 0
        outputs_i = []
        outputs_t = []
        for image, text, label, index in tqdm(data_loader):
            with torch.no_grad():
                start_encoder_time = time.time()
                image = image.to(self.rank, non_blocking=True)
                text = text.to(self.rank, non_blocking=True)
                index = index.numpy()
                image_hash, pre_i, mu_img, sigma_sq_img = self.model.encode_image(image)
                image_hash = torch.sign(image_hash)
                text_hash, pre_t, mu_txt, sigma_sq_txt = self.model.encode_text(text)
                text_hash = torch.sign(text_hash)
                encoder_time = time.time() - start_encoder_time
                img_buffer[index, :] = image_hash.data
                text_buffer[index, :] = text_hash.data
        return img_buffer, text_buffer, encoder_time, outputs_i, outputs_t

    def zero2eps(self, x):
        x[x == 0] = 1
        return x

    def normalize(self, affinity):
        col_sum = self.zero2eps(np.sum(affinity, axis=1)[:, np.newaxis])
        row_sum = self.zero2eps(np.sum(affinity, axis=0))
        out_affnty = affinity / col_sum
        in_affnty = np.transpose(affinity / row_sum)
        return in_affnty, out_affnty

    def affinity_tag_multi(self, tag1: np.ndarray, tag2: np.ndarray):

        aff = np.matmul(tag1, tag2.T)
        affinity_matrix = np.float32(aff)

        affinity_matrix = 1 / (1 + np.exp(-affinity_matrix))
        affinity_matrix = 2 * affinity_matrix - 1
        in_aff, out_aff = self.normalize(affinity_matrix)

        return in_aff, out_aff, affinity_matrix
