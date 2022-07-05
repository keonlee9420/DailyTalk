import torch
import torch.nn as nn
import torch.nn.functional as F

from text import sil_phonemes_ids


class CompTransTTSLoss(nn.Module):
    """ CompTransTTS Loss """

    def __init__(self, preprocess_config, model_config, train_config):
        super(CompTransTTSLoss, self).__init__()
        self.loss_config = train_config["loss"]
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.learn_alignment = model_config["duration_modeling"]["learn_alignment"]
        self.binarization_loss_enable_steps = train_config["duration"]["binarization_loss_enable_steps"]
        self.binarization_loss_warmup_steps = train_config["duration"]["binarization_loss_warmup_steps"]
        self.sum_loss = ForwardSumLoss()
        self.bin_loss = BinLoss()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.sil_ph_ids = sil_phonemes_ids()
        self.var_start_steps = train_config["step"]["var_start_steps"]

    def forward(self, inputs, predictions, step):
        (
            texts,
            _,
            _,
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
            *_,
        ) = inputs[3:]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            attn_outs,
        ) = predictions
        self.src_masks = src_masks = ~src_masks
        mel_masks = ~mel_masks
        if self.learn_alignment:
            attn_soft, attn_hard, attn_hard_dur, attn_logprob = attn_outs
            duration_targets = attn_hard_dur
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        self.mel_masks = mel_masks = mel_masks[:, :mel_masks.shape[1]]

        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        pitch_loss = energy_loss = torch.zeros(1).to(mel_targets.device)
        duration_loss = {
            "pdur": torch.zeros(1).to(mel_targets.device),
            "wdur": torch.zeros(1).to(mel_targets.device),
            "sdur": torch.zeros(1).to(mel_targets.device),
        }

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        ctc_loss = bin_loss = torch.zeros(1).to(mel_targets.device)
        if self.learn_alignment:
            ctc_loss = self.sum_loss(attn_logprob=attn_logprob, in_lens=src_lens, out_lens=mel_lens)
            if step < self.binarization_loss_enable_steps:
                bin_loss_weight = 0.
            else:
                bin_loss_weight = min((step-self.binarization_loss_enable_steps) / self.binarization_loss_warmup_steps, 1.0) * 1.0
            bin_loss = self.bin_loss(hard_attention=attn_hard, soft_attention=attn_soft) * bin_loss_weight

        total_loss = mel_loss + postnet_mel_loss + ctc_loss + bin_loss
        if step >= self.var_start_steps:
            pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
            energy_loss = self.mse_loss(energy_predictions, energy_targets)
            duration_loss = self.get_duration_loss(log_duration_predictions, duration_targets, texts)
            total_loss += sum(duration_loss.values()) + pitch_loss + energy_loss

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            ctc_loss,
            bin_loss,
        )

    def get_duration_loss(self, dur_pred, dur_gt, txt_tokens):
        """
        :param dur_pred: [B, T], float, log scale
        :param txt_tokens: [B, T]
        :return:
        """
        losses = {}
        B, T = txt_tokens.shape
        nonpadding = self.src_masks.float()
        dur_gt = dur_gt.float() * nonpadding
        is_sil = torch.zeros_like(txt_tokens).bool()
        for p_id in self.sil_ph_ids:
            is_sil = is_sil | (txt_tokens == p_id)
        is_sil = is_sil.float()  # [B, T_txt]

        # phone duration loss
        if self.loss_config["dur_loss"] == "mse":
            losses["pdur"] = F.mse_loss(dur_pred, (dur_gt + 1).log(), reduction="none")
            losses["pdur"] = (losses["pdur"] * nonpadding).sum() / nonpadding.sum()
            dur_pred = (dur_pred.exp() - 1).clamp(min=0)
        elif self.loss_config["dur_loss"] == "mog":
            return NotImplementedError
        elif self.loss_config["dur_loss"] == "crf":
            # losses["pdur"] = -self.model.dur_predictor.crf(
            #     dur_pred, dur_gt.long().clamp(min=0, max=31), mask=nonpadding > 0, reduction="mean")
            return NotImplementedError
        losses["pdur"] = losses["pdur"] * self.loss_config["lambda_ph_dur"]

        # use linear scale for sent and word duration
        if self.loss_config["lambda_word_dur"] > 0:
            word_id = (is_sil.cumsum(-1) * (1 - is_sil)).long()
            word_dur_p = dur_pred.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_pred)[:, 1:]
            word_dur_g = dur_gt.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_gt)[:, 1:]
            wdur_loss = F.mse_loss((word_dur_p + 1).log(), (word_dur_g + 1).log(), reduction="none")
            word_nonpadding = (word_dur_g > 0).float()
            wdur_loss = (wdur_loss * word_nonpadding).sum() / word_nonpadding.sum()
            losses["wdur"] = wdur_loss * self.loss_config["lambda_word_dur"]
        if self.loss_config["lambda_sent_dur"] > 0:
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            sdur_loss = F.mse_loss((sent_dur_p + 1).log(), (sent_dur_g + 1).log(), reduction="mean")
            losses["sdur"] = sdur_loss.mean() * self.loss_config["lambda_sent_dur"]
        return losses


class ForwardSumLoss(nn.Module):
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=3)
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(input=attn_logprob, pad=(1, 0), value=self.blank_logprob)

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[: query_lens[bid], :, : key_lens[bid] + 1]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            total_loss += loss

        total_loss /= attn_logprob.shape[0]
        return total_loss


class BinLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hard_attention, soft_attention):
        log_sum = torch.log(torch.clamp(soft_attention[hard_attention == 1], min=1e-12)).sum()
        return -log_sum / hard_attention.sum()
