import json
import math
import os

import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import get_variance_level, pad_1D, pad_2D, pad_3D


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, model_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.raw_path = preprocess_config["path"]["raw_path"]
        self.sub_dir_name = preprocess_config["path"]["sub_dir_name"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.learn_alignment = model_config["duration_modeling"]["learn_alignment"]
        self.load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'
        self.load_emotion = model_config["multi_emotion"]
        self.history_type = model_config["history_encoder"]["type"]
        self.text_emb_size = model_config["history_encoder"]["text_emb_size"]
        self.max_history_len = model_config["history_encoder"]["max_history_len"]

        self.pitch_level_tag, self.energy_level_tag, *_ = get_variance_level(preprocess_config, model_config)

        self.basename, self.speaker, self.text, self.raw_text, self.emotion = self.process_meta(
            filename
        )
        self.basename_to_id = dict((v, k) for k, v in enumerate(self.basename))
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        with open(os.path.join(self.preprocessed_path, "emotions.json")) as f:
            self.emotion_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        emotion_id = self.emotion_map[self.emotion[idx]] if self.load_emotion else None
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel_{}".format(self.pitch_level_tag),
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch_{}".format(self.pitch_level_tag),
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy_{}".format(self.energy_level_tag),
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        if self.learn_alignment:
            attn_prior_path = os.path.join(
                self.preprocessed_path,
                "attn_prior",
                "{}-attn_prior-{}.npy".format(speaker, basename),
            )
            attn_prior = np.load(attn_prior_path)
            duration = None
        else:
            duration_path = os.path.join(
                self.preprocessed_path,
                "duration",
                "{}-duration-{}.npy".format(speaker, basename),
            )
            duration = np.load(duration_path)
            attn_prior = None
        spker_embed = np.load(os.path.join(
            self.preprocessed_path,
            "spker_embed",
            "{}-spker_embed.npy".format(speaker),
        )) if self.load_spker_embed else None

        # History
        dialog = basename.split("_")[2].strip("d")
        turn = int(basename.split("_")[0])
        history_len = min(self.max_history_len, turn)
        history_text = list()
        history_text_emb = list()
        history_text_len = list()
        history_pitch = list()
        history_energy = list()
        history_duration = list()
        history_emotion = list()
        history_speaker = list()
        history_mel_len = list()
        history = None
        if self.history_type != "none":
            history_basenames = sorted([tg_path.replace(".wav", "") for tg_path in os.listdir(os.path.join(self.raw_path, self.sub_dir_name, f"{dialog}")) if ".wav" in tg_path], key=lambda x:int(x.split("_")[0]))
            history_basenames = history_basenames[:turn][-history_len:]

            if self.history_type == "Guo":
                text_emb_path = os.path.join(
                    self.preprocessed_path,
                    "text_emb",
                    "{}-text_emb-{}.npy".format(speaker, basename),
                )
                text_emb = np.load(text_emb_path)

                for i, h_basename in enumerate(history_basenames):
                    h_idx = int(self.basename_to_id[h_basename])
                    h_speaker = self.speaker[h_idx]
                    h_speaker_id = self.speaker_map[h_speaker]
                    h_text_emb_path = os.path.join(
                        self.preprocessed_path,
                        "text_emb",
                        "{}-text_emb-{}.npy".format(h_speaker, h_basename),
                    )
                    h_text_emb = np.load(h_text_emb_path)
                    
                    history_text_emb.append(h_text_emb)
                    history_speaker.append(h_speaker_id)

                    # Padding
                    if i == history_len-1 and history_len < self.max_history_len:
                        self.pad_history(
                            self.max_history_len-history_len,
                            history_text_emb=history_text_emb,
                            history_speaker=history_speaker,
                        )
                if turn == 0:
                    self.pad_history(
                        self.max_history_len,
                        history_text_emb=history_text_emb,
                        history_speaker=history_speaker,
                    )

                history = {
                    "text_emb": text_emb,
                    "history_len": history_len,
                    "history_text_emb": history_text_emb,
                    "history_speaker": history_speaker,
                }

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "attn_prior": attn_prior,
            "spker_embed": spker_embed,
            "emotion": emotion_id,
            "history": history,
        }

        return sample

    def pad_history(self, 
            pad_size,
            history_text=None,
            history_text_emb=None,
            history_text_len=None,
            history_pitch=None,
            history_energy=None,
            history_duration=None,
            history_emotion=None,
            history_speaker=None,
            history_mel_len=None,
        ):
        for _ in range(pad_size):
            history_text.append(np.zeros(1, dtype=np.int64)) if history_text is not None else None
            history_text_emb.append(np.zeros(self.text_emb_size, dtype=np.float32)) if history_text_emb is not None else None
            history_text_len.append(0) if history_text_len is not None else None # meaningless zero padding, should be cut out by mask of history_len
            history_pitch.append(np.zeros(1, dtype=np.float64)) if history_pitch is not None else None
            history_energy.append(np.zeros(1, dtype=np.float32)) if history_energy is not None else None
            history_duration.append(np.zeros(1, dtype=np.float64)) if history_duration is not None else None
            history_emotion.append(0) if history_emotion is not None else None # meaningless zero padding, should be cut out by mask of history_len
            history_speaker.append(0) if history_speaker is not None else None # meaningless zero padding, should be cut out by mask of history_len
            history_mel_len.append(0) if history_mel_len is not None else None # meaningless zero padding, should be cut out by mask of history_len

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            emotion = []
            for line in f.readlines():
                if self.load_emotion:
                    n, s, t, r, e = line.strip("\n").split("|")
                else:
                    n, s, t, r, *_ = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
                if self.load_emotion:
                    emotion.append(e)
            return name, speaker, text, raw_text, emotion

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs] if not self.learn_alignment else None
        attn_priors = [data[idx]["attn_prior"] for idx in idxs] if self.learn_alignment else None
        spker_embeds = np.concatenate(np.array([data[idx]["spker_embed"] for idx in idxs]), axis=0) \
            if self.load_spker_embed else None
        emotions = np.array([data[idx]["emotion"] for idx in idxs]) if self.load_emotion else None

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        if self.learn_alignment:
            attn_priors = pad_3D(attn_priors, len(idxs), max(text_lens), max(mel_lens))
        else:
            durations = pad_1D(durations)

        history_info = None
        if self.history_type != "none":
            if self.history_type == "Guo":
                text_embs = [data[idx]["history"]["text_emb"] for idx in idxs]
                history_lens = [data[idx]["history"]["history_len"] for idx in idxs]
                history_text_embs = [data[idx]["history"]["history_text_emb"] for idx in idxs]
                history_speakers = [data[idx]["history"]["history_speaker"] for idx in idxs]

                text_embs = np.array(text_embs)
                history_lens = np.array(history_lens)
                history_text_embs = np.array(history_text_embs)
                history_speakers = np.array(history_speakers)

                history_info = (
                    text_embs,
                    history_lens,
                    history_text_embs,
                    history_speakers,
                )

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
            attn_priors,
            spker_embeds,
            emotions,
            history_info,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config, model_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.raw_path = preprocess_config["path"]["raw_path"]
        self.sub_dir_name = preprocess_config["path"]["sub_dir_name"]
        self.load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'
        self.load_emotion = model_config["multi_emotion"]
        self.history_type = model_config["history_encoder"]["type"]
        self.text_emb_size = model_config["history_encoder"]["text_emb_size"]
        self.max_history_len = model_config["history_encoder"]["max_history_len"]

        self.basename, self.speaker, self.text, self.raw_text, self.emotion = self.process_meta(
            filepath
        )
        self.basename_to_id = dict((v, k) for k, v in enumerate(self.basename))
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        with open(os.path.join(self.preprocessed_path, "emotions.json")) as f:
            self.emotion_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        emotion_id = self.emotion_map[self.emotion[idx]] if self.load_emotion else None
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        spker_embed = np.load(os.path.join(
            self.preprocessed_path,
            "spker_embed",
            "{}-spker_embed.npy".format(speaker),
        )) if self.load_spker_embed else None

        # History
        dialog = basename.split("_")[2].strip("d")
        turn = int(basename.split("_")[0])
        history_len = min(self.max_history_len, turn)
        history_text = list()
        history_text_emb = list()
        history_text_len = list()
        history_pitch = list()
        history_energy = list()
        history_duration = list()
        history_emotion = list()
        history_speaker = list()
        history_mel_len = list()
        history = None
        if self.history_type != "none":
            history_basenames = sorted([tg_path.replace(".wav", "") for tg_path in os.listdir(os.path.join(self.raw_path, self.sub_dir_name, f"{dialog}")) if ".wav" in tg_path], key=lambda x:int(x.split("_")[0]))
            history_basenames = history_basenames[:turn][-history_len:]

            if self.history_type == "Guo":
                text_emb_path = os.path.join(
                    self.preprocessed_path,
                    "text_emb",
                    "{}-text_emb-{}.npy".format(speaker, basename),
                )
                text_emb = np.load(text_emb_path)

                for i, h_basename in enumerate(history_basenames):
                    h_idx = int(self.basename_to_id[h_basename])
                    h_speaker = self.speaker[h_idx]
                    h_speaker_id = self.speaker_map[h_speaker]
                    h_text_emb_path = os.path.join(
                        self.preprocessed_path,
                        "text_emb",
                        "{}-text_emb-{}.npy".format(h_speaker, h_basename),
                    )
                    h_text_emb = np.load(h_text_emb_path)
                    
                    history_text_emb.append(h_text_emb)
                    history_speaker.append(h_speaker_id)

                    # Padding
                    if i == history_len-1 and history_len < self.max_history_len:
                        self.pad_history(
                            self.max_history_len-history_len,
                            history_text_emb=history_text_emb,
                            history_speaker=history_speaker,
                        )
                if turn == 0:
                    self.pad_history(
                        self.max_history_len,
                        history_text_emb=history_text_emb,
                        history_speaker=history_speaker,
                    )

                history = {
                    "text_emb": text_emb,
                    "history_len": history_len,
                    "history_text_emb": history_text_emb,
                    "history_speaker": history_speaker,
                }

        return (basename, speaker_id, phone, raw_text, spker_embed, emotion_id, history)

    def pad_history(self, 
            pad_size,
            history_text=None,
            history_text_emb=None,
            history_text_len=None,
            history_pitch=None,
            history_energy=None,
            history_duration=None,
            history_emotion=None,
            history_speaker=None,
            history_mel_len=None,
        ):
        for _ in range(pad_size):
            history_text.append(np.zeros(1, dtype=np.int64)) if history_text is not None else None
            history_text_emb.append(np.zeros(self.text_emb_size, dtype=np.float32)) if history_text_emb is not None else None
            history_text_len.append(0) if history_text_len is not None else None # meaningless zero padding, should be cut out by mask of history_len
            history_pitch.append(np.zeros(1, dtype=np.float64)) if history_pitch is not None else None
            history_energy.append(np.zeros(1, dtype=np.float32)) if history_energy is not None else None
            history_duration.append(np.zeros(1, dtype=np.float64)) if history_duration is not None else None
            history_emotion.append(0) if history_emotion is not None else None # meaningless zero padding, should be cut out by mask of history_len
            history_speaker.append(0) if history_speaker is not None else None # meaningless zero padding, should be cut out by mask of history_len
            history_mel_len.append(0) if history_mel_len is not None else None # meaningless zero padding, should be cut out by mask of history_len

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            emotion = []
            for line in f.readlines():
                if self.load_emotion:
                    n, s, t, r, e = line.strip("\n").split("|")
                else:
                    n, s, t, r, *_ = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
                if self.load_emotion:
                    emotion.append(e)
            return name, speaker, text, raw_text, emotion

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])
        spker_embeds = np.concatenate(np.array([d[4] for d in data]), axis=0) \
            if self.load_spker_embed else None
        emotions = np.array([d[5] for d in data]) if self.load_emotion else None

        texts = pad_1D(texts)

        history_info = None
        if self.history_type != "none":
            if self.history_type == "Guo":
                text_embs = [d[6]["text_emb"] for d in data]
                history_lens = [d[6]["history_len"] for d in data]
                history_text_embs = [d[6]["history_text_emb"] for d in data]
                history_speakers = [d[6]["history_speaker"] for d in data]

                text_embs = np.array(text_embs)
                history_lens = np.array(history_lens)
                history_text_embs = np.array(history_text_embs)
                history_speakers = np.array(history_speakers)

                history_info = (
                    text_embs,
                    history_lens,
                    history_text_embs,
                    history_speakers,
                )

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens), spker_embeds, emotions, history_info
