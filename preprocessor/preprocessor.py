import os
import random
import re
import json
import copy

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.stats import betabinom
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio
from text import text_to_sequence, sequence_to_text, grapheme_to_phoneme
from utils.tools import get_phoneme_level_pitch, get_phoneme_level_energy
from g2p_en import G2p
from sentence_transformers import SentenceTransformer


class Preprocessor:
    def __init__(self, preprocess_config, model_config, train_config):
        random.seed(train_config['seed'])
        self.config = preprocess_config
        self.dataset = preprocess_config["dataset"]
        self.speakers = set()
        self.emotions = set()
        self.sub_dir = preprocess_config["path"]["sub_dir_name"]
        self.data_dir = preprocess_config["path"]["corpus_path"]
        self.in_dir = os.path.join(preprocess_config["path"]["raw_path"], self.sub_dir)
        self.out_dir = preprocess_config["path"]["preprocessed_path"]
        self.val_size = 2
        # self.val_size = preprocess_config["preprocessing"]["val_size"]
        self.val_dialog_ids = self.get_val_dialog_ids()
        self.metadata = self.load_metadata()
        self.sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
        self.filter_length = preprocess_config["preprocessing"]["stft"]["filter_length"]
        self.trim_top_db = preprocess_config["preprocessing"]["audio"]["trim_top_db"]
        self.beta_binomial_scaling_factor = preprocess_config["preprocessing"]["duration"]["beta_binomial_scaling_factor"]
        self.g2p = G2p()
        self.text_embbeder = SentenceTransformer('distiluse-base-multilingual-cased-v1')

        assert preprocess_config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert preprocess_config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = preprocess_config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = preprocess_config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            preprocess_config["preprocessing"]["stft"]["filter_length"],
            preprocess_config["preprocessing"]["stft"]["hop_length"],
            preprocess_config["preprocessing"]["stft"]["win_length"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            preprocess_config["preprocessing"]["audio"]["sampling_rate"],
            preprocess_config["preprocessing"]["mel"]["mel_fmin"],
            preprocess_config["preprocessing"]["mel"]["mel_fmax"],
        )
        self.val_dialog_ids_prior_frame = self.get_val_dialog_ids_prior(os.path.join(self.out_dir, "val_frame.txt"))
        self.val_dialog_ids_prior_phone = self.get_val_dialog_ids_prior(os.path.join(self.out_dir, "val_phone.txt"))

    def get_val_dialog_ids_prior(self, val_prior_path):
        val_dialog_ids_prior = set()
        if os.path.isfile(val_prior_path):
            print("Load pre-defined validation set...")
            with open(val_prior_path, "r", encoding="utf-8") as f:
                for m in f.readlines():
                    val_dialog_ids_prior.add(int(m.split("|")[0].split("_")[-1].strip("d")))
            return list(val_dialog_ids_prior)
        else:
            return None

    def get_val_dialog_ids(self):
        data_size = len(os.listdir(self.in_dir))
        val_dialog_ids = random.sample(range(data_size), k=self.val_size)
        # print("val_dialog_ids:", val_dialog_ids)
        return val_dialog_ids

    def load_metadata(self):
        with open(os.path.join(self.data_dir, "metadata.json")) as f:
            metadata = json.load(f)
        return metadata

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "text_emb")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "mel_frame")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "mel_phone")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch_frame")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch_phone")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy_frame")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy_phone")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "attn_prior")), exist_ok=True)

        print("Processing Data ...")
        filtered_out_dialog_frame = set()
        filtered_out_dialog_phone = set()
        train_frame = list()
        val_frame = list()
        train_phone = list()
        val_phone = list()
        n_frames = 0
        max_seq_len = -float('inf')
        pitch_frame_scaler = StandardScaler()
        pitch_phone_scaler = StandardScaler()
        energy_frame_scaler = StandardScaler()
        energy_phone_scaler = StandardScaler()

        def partial_fit(scaler, value):
            if len(value) > 0:
                scaler.partial_fit(value.reshape((-1, 1)))

        def compute_stats(pitch_scaler, energy_scaler, pitch_dir="pitch", energy_dir="energy"):
            if self.pitch_normalization:
                pitch_mean = pitch_scaler.mean_[0]
                pitch_std = pitch_scaler.scale_[0]
            else:
                # A numerical trick to avoid normalization...
                pitch_mean = 0
                pitch_std = 1
            if self.energy_normalization:
                energy_mean = energy_scaler.mean_[0]
                energy_std = energy_scaler.scale_[0]
            else:
                energy_mean = 0
                energy_std = 1

            pitch_min, pitch_max = self.normalize(
                os.path.join(self.out_dir, pitch_dir), pitch_mean, pitch_std
            )
            energy_min, energy_max = self.normalize(
                os.path.join(self.out_dir, energy_dir), energy_mean, energy_std
            )
            return (pitch_min, pitch_max, pitch_mean, pitch_std), (energy_min, energy_max, energy_mean, energy_std)

        # Compute pitch, energy, duration, and mel-spectrogram
        # speakers = self.speakers.copy()
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))): # here, speaker is actually a dialog_id
            # if len(self.speakers) == 0:
            #     speakers[speaker] = i
            for wav_name in os.listdir(os.path.join(self.in_dir, speaker)):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]
                tg_path = os.path.join(
                    self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                )
                (
                    info_frame,
                    info_phone,
                    pitch_frame,
                    pitch_phone,
                    energy_frame,
                    energy_phone,
                    n,
                ) = self.process_utterance(tg_path, speaker, basename)
                if info_frame is None and info_phone is None:
                    filtered_out_dialog_frame.add(int(speaker))
                    filtered_out_dialog_phone.add(int(speaker))
                    continue
                else:
                    # Save frame level information
                    if info_frame is not None:
                        if self.val_dialog_ids_prior_frame is not None:
                            if int(speaker) not in self.val_dialog_ids_prior_frame:
                                train_frame.append(info_frame)
                            else:
                                val_frame.append(info_frame)
                        else:
                            if int(speaker) not in self.val_dialog_ids:
                                train_frame.append(info_frame)
                            else:
                                val_frame.append(info_frame)

                        partial_fit(pitch_frame_scaler, pitch_frame)
                        partial_fit(energy_frame_scaler, energy_frame)
                    else:
                        filtered_out_dialog_frame.add(int(speaker))
                    # Save phone level information
                    if info_phone is not None:
                        if self.val_dialog_ids_prior_phone is not None:
                            if int(speaker) not in self.val_dialog_ids_prior_phone:
                                train_phone.append(info_phone)
                            else:
                                val_phone.append(info_phone)
                        else:
                            if int(speaker) not in self.val_dialog_ids:
                                train_phone.append(info_phone)
                            else:
                                val_phone.append(info_phone)

                        partial_fit(pitch_phone_scaler, pitch_phone)
                        partial_fit(energy_phone_scaler, energy_phone)
                    else:
                        filtered_out_dialog_phone.add(int(speaker))

                    if n > max_seq_len:
                        max_seq_len = n

                    n_frames += n

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        pitch_frame_stats, energy_frame_stats = compute_stats(
            pitch_frame_scaler,
            energy_frame_scaler,
            pitch_dir="pitch_frame",
            energy_dir="energy_frame",
        )
        pitch_phone_stats, energy_phone_stats = compute_stats(
            pitch_phone_scaler,
            energy_phone_scaler,
            pitch_dir="pitch_phone",
            energy_dir="energy_phone",
        )

        # Save files
        # with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
        #     f.write(json.dumps(speakers))

        if len(self.speakers) != 0:
            speaker_dict = dict()
            for i, speaker in enumerate(list(self.speakers)):
                speaker_dict[speaker] = int(speaker)
            with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
                f.write(json.dumps(speaker_dict))

        if len(self.emotions) != 0:
            emotion_dict = dict()
            for i, emotion in enumerate(list(self.emotions)):
                emotion_dict[emotion] = i
            with open(os.path.join(self.out_dir, "emotions.json"), "w") as f:
                f.write(json.dumps(emotion_dict))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch_frame": [float(var) for var in pitch_frame_stats],
                "pitch_phone": [float(var) for var in pitch_phone_stats],
                "energy_frame": [float(var) for var in energy_frame_stats],
                "energy_phone": [float(var) for var in energy_phone_stats],
                "max_seq_len": max_seq_len
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(train_frame)
        random.shuffle(train_phone)
        train_frame = [r for r in train_frame if r is not None]
        train_phone = [r for r in train_phone if r is not None]
        val_frame = [r for r in val_frame if r is not None]
        val_phone = [r for r in val_phone if r is not None]
        # Filter out incomplete dialog in all train & val set
        filtered_out_dialog_frame, filtered_out_dialog_phone = list(filtered_out_dialog_frame), list(filtered_out_dialog_phone)
        train_frame = [r for r in train_frame if int(r.split("|")[0].split("_")[-1].strip("d")) not in filtered_out_dialog_frame]
        train_phone = [r for r in train_phone if int(r.split("|")[0].split("_")[-1].strip("d")) not in filtered_out_dialog_phone]
        val_frame = [r for r in val_frame if int(r.split("|")[0].split("_")[-1].strip("d")) not in filtered_out_dialog_frame]
        val_phone = [r for r in val_phone if int(r.split("|")[0].split("_")[-1].strip("d")) not in filtered_out_dialog_phone]
        # Sort validation set by dialog
        val_frame = sorted(val_frame, key=lambda x: (int(x.split("|")[0].split("_")[-1].lstrip("d")), int(x.split("|")[0].split("_")[0])))
        val_phone = sorted(val_phone, key=lambda x: (int(x.split("|")[0].split("_")[-1].lstrip("d")), int(x.split("|")[0].split("_")[0])))

        # Write metadata
        with open(os.path.join(self.out_dir, "filtered_out_dialog_frame.txt"), "w", encoding="utf-8") as f:
            for m in sorted(filtered_out_dialog_frame):
                f.write(str(m) + "\n")
        with open(os.path.join(self.out_dir, "filtered_out_dialog_phone.txt"), "w", encoding="utf-8") as f:
            for m in sorted(filtered_out_dialog_phone):
                f.write(str(m) + "\n")
        with open(os.path.join(self.out_dir, "train_frame.txt"), "w", encoding="utf-8") as f:
            for m in train_frame:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val_frame.txt"), "w", encoding="utf-8") as f:
            for m in val_frame:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "train_phone.txt"), "w", encoding="utf-8") as f:
            for m in train_phone:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val_phone.txt"), "w", encoding="utf-8") as f:
            for m in val_phone:
                f.write(m + "\n")

        return (train_frame ,train_phone, val_frame ,val_phone)

    def load_audio(self, wav_path):
        wav_raw, _ = librosa.load(wav_path, self.sampling_rate)
        _, index = librosa.effects.trim(wav_raw, top_db=self.trim_top_db, frame_length=self.filter_length, hop_length=self.hop_length)
        wav = wav_raw[index[0]:index[1]]
        duration = (index[1] - index[0]) / self.hop_length
        return wav_raw.astype(np.float32), wav.astype(np.float32), int(duration)

    def process_utterance(self, tg_path, speaker, basename):
        phone_out_exist, frame_out_exist = True, True

        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))

        speaker = basename.split("_")[1]
        dialog_id = basename.split("_")[-1].lstrip("d")
        uttr_id = basename.split("_")[0]
        emotion = self.metadata[dialog_id][uttr_id]["emotion"]
        if emotion == "no emotion":
            emotion = "none"
        self.speakers.add(speaker)
        self.emotions.add(emotion)

        wav_raw, wav, duration = self.load_audio(wav_path)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")
        phone = grapheme_to_phoneme(raw_text, self.g2p)
        phones = "{" + "}{".join(phone) + "}"
        phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
        text_frame = phones.replace("}{", " ")

        # Text embedding
        text_emb = self.text_embbeder.encode([raw_text])[0]

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)
        pitch = pitch[: duration]
        if np.sum(pitch != 0) <= 1:
            frame_out_exist = False
        else:
            # Compute mel-scale spectrogram and energy
            mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
            mel_spectrogram = mel_spectrogram[:, : duration]
            energy = energy[: duration]

            # Compute alignment prior
            attn_prior = self.beta_binomial_prior_distribution(
                mel_spectrogram.shape[1],
                len(phone),
                self.beta_binomial_scaling_factor,
            )

            # Frame-level variance
            pitch_frame, energy_frame = copy.deepcopy(pitch), copy.deepcopy(energy)
            mel_spectrogram_frame = copy.deepcopy(mel_spectrogram)

            # Save files
            text_emb_filename = "{}-text_emb-{}.npy".format(speaker, basename)
            np.save(os.path.join(self.out_dir, "text_emb", text_emb_filename), text_emb)

            attn_prior_filename = "{}-attn_prior-{}.npy".format(speaker, basename)
            np.save(os.path.join(self.out_dir, "attn_prior", attn_prior_filename), attn_prior)

            pitch_frame_filename = "{}-pitch-{}.npy".format(speaker, basename)
            np.save(os.path.join(self.out_dir, "pitch_frame", pitch_frame_filename), pitch_frame)

            energy_frame_filename = "{}-energy-{}.npy".format(speaker, basename)
            np.save(os.path.join(self.out_dir, "energy_frame", energy_frame_filename), energy_frame)

            mel_frame_filename = "{}-mel-{}.npy".format(speaker, basename)
            np.save(
                os.path.join(self.out_dir, "mel_frame", mel_frame_filename),
                mel_spectrogram_frame.T,
            )

        # Supervised duration features
        if os.path.exists(tg_path):
            # Get alignments
            textgrid = tgt.io.read_textgrid(tg_path)
            phone, duration, start, end = self.get_alignment(
                textgrid.get_tier_by_name("phones")
            )
            text_phone = "{" + " ".join(phone) + "}"
            if start >= end:
                phone_out_exist = False
            else:
                # Read and trim wav files
                wav, _ = librosa.load(wav_path, self.sampling_rate)
                wav = wav.astype(np.float32)
                wav = wav[
                    int(self.sampling_rate * start) : int(self.sampling_rate * end)
                ]

                # Compute fundamental frequency
                pitch, t = pw.dio(
                    wav.astype(np.float64),
                    self.sampling_rate,
                    frame_period=self.hop_length / self.sampling_rate * 1000,
                )
                pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

                pitch = pitch[: sum(duration)]
                if np.sum(pitch != 0) <= 1:
                    phone_out_exist = False
                else:
                    # Compute mel-scale spectrogram and energy
                    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
                    mel_spectrogram = mel_spectrogram[:, : sum(duration)]
                    energy = energy[: sum(duration)]

                    # Phone-level variance
                    pitch_phone, energy_phone = get_phoneme_level_pitch(duration, pitch), get_phoneme_level_energy(duration, energy)
                    mel_spectrogram_phone = copy.deepcopy(mel_spectrogram)

                    # Save files
                    if not frame_out_exist:
                        text_emb_filename = "{}-text_emb-{}.npy".format(speaker, basename)
                        np.save(os.path.join(self.out_dir, "text_emb", text_emb_filename), text_emb)

                    dur_filename = "{}-duration-{}.npy".format(speaker, basename)
                    np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

                    pitch_phone_filename = "{}-pitch-{}.npy".format(speaker, basename)
                    np.save(os.path.join(self.out_dir, "pitch_phone", pitch_phone_filename), pitch_phone)

                    energy_phone_filename = "{}-energy-{}.npy".format(speaker, basename)
                    np.save(os.path.join(self.out_dir, "energy_phone", energy_phone_filename), energy_phone)

                    mel_phone_filename = "{}-mel-{}.npy".format(speaker, basename)
                    np.save(
                        os.path.join(self.out_dir, "mel_phone", mel_phone_filename),
                        mel_spectrogram_phone.T,
                    )
        else:
            phone_out_exist = False

        if not phone_out_exist and not frame_out_exist:
            return tuple([None]*7)
        else:
            return (
                "|".join([basename, speaker, text_frame, raw_text, emotion]) if frame_out_exist else None,
                "|".join([basename, speaker, text_phone, raw_text, emotion]) if phone_out_exist else None,
                self.remove_outlier(pitch_frame) if frame_out_exist else None,
                self.remove_outlier(pitch_phone) if phone_out_exist else None,
                self.remove_outlier(energy_frame) if frame_out_exist else None,
                self.remove_outlier(energy_phone) if phone_out_exist else None,
                max(mel_spectrogram_frame.shape[1] if frame_out_exist else -1, mel_spectrogram_phone.shape[1] if phone_out_exist else -1)
            )

    def beta_binomial_prior_distribution(self, phoneme_count, mel_count, scaling_factor=1.0):
        P, M = phoneme_count, mel_count
        x = np.arange(0, P)
        mel_text_probs = []
        for i in range(1, M+1):
            a, b = scaling_factor*i, scaling_factor*(M+1-i)
            rv = betabinom(P, a, b)
            mel_i_prob = rv.pmf(x)
            mel_text_probs.append(mel_i_prob)
        return np.array(mel_text_probs)

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
