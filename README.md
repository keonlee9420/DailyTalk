[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fkeonlee9420%2FDailyTalk&count_bg=%23707EE1&title_bg=%23555555&icon=pytorch.svg&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
# DailyTalk: Spoken Dialogue Dataset for Conversational Text-to-Speech

### Keon Lee, Kyumin Park, Daeyoung Kim

In this [paper](https://arxiv.org/abs/2207.01063), we introduce DailyTalk, a high-quality conversational speech dataset designed for Text-to-Speech.

<p align="center">
    <img src="img/dailytalk_table.png" width="70%">
</p>

<p align="center">
    <img src="img/dailytalk_model.png" width="70%">
</p>

**Abstract:** The majority of current TTS datasets, which are collections of individual utterances, contain few conversational aspects in terms of both style and metadata. In this paper, we introduce DailyTalk, a high-quality conversational speech dataset designed for Text-to-Speech. We sampled, modified, and recorded 2,541 dialogues from the open-domain dialogue dataset DailyDialog which are adequately long to represent context of each dialogue. During the data construction step, we maintained attributes distribution originally annotated in DailyDialog to support diverse dialogue in DailyTalk. On top of our dataset, we extend prior work as our baseline, where a non-autoregressive TTS is conditioned on historical information in a dialog. We gather metadata so that a TTS model can learn historical dialog information, the key to generating context-aware speech. From the baseline experiment results, we show that DailyTalk can be used to train neural text-to-speech models, and our baseline can represent contextual information. The DailyTalk dataset and baseline code are freely available for academic use with CC-BY-SA 4.0 license.

<details>
<summary><b><h3>Statistic Details</h3></b></summary>

#### Phoneme Distribution
<p align="center">
    <img src="img/phoneme-distribution.png" width="70%">
</p>

#### Characteristic Distribution
<p align="center">
    <img src="img/feature-distribution.png" width="70%">
</p>

</details>

# Dataset
You can download our [dataset](https://drive.google.com/drive/folders/1WRt-EprWs-2rmYxoWYT9_13omlhDHcaL).

# Pretrained Models
You can download our [pretrained models](https://drive.google.com/drive/folders/1RrmWzJM1iWhQg2_wvDHlszVDJRV60Wfu?usp=sharing). There are two different directories: 'history_none' and 'history_guo'. The former has no historical encodings so that it is not a conversational context-aware model. The latter has historical encodings following [Conversational End-to-End TTS for Voice Agent](https://arxiv.org/abs/2005.10438) (Guo et al., 2020).

Toggle the type of history encodings by
```yaml
# In the model.yaml
history_encoder:
  type: "Guo" # ["none", "Guo"]
```

# Quickstart

## Dependencies
You can install the Python dependencies with
```
pip3 install -r requirements.txt
```
Also, `Dockerfile` is provided for `Docker` users.

## Inference

You have to download the [pretrained models](https://drive.google.com/drive/folders/1RrmWzJM1iWhQg2_wvDHlszVDJRV60Wfu?usp=sharing) and put them in `output/ckpt/DailyTalk/`. The models are trained with unsupervised duration modeling under transformer building block and the history encoding types.

Only the batch inference is supported as the generation of a turn may need contextual history of the conversation. Try

```
python3 synthesize.py --source preprocessed_data/DailyTalk/val_*.txt --restore_step RESTORE_STEP --mode batch --dataset DailyTalk
```
to synthesize all utterances in `preprocessed_data/DailyTalk/val_*.txt`.


# Training

## Preprocessing

- For a **multi-speaker TTS** with external speaker embedder, download [ResCNN Softmax+Triplet pretrained model](https://drive.google.com/file/d/1F9NvdrarWZNktdX9KlRYWWHDwRkip_aP) of [philipperemy's DeepSpeaker](https://github.com/philipperemy/deep-speaker) for the speaker embedding and locate it in `./deepspeaker/pretrained_models/`. Please note that our pretrained models are not trained with this (they are trained with `speaker_embedder: "none"`).
- Run 
  ```
  python3 prepare_align.py --dataset DailyTalk
  ```
  for some preparations.

  For the forced alignment, [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/) (MFA) is used to obtain the alignments between the utterances and the phoneme sequences.
  Pre-extracted alignments for the datasets are provided [here](https://drive.google.com/drive/folders/1fizpyOiQ1lG2UDaMlXnT3Ll4_j6Xwg7K?usp=sharing). 
  You have to unzip the files in `preprocessed_data/DailyTalk/TextGrid/`. Alternately, you can [run the aligner by yourself](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/index.html). Please note that our pretrained models are not trained with supervised duration modeling (they are trained with `learn_alignment: True`).

  After that, run the preprocessing script by
  ```
  python3 preprocess.py --dataset DailyTalk
  ```

## Training

Train your model with
```
python3 train.py --dataset DailyTalk
```
Useful options:
- To use a [Automatic Mixed Precision](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html), append `--use_amp` argument to the above command.
- The trainer assumes single-node multi-GPU training. To use specific GPUs, specify `CUDA_VISIBLE_DEVICES=<GPU_IDs>` at the beginning of the above command.

# TensorBoard

Use
```
tensorboard --logdir output/log
```

to serve TensorBoard on your localhost.
The loss curves, synthesized mel-spectrograms, and audios are shown.

![](./img/tensorboard_loss.png)
![](./img/tensorboard_spec.png)
![](./img/tensorboard_audio.png)

# Notes

- Convolutional embedding is used as [StyleSpeech](https://github.com/keonlee9420/StyleSpeech) for phoneme-level variance in unsupervised duration modeling. Otherwise, bucket-based embedding is used as [FastSpeech2](https://github.com/keonlee9420/Comprehensive-Transformer-TTS).
- Unsupervised duration modeling in phoneme-level will take longer time than frame-level since the additional computation of phoneme-level variance is activated at runtime.
- Two options for embedding for the **multi-speaker TTS** setting: training speaker embedder from scratch or using a pre-trained [philipperemy's DeepSpeaker](https://github.com/philipperemy/deep-speaker) model (as [STYLER](https://github.com/keonlee9420/STYLER) did). You can toggle it by setting the config (between `'none'` and `'DeepSpeaker'`).
- For vocoder, **HiFi-GAN** is used for all experiments in our paper.

# Citation

If you would like to use our dataset and code or refer to our paper, please cite as follows.
```bash
@misc{lee2022dailytalk,
    title={DailyTalk: Spoken Dialogue Dataset for Conversational Text-to-Speech},
    author={Keon Lee and Kyumin Park and Daeyoung Kim},
    year={2022},
    eprint={2207.01063},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```

# License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

# References
- [keonlee9420's STYLER](https://github.com/keonlee9420/STYLER)
- [keonlee9420's Expressive-FastSpeech2](https://github.com/keonlee9420/Expressive-FastSpeech2)
- [keonlee9420's Comprehensive-Transformer-TTS](https://github.com/keonlee9420/Comprehensive-Transformer-TTS)
