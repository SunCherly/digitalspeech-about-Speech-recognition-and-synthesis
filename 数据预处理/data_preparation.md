# Data Preparation

## Generate Manifest

*DeepSpeech2 on PaddlePaddle* accepts a textual **manifest** file as its data set interface. A manifest file summarizes a set of speech data, with each line containing some meta data (e.g. file path, transcription, duration) of one audio clip, in [JSON](http://www.json.org/) format, such as:

DeepSpeech2 on PaddlePaddle 使用文本 清单 文件作为其数据集接口。清单文件总结了一组语音数据，每行包含一个音频片段的一些元数据（例如文件路径、转录、持续时间），格式为 JSON：

```
{"audio_filepath": "/home/work/.cache/paddle/Libri/134686/1089-134686-0001.flac", "duration": 3.275, "text": "stuff it into you his belly counselled him"}

{"audio_filepath": "/home/work/.cache/paddle/Libri/134686/1089-134686-0007.flac", "duration": 4.275, "text": "a cold lucid indifference reigned in his soul"}
```

To use your custom data, you only need to generate such manifest files to summarize the dataset. Given such summarized manifests, training, inference and all other modules can be aware of where to access the audio files, as well as their meta data including the transcription labels.
要使用自定义数据，您只需生成这样的清单文件来总结数据集。通过这些总结的清单文件，训练、推理和所有其他模块可以知道如何访问音频文件以及它们的元数据，包括转录标签。

For how to generate such manifest files, please refer to `examples/librispeech/local/librispeech.py`, which will download data and generate manifest files for LibriSpeech dataset.

有关如何生成此类清单文件，请参考 examples/librispeech/local/librispeech.py，该文件将下载数据并为 LibriSpeech 数据集生成清单文件。

## Compute Mean & Stddev for Normalizer
计算标准化的均值和标准差


To perform z-score normalization (zero-mean, unit stddev) upon audio features, we have to estimate in advance the mean and standard deviation of the features, with some training samples:
为了对音频特征执行 z-score 标准化（零均值，单位标准差），我们必须提前用一些训练样本估计特征的均值和标准差：

```bash
python3 utils/compute_mean_std.py \
--num_samples 2000 \
--spectrum_type linear \
--manifest_path examples/librispeech/data/manifest.train \
--output_path examples/librispeech/data/mean_std.npz
```

It will compute the mean and standard deviations of the power spectrum feature with 2000 random sampled audio clips listed in `examples/librispeech/data/manifest.train` and save the results to `examples/librispeech/data/mean_std.npz` for further usage.
它将计算 examples/librispeech/data/manifest.train 中列出的 2000 个随机抽取的音频片段的功率谱特征的均值和标准差，并将结果保存到 examples/librispeech/data/mean_std.npz 以供进一步使用。


## Build Vocabulary

A vocabulary of possible characters is required to convert the transcription into a list of token indices for training, and in decoding, to convert from a list of indices back to the text again. Such a character-based vocabulary can be built with `utils/build_vocab.py`.
需要一个可能字符的词汇表来将转录转换为训练的标记索引列表，并在解码时将索引列表转换回文本。这样的基于字符的词汇表可以用 utils/build_vocab.py 构建。
```bash
python3 utils/build_vocab.py \
--count_threshold 0 \
--vocab_path examples/librispeech/data/eng_vocab.txt \
--manifest_paths examples/librispeech/data/manifest.train
```

It will write a vocabulary file `examples/librispeech/data/vocab.txt` with all transcription text in `examples/librispeech/data/manifest.train`, without vocabulary truncation (`--count_threshold 0`).
它将根据 examples/librispeech/data/manifest.train 中的所有转录文本写一个词汇表文件 examples/librispeech/data/vocab.txt，不进行词汇表截断（--count_threshold 0）。