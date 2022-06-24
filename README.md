# Birds of Istanbul [![IOS](https://img.shields.io/badge/Platform-iOS-blue)](https://apps.apple.com/tr/app/birds-of-istanbul/id1628864377?l=tr)

> Shazam for birds of istanbul!

This repository contains the code for the deep learning model for Birds of Istanbul application on iOS.

<img src="https://github.com/farrinfedra/BirdsOfIstanbul/blob/main/logo.png?raw=true" alt="logo" style = "width:100px; margin-right:0px;" />

*******
## Table of Contents
1. [Introduction](#introduction)
2.  [Features](#features)
3.  [Model](#model)
4.  [Dataset](#dataset)
5.  [Preprocessing](#preprocessing)
6.  [Results](#results)
7.  [Members](#members)
8.  [References](#references)
 
*******
## Introduction
> What is Birds of Istanbul?

An iOS application for classifying bird songs developed for ornithologists, bird watcher, or those who are curious and want to explore birds in their surroundings. 

<img src="https://github.com/farrinfedra/BirdsOfIstanbul/blob/main/app_snapshots.png?raw=true" alt="app" style = "width:650px;" />

Click to visit the repository for Swift code: [![Github](http://img.shields.io/badge/github-%231877F2.svg?&style=for-the-badge&logo=github&logoColor=white&color=black)](https://github.com/omeraksoy1/COMP491)


## Features
> What features does Birds of Istanbul offer?
- You can record bird songs in the app or upload your previously recorded bird recordings and learn the species.
- You can explore birds in your neighborhood and visualize them on the map.
- Get to know your classified birds as well as 400 species in different regions of Türkiye.

## Model
This section is about the birds of istanbul model.
> All about the Birds of Istanbul Model.

Based on Audio Spectrogram transformer (AST) [[1]](#1), pre-trained on 397 bird species, fine-tuned on 400 bird species from different regions of Türkiye. AST takes as an input a raw waveform of a bird song and converts it into a 128 × 100t spectrogram which is then converted into a sequence of 16 × 16 patches. These patches are fed into a linear projection layer that result in 1-dimensional patch embedding of size 768. The patch embeddings are then accompanied by their corresponding learnable positional embeddings along with a classification token (CLS) as shown in figure 6. And, then fed into a Transformer encoder with 12 layers, 12 heads and an embedding dimension of 768. Lastly, the output of the Transformer encoder for CLS token which represents the spectrogram, is fed into a linear layer and the resulting classification labels are obtained. In our case, following the same process as in the pre-trained AST on 397 bird species, the model predicts bird species in each 5 second chunk of an audio recording and the bird species with the highest score is extracted.

<img src="https://github.com/farrinfedra/BirdsOfIstanbul/blob/main/app_model_pic.png?raw=true" alt="app2" style = "width:650px;" />

## Dataset
All bird recordings are obtained from Xeno Canto [[2]](#2) website. Downloaded 335k bird recordings of 400 bird species in Türkiye and created metadata. Here are train - validation - test dataset statistics.

|5 seconds    | Train       | Validation  | Test        | 
| ----------- | ----------- | ----------- | ----------- | 
| No          | 268k        |  33.5k      | 33.5k       | 
| `Yes`       | 1.4 M       | 600k        | 300k        |

## Preprocessing
- [x] Converted recordings to wav format.
- [x] Re-sampled to 16 kHz.
- [x] Split audios to 40 seconds to speed up the mel spectrogram conversion process.
- [x] Create metadata and checked labels with that of eBird [[3]](#3).
- [x] Split data into train, validation and test in 80% - 10% - 10% portions, respectively.

## Results
Here are some results of our model. The model is tested on two different datasets. The first test set is from Xeno Canto described in [Dataset](#dataset) section and the second dataset is obtained from eBird [[3]](#3) and contains around 7k recordings of real recordings from different regions in Türkiye.

|Metric           | Xeno Canto  | eBird  | 
| -----------     | ----------- | ----------- | 
| F1 Micro        | 0.7229      |  0.7061     | 
| Precision Macro | 0.9272      | 0.65        | 
| Precision Micro | 0.9272      | 0.8549      | 
| Recall Micro    | 0.602       | 0.59        | 

## Members
- Farrin Marouf Sofian: ML Researcher 
- Andrew Bond: DevOps
- Kutay Eroğlu: DevOps
- Ömer Faruk Aksoy: Full Stack iOS Development
- Can Köz: Full Stack iOS Development

```diff
@@ Special thanks to Prof. Aykut Erdem, Prof. Bariş Akgün, Prof. Erkut Erdem, Prof. Çaglar Akçay and Burak Can Biner For their help and guidance throughout the project. @@
```
******
## References
<a id = "1">[1]</a> 
Gong, Y., Chung, Y. and Glass, J., 2021. AST: Audio Spectrogram Transformer. In Interspeech.

<a id = "2">[2]</a> 
Canto Foundation, X., 2022. URL https://xeno-canto.org.

<a id = "3">[3]</a> 
eBird. 2021. eBird: An online database of bird distribution and abundance [web application]. eBird, Cornell Lab of Ornithology, Ithaca, New York. Available: http://www.ebird.org (Accessed: May 15, 2022) 

<a id = "4">[4]</a> 
Swift. [Online]. Available: https://www.swift.org/ . (Accessed: May 24, 2022).

