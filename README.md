# Content-Violation-Detection


## Problem Statement
Cyber bullying involves **posting and sharing wrong, private, negative and harmful information** about victims. With the world under quarantine, this is only going to become worse.


### Prerequisites
- Python (3.7 Recommended)
- For installation of [PyTorch](https://pytorch.org/), refer their official website. 
 

### Image Classification
 Dataset: **[NSFW Image Classification Dataset](https://www.kaggle.com/krishnaalagiri/nsfw-image-classification)**  
 The classification model has been pretrained using **Self Supervised Learning** using only **unlabeled Imagenet data**(Pretrained model and script to be released soon :p). The model is then trained with a linear classifier over the above mentioned data over 5 categories:
1. **`pornography`** - Nudes and pornography images
2. **`hentai`** - Hentai images, but also includes pornographic drawings
3. **`sexually_provocative`** - Sexually explicit images, but not pornography. Think semi-nude photos, playboy, bikini, beach volleyball, etc. Considered acceptable by most public social media platforms.
4. **`neutral`** - Safe for work neutral images of everyday things and people
5. **`drawing`** - Safe for work drawings (including anime)

This achieved an accuracy of **83% using purely unlabaled data** and **a single linear layer**.

### Text Toxicity Prediction

The text classification model is built to predict the toxicity of texts to pre-emptively prevent any occurrence of cyberbullying and harassment before they tend to occur. We're using **BERT to overcome the current challenges including understanding the context of text so as to detect sarcasm and cultural references, as it uses Stacked Transformer Encoders and Self-Attention Mechanism to understand the relationship between words and sentences, the context from a given sentence**. A combination of datasets have been used for this purpose. 

```
Text_Input: I want to drug and rape her 
======================
Toxic: 0.987 
Severe_Toxic: 0.053 
Obscene: 0.100 
Threat 0.745 
Insult: 0.124 
Identity_Hate: 0.019 
======================
Result: Extremely Toxic as classified as Threat, Toxic 
Action: Text has been blocked. 

```

Referred to [lonePatient](https://github.com/lonePatient/Bert-Multi-Label-Text-Classification) for training my BERT model




## License
MIT © [ajaykrishnan23](/LICENSE)


<br><br>
<p align="center">
  Stay Fab ❤️ 
</p>


