---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: Could you check why alarms sound so loud through my hearing aid?
- text: When the sound gets loud, my hearing aid makes it feel even louder.
- text: Both hearing aids make running water sound excessively loud.
- text: Running water sounds too loud when I wear my hearing aid.
- text: The general restaurant chatter is too loud even when I'm sitting at a quiet
    table.
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: true
base_model: sentence-transformers/all-MiniLM-L12-v2
---

# SetFit with sentence-transformers/all-MiniLM-L12-v2

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 128 tokens
- **Number of Classes:** 31 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label                       | Examples                                                                                                                                                                                                                                                                                                                                  |
|:----------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| LOUDNESS_SOFT_TOO_SOFT      | <ul><li>'All the quiet sounds are too quiet for me.'</li><li>'Even the little sounds seem too low.'</li><li>'Everything on the softer end is hard to hear through my aid.'</li></ul>                                                                                                                                                      |
| LOUDNESS_WATER_FAN_LOUD     | <ul><li>'Running water sounds too loud through my hearing aid.'</li><li>'The tap running sounds much louder than it should.'</li><li>'Water flowing into the sink is too loud when I wear my device.'</li></ul>                                                                                                                           |
| NOISE_AMBIENT_NOTICEABLE    | <ul><li>'Soft ambience is not loud, but it is distracting with my hearing machine.'</li><li>'Quiet background noise is more prominent than I want even when I wear my hearing aid.'</li><li>'Soft ambience is not loud, but it is constantly noticeable with my hearing machine.'</li></ul>                                               |
| NOISE_SOUND_UNCOMFORTABLE   | <ul><li>'People speaking sound sharp and unpleasant through my hearing aid when background noise is present.'</li><li>'It is hard to listen to phone calls at a busy place because they sound uncomfortable.'</li><li>'The sounds around me sound hard to tolerate through my hearing device when background noise is present.'</li></ul> |
| NOISE_SUDDEN_UNCOMFORTABLE  | <ul><li>'Sudden noises feel really uncomfortable when I wear my hearing aid.'</li><li>'A sudden sound makes me jump; it feels unpleasant through the aid.'</li><li>'Why do unexpected noises bother me so much with my hearing machine on?'</li></ul>                                                                                     |
| NOISE_BACKGROUND_RESTAURANT | <ul><li>'The background noise is too loud through my hearing aid.'</li><li>'I can hear the room noise more loudly than I want to.'</li><li>'My hearing machine makes the background sound too loud.'</li></ul>                                                                                                                            |
| FEEDBACK_WHISTLING          | <ul><li>'My hearing aid keeps whistling.'</li><li>'There is a high-pitched whistle coming from my hearing aid.'</li><li>'My hearing aid makes a whistling sound when I wear it.'</li></ul>                                                                                                                                                |
| LOUDNESS_LOUD_DISTORTED     | <ul><li>'Loud sounds are distorted through my hearing aid.'</li><li>'When a sound gets loud, it starts to sound distorted.'</li><li>'My hearing aid makes loud noises sound crackly and unclear.'</li></ul>                                                                                                                               |
| LOUDNESS_ALARMS_CRYING_LOUD | <ul><li>"Alarms and children's crying sound too loud through my hearing aid."</li><li>'The alarm is painfully loud when I wear my hearing aid.'</li><li>'Children crying sounds much too loud with my hearing aid on.'</li></ul>                                                                                                          |
| LOUDNESS_SUDDEN_SOUNDS_LOUD | <ul><li>'Sudden sounds are too loud with my hearing aid.'</li><li>'A door slamming sounds painfully loud when I wear my hearing aid.'</li><li>'Unexpected noises are much too loud through my hearing device.'</li></ul>                                                                                                                  |
| LOUDNESS_CHEWING_TOO_LOUD   | <ul><li>'My chewing sounds are too loud with my hearing aid.'</li><li>'I can hear my own chewing very loudly when I wear my hearing aid.'</li><li>'The sound of me chewing is overpowering through my hearing device.'</li></ul>                                                                                                          |
| OWN_VOICE_MUFFLED_DULL      | <ul><li>'My own voice sounds muffled when I wear my hearing aid.'</li><li>'My voice sounds dull and unclear through the hearing aid.'</li><li>'When I speak, my voice sounds blocked up.'</li></ul>                                                                                                                                       |
| OWN_VOICE_SHARP_METALLIC    | <ul><li>'My own voice sounds sharp and metallic when I wear my hearing aid.'</li><li>'My voice has a tinny sound through the hearing aid.'</li><li>'When I speak, my voice sounds too sharp and unnatural.'</li></ul>                                                                                                                     |
| SPEECH_TOO_SOFT             | <ul><li>"People's speech sounds too soft through my hearing aid."</li><li>'I can hear other sounds, but voices are too quiet.'</li><li>'When someone talks to me, their voice is too soft to hear clearly.'</li></ul>                                                                                                                     |
| SPEECH_POOR_QUALITY         | <ul><li>'Speech quality sounds poor through my hearing aid.'</li><li>"People's voices sound unclear and unnatural with my device."</li><li>'I can hear people talking, but the speech quality is bad.'</li></ul>                                                                                                                          |
| SOUND_MUFFLED_DULL          | <ul><li>'Sounds are muffled and dull through my hearing aid.'</li><li>'Everything sounds dull, as if something is covering the speaker.'</li><li>'The sounds around me lack clarity and seem muffled.'</li></ul>                                                                                                                          |
| SOUND_ARTIFICIAL_MECHANICAL | <ul><li>'Sounds from my hearing aid feel artificial and mechanical.'</li><li>'Everything sounds like it is coming from a machine.'</li><li>'The sound quality does not feel natural; it sounds processed.'</li></ul>                                                                                                                      |
| SOUND_TOO_MUCH_BASS         | <ul><li>'There is too much bass coming through my hearing aid.'</li><li>'The sound is very bass-heavy and lacks balance.'</li><li>'Low sounds are overpowering everything else.'</li></ul>                                                                                                                                                |
| SOUND_DISTORTED             | <ul><li>'Sounds from my hearing aid are distorted.'</li><li>'The audio sounds crackly and unclear when I wear my device.'</li><li>'Everything sounds warped through my hearing aid.'</li></ul>                                                                                                                                            |
| LOUDNESS_LOUD_TOO_LOUD      | <ul><li>'All loud sounds feel too loud when I use my hearing aid.'</li><li>'Sudden or strong sounds are uncomfortably loud through my device.'</li><li>'My hearing aid makes loud noises seem even louder than they should.'</li></ul>                                                                                                    |
| LOUDNESS_NEARBY_TOO_LOUD    | <ul><li>'Sounds from people or objects close to me are too loud through my hearing aid.'</li><li>'Nearby sounds feel much louder than they should.'</li><li>'When someone speaks right beside me, their voice is too loud.'</li></ul>                                                                                                     |
| OWN_VOICE_HOLLOW            | <ul><li>'My voice sounds as if I am speaking inside a tunnel.'</li><li>'When I talk, my voice has an empty, hollow quality.'</li><li>'My voice sounds echoey and hollow to me with the device on.'</li></ul>                                                                                                                              |
| OWN_VOICE_THIN              | <ul><li>'My voice sounds weak and thin with the device switched on.'</li><li>'When I speak, my voice lacks its usual fullness.'</li><li>'My own voice sounds light and hollow instead of normal.'</li></ul>                                                                                                                               |
| OWN_VOICE_BOOMY_LOUD        | <ul><li>'My voice sounds boomy with the device switched on.'</li><li>'When I speak, my voice seems much louder than normal.'</li><li>'My own voice feels amplified inside my ears.'</li></ul>                                                                                                                                             |
| SPEECH_WHISPER_DISTANT_SOFT | <ul><li>'I cannot hear whispers clearly with my hearing aid.'</li><li>'Distant voices sound too soft for me to understand.'</li><li>'People speaking from across the room are too quiet.'</li></ul>                                                                                                                                       |
| SPEECH_TOO_LOUD_BOOMY       | <ul><li>'Speech sounds too loud through my hearing aid.'</li><li>"People's voices sound boomy when I wear the device."</li><li>'Conversation is uncomfortably loud with my hearing aid on.'</li></ul>                                                                                                                                     |
| SPEECH_ECHOIC               | <ul><li>"People's voices seem to echo when they talk to me."</li><li>'Conversation has an echo effect with my hearing aid on.'</li><li>'Voices sound as if they are bouncing around in a room.'</li></ul>                                                                                                                                 |
| SOUND_TINY_METALLIC_SHARP   | <ul><li>'Sounds from my hearing aid are too tiny and sharp.'</li><li>'Everything sounds metallic when I wear my hearing aid.'</li><li>'Voices have a thin, tinny sound through the device.'</li></ul>                                                                                                                                     |
| SOUND_ECHOIC                | <ul><li>'Everything sounds echoey through my hearing aid.'</li><li>'Sounds seem to bounce back when I listen with the device on.'</li><li>'My hearing aid makes everyday sounds echo.'</li></ul>                                                                                                                                          |
| NOISE_WIND_TOO_LOUD         | <ul><li>'The wind noise is too loud through my hearing aid.'</li><li>'When I go outside, the wind sounds much too loud.'</li><li>'My hearing aid makes the sound of wind very noticeable.'</li></ul>                                                                                                                                      |
| NOISE_CROWD_TOO_LOUD        | <ul><li>'The background noise is too loud when I am in a crowd.'</li><li>'Crowd noise becomes too loud through my hearing aid.'</li><li>'The sound of many people around me is overwhelming.'</li></ul>                                                                                                                                   |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("Running water sounds too loud when I wear my hearing aid.")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median  | Max |
|:-------------|:----|:--------|:----|
| Word count   | 3   | 10.2423 | 23  |

| Label                       | Training Sample Count |
|:----------------------------|:----------------------|
| FEEDBACK_WHISTLING          | 49                    |
| LOUDNESS_ALARMS_CRYING_LOUD | 50                    |
| LOUDNESS_CHEWING_TOO_LOUD   | 50                    |
| LOUDNESS_LOUD_DISTORTED     | 50                    |
| LOUDNESS_LOUD_TOO_LOUD      | 50                    |
| LOUDNESS_NEARBY_TOO_LOUD    | 50                    |
| LOUDNESS_SOFT_TOO_SOFT      | 50                    |
| LOUDNESS_SUDDEN_SOUNDS_LOUD | 50                    |
| LOUDNESS_WATER_FAN_LOUD     | 99                    |
| NOISE_AMBIENT_NOTICEABLE    | 100                   |
| NOISE_BACKGROUND_RESTAURANT | 150                   |
| NOISE_CROWD_TOO_LOUD        | 50                    |
| NOISE_SOUND_UNCOMFORTABLE   | 50                    |
| NOISE_SUDDEN_UNCOMFORTABLE  | 50                    |
| NOISE_WIND_TOO_LOUD         | 50                    |
| OWN_VOICE_BOOMY_LOUD        | 49                    |
| OWN_VOICE_HOLLOW            | 49                    |
| OWN_VOICE_MUFFLED_DULL      | 50                    |
| OWN_VOICE_SHARP_METALLIC    | 50                    |
| OWN_VOICE_THIN              | 48                    |
| SOUND_ARTIFICIAL_MECHANICAL | 50                    |
| SOUND_DISTORTED             | 50                    |
| SOUND_ECHOIC                | 50                    |
| SOUND_MUFFLED_DULL          | 50                    |
| SOUND_TINY_METALLIC_SHARP   | 50                    |
| SOUND_TOO_MUCH_BASS         | 50                    |
| SPEECH_ECHOIC               | 48                    |
| SPEECH_POOR_QUALITY         | 50                    |
| SPEECH_TOO_LOUD_BOOMY       | 50                    |
| SPEECH_TOO_SOFT             | 50                    |
| SPEECH_WHISPER_DISTANT_SOFT | 50                    |

### Training Hyperparameters
- batch_size: (16, 16)
- num_epochs: (1, 16)
- max_steps: -1
- sampling_strategy: oversampling
- num_iterations: 32
- body_learning_rate: (2e-05, 1e-05)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: True

### Training Results
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0001 | 1    | 0.2297        | -               |
| 0.0072 | 50   | 0.1975        | -               |
| 0.0144 | 100  | 0.1922        | -               |
| 0.0215 | 150  | 0.1850        | -               |
| 0.0287 | 200  | 0.1655        | -               |
| 0.0359 | 250  | 0.1450        | -               |
| 0.0431 | 300  | 0.1341        | -               |
| 0.0502 | 350  | 0.1279        | -               |
| 0.0574 | 400  | 0.1136        | -               |
| 0.0646 | 450  | 0.1059        | -               |
| 0.0718 | 500  | 0.0908        | -               |
| 0.0789 | 550  | 0.0847        | -               |
| 0.0861 | 600  | 0.0697        | -               |
| 0.0933 | 650  | 0.0762        | -               |
| 0.1005 | 700  | 0.0659        | -               |
| 0.1076 | 750  | 0.0584        | -               |
| 0.1148 | 800  | 0.0542        | -               |
| 0.1220 | 850  | 0.0551        | -               |
| 0.1292 | 900  | 0.0515        | -               |
| 0.1363 | 950  | 0.0525        | -               |
| 0.1435 | 1000 | 0.0441        | -               |
| 0.1507 | 1050 | 0.0454        | -               |
| 0.1579 | 1100 | 0.0467        | -               |
| 0.1650 | 1150 | 0.0405        | -               |
| 0.1722 | 1200 | 0.0435        | -               |
| 0.1794 | 1250 | 0.0385        | -               |
| 0.1866 | 1300 | 0.0345        | -               |
| 0.1937 | 1350 | 0.0320        | -               |
| 0.2009 | 1400 | 0.0318        | -               |
| 0.2081 | 1450 | 0.0350        | -               |
| 0.2153 | 1500 | 0.0295        | -               |
| 0.2224 | 1550 | 0.0293        | -               |
| 0.2296 | 1600 | 0.0273        | -               |
| 0.2368 | 1650 | 0.0236        | -               |
| 0.2440 | 1700 | 0.0233        | -               |
| 0.2511 | 1750 | 0.0256        | -               |
| 0.2583 | 1800 | 0.0234        | -               |
| 0.2655 | 1850 | 0.0207        | -               |
| 0.2727 | 1900 | 0.0222        | -               |
| 0.2799 | 1950 | 0.0187        | -               |
| 0.2870 | 2000 | 0.0198        | -               |
| 0.2942 | 2050 | 0.0212        | -               |
| 0.3014 | 2100 | 0.0207        | -               |
| 0.3086 | 2150 | 0.0151        | -               |
| 0.3157 | 2200 | 0.0173        | -               |
| 0.3229 | 2250 | 0.0186        | -               |
| 0.3301 | 2300 | 0.0190        | -               |
| 0.3373 | 2350 | 0.0178        | -               |
| 0.3444 | 2400 | 0.0184        | -               |
| 0.3516 | 2450 | 0.0162        | -               |
| 0.3588 | 2500 | 0.0154        | -               |
| 0.3660 | 2550 | 0.0129        | -               |
| 0.3731 | 2600 | 0.0122        | -               |
| 0.3803 | 2650 | 0.0193        | -               |
| 0.3875 | 2700 | 0.0163        | -               |
| 0.3947 | 2750 | 0.0163        | -               |
| 0.4018 | 2800 | 0.0153        | -               |
| 0.4090 | 2850 | 0.0118        | -               |
| 0.4162 | 2900 | 0.0127        | -               |
| 0.4234 | 2950 | 0.0164        | -               |
| 0.4305 | 3000 | 0.0109        | -               |
| 0.4377 | 3050 | 0.0101        | -               |
| 0.4449 | 3100 | 0.0109        | -               |
| 0.4521 | 3150 | 0.0085        | -               |
| 0.4592 | 3200 | 0.0124        | -               |
| 0.4664 | 3250 | 0.0133        | -               |
| 0.4736 | 3300 | 0.0090        | -               |
| 0.4808 | 3350 | 0.0101        | -               |
| 0.4879 | 3400 | 0.0097        | -               |
| 0.4951 | 3450 | 0.0120        | -               |
| 0.5023 | 3500 | 0.0074        | -               |
| 0.5095 | 3550 | 0.0067        | -               |
| 0.5166 | 3600 | 0.0072        | -               |
| 0.5238 | 3650 | 0.0080        | -               |
| 0.5310 | 3700 | 0.0103        | -               |
| 0.5382 | 3750 | 0.0076        | -               |
| 0.5454 | 3800 | 0.0082        | -               |
| 0.5525 | 3850 | 0.0120        | -               |
| 0.5597 | 3900 | 0.0076        | -               |
| 0.5669 | 3950 | 0.0082        | -               |
| 0.5741 | 4000 | 0.0060        | -               |
| 0.5812 | 4050 | 0.0063        | -               |
| 0.5884 | 4100 | 0.0084        | -               |
| 0.5956 | 4150 | 0.0100        | -               |
| 0.6028 | 4200 | 0.0071        | -               |
| 0.6099 | 4250 | 0.0038        | -               |
| 0.6171 | 4300 | 0.0070        | -               |
| 0.6243 | 4350 | 0.0074        | -               |
| 0.6315 | 4400 | 0.0064        | -               |
| 0.6386 | 4450 | 0.0054        | -               |
| 0.6458 | 4500 | 0.0071        | -               |
| 0.6530 | 4550 | 0.0041        | -               |
| 0.6602 | 4600 | 0.0045        | -               |
| 0.6673 | 4650 | 0.0071        | -               |
| 0.6745 | 4700 | 0.0062        | -               |
| 0.6817 | 4750 | 0.0058        | -               |
| 0.6889 | 4800 | 0.0058        | -               |
| 0.6960 | 4850 | 0.0035        | -               |
| 0.7032 | 4900 | 0.0035        | -               |
| 0.7104 | 4950 | 0.0074        | -               |
| 0.7176 | 5000 | 0.0044        | -               |
| 0.7247 | 5050 | 0.0067        | -               |
| 0.7319 | 5100 | 0.0086        | -               |
| 0.7391 | 5150 | 0.0057        | -               |
| 0.7463 | 5200 | 0.0036        | -               |
| 0.7534 | 5250 | 0.0051        | -               |
| 0.7606 | 5300 | 0.0032        | -               |
| 0.7678 | 5350 | 0.0031        | -               |
| 0.7750 | 5400 | 0.0036        | -               |
| 0.7821 | 5450 | 0.0055        | -               |
| 0.7893 | 5500 | 0.0067        | -               |
| 0.7965 | 5550 | 0.0049        | -               |
| 0.8037 | 5600 | 0.0035        | -               |
| 0.8108 | 5650 | 0.0037        | -               |
| 0.8180 | 5700 | 0.0056        | -               |
| 0.8252 | 5750 | 0.0042        | -               |
| 0.8324 | 5800 | 0.0025        | -               |
| 0.8396 | 5850 | 0.0033        | -               |
| 0.8467 | 5900 | 0.0026        | -               |
| 0.8539 | 5950 | 0.0044        | -               |
| 0.8611 | 6000 | 0.0047        | -               |
| 0.8683 | 6050 | 0.0035        | -               |
| 0.8754 | 6100 | 0.0041        | -               |
| 0.8826 | 6150 | 0.0038        | -               |
| 0.8898 | 6200 | 0.0042        | -               |
| 0.8970 | 6250 | 0.0033        | -               |
| 0.9041 | 6300 | 0.0043        | -               |
| 0.9113 | 6350 | 0.0030        | -               |
| 0.9185 | 6400 | 0.0032        | -               |
| 0.9257 | 6450 | 0.0030        | -               |
| 0.9328 | 6500 | 0.0028        | -               |
| 0.9400 | 6550 | 0.0021        | -               |
| 0.9472 | 6600 | 0.0019        | -               |
| 0.9544 | 6650 | 0.0020        | -               |
| 0.9615 | 6700 | 0.0021        | -               |
| 0.9687 | 6750 | 0.0030        | -               |
| 0.9759 | 6800 | 0.0040        | -               |
| 0.9831 | 6850 | 0.0019        | -               |
| 0.9902 | 6900 | 0.0021        | -               |
| 0.9974 | 6950 | 0.0020        | -               |
| 1.0    | 6968 | -             | 0.0354          |

### Framework Versions
- Python: 3.10.12
- SetFit: 1.2.0
- Sentence Transformers: 6.3.0
- Transformers: 5.20.2
- PyTorch: 2.14.1+cpu
- Datasets: 5.2.0
- Tokenizers: 0.23.3

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->