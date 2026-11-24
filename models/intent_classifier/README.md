---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: I hear my own chewing more clearly than I want to.
- text: Could someone review my fitting because sounds seem echoic?
- text: The loud sounds is breaking up when volume high.
- text: People's voices sound altered and unclear.
- text: People's voices sound robotic when I wear my device.
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: true
base_model: sentence-transformers/all-MiniLM-L6-v2
---

# SetFit with sentence-transformers/all-MiniLM-L6-v2

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 256 tokens
- **Number of Classes:** 28 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label                       | Examples                                                                                                                                                                                                                                |
|:----------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| SOUND_ECHOIC                | <ul><li>'A high-pitched tone keeps coming from the device.'</li><li>'I can hear a high-pitched whistle coming from my hearing aid.'</li><li>'Hearing aid makes a squeek when I touch it.'</li></ul>                                     |
| SOUND_TINY_METALLIC_SHARP   | <ul><li>'can u make it less sharp pls'</li><li>'High notes are harsh and piercing through the hearing aid.'</li><li>'Sounds lack fullness and have a thin, bright quality.'</li></ul>                                                   |
| NOISE_WIND_TOO_LOUD         | <ul><li>'wind noise um gets really loud when i walk'</li><li>'My hearing aid picks up the wind too strongly.'</li><li>'I hear a loud hiss of wind around the hearing aid.'</li></ul>                                                    |
| OWN_VOICE_SHARP_METALLIC    | <ul><li>'my voice is tinny but everyone else sounds ok'</li><li>'Please help make voices less bass-heavy and more comfortable.'</li><li>'Even a calm voice sounds loud and boomy to me.'</li></ul>                                      |
| NOISE_CROWD_TOO_LOUD        | <ul><li>'The noise from a busy group is disturbing my listening.'</li><li>'Street noise is too loud.'</li><li>'The noise around a gathering is too prominent.'</li></ul>                                                                |
| OWN_VOICE_MUFFLED_DULL      | <ul><li>'Both hearing aids make my voice sound muffled.'</li><li>'The sound of my talking is deep, boomy, and uncomfortable.'</li><li>'Even when someone talks close to me, their voice sounds quiet.'</li></ul>                        |
| SOUND_DISTORTED             | <ul><li>'why sounds seem distorted with hearing aid'</li><li>'Environmental audio seems to reverberate too much.'</li><li>"People's voices sound distorted instead of clear."</li></ul>                                                 |
| LOUDNESS_NEARBY_TOO_LOUD    | <ul><li>'the person beside me is sounding too loud'</li><li>'noises keep ringing on after they happen'</li><li>'Sounds seem echoey when I use my hearing aid.'</li></ul>                                                                |
| FEEDBACK_WHISTLING          | <ul><li>'aid whistles when i move it around'</li><li>'can u stop the feedback noise'</li><li>'My hearing aid starts squeaking when I cover my ear.'</li></ul>                                                                           |
| SOUND_MUFFLED_DULL          | <ul><li>'The hearing aid audio is hard to make out.'</li><li>'Sounds are dull and lack clarity when I wear the device.'</li><li>'audio feels blocked and lacks clarity'</li></ul>                                                       |
| LOUDNESS_LOUD_TOO_LOUD      | <ul><li>'I hear a crackling effect on high-volume sounds.'</li><li>'The sudden ring of a phone feels jarring through my hearing aid.'</li><li>'Backgroud noise is too loud in my hearing aid.'</li></ul>                                |
| NOISE_AMBIENT_NOTICEABLE    | <ul><li>'Could you help me reduce the background noise level? It is too loud.'</li><li>'Is there a way to make the general background sound less loud?'</li><li>'I would like the quiet background sounds to stand out less.'</li></ul> |
| LOUDNESS_CHEWING_TOO_LOUD   | <ul><li>'Chewing feels noisy inside my head when I use the hearing aid.'</li><li>'when i chew its very loud'</li><li>'please reduce chewing sound'</li></ul>                                                                            |
| OWN_VOICE_HOLLOW            | <ul><li>'Why do voices sound hollow with my hearing aid on?'</li><li>'The audio lacks body and sounds hollow.'</li><li>'My voice sounds hollow every time I speak with the hearing aid on.'</li></ul>                                   |
| LOUDNESS_WATER_FAN_LOUD     | <ul><li>'Could you check why running water sounds so loud?'</li><li>'Is there a setting to reduce the loudness of running water?'</li><li>'shower water is too loud with aid'</li></ul>                                                 |
| OWN_VOICE_BOOMY_LOUD        | <ul><li>'My hearing aid does not make voices loud enough.'</li><li>'hearing aid makes my voice bassy n loud plz check'</li><li>'The TV dialogue is too quiet with my hearing aid on.'</li></ul>                                         |
| OWN_VOICE_THIN              | <ul><li>'sound thin not natural'</li><li>'Why do sounds have a tiny, tinny quality when I wear my aid?'</li><li>'My voice has a low, hollow sound when I wear my hearing aid.'</li></ul>                                                |
| LOUDNESS_LOUD_DISTORTED     | <ul><li>'When something gets loud, the sound becomes unclear.'</li><li>'The sound seems to repeat slightly in my hearing aid.'</li><li>'Can someone troubleshoot the distortion I hear at high sound levels?'</li></ul>                 |
| SOUND_ARTIFICIAL_MECHANICAL | <ul><li>'My hearing aid makes music sound synthetic and unnatural.'</li><li>'Why do sounds seem artificial with my hearing aid?'</li><li>'Everyday sounds do not sound real when I wear the device.'</li></ul>                          |
| NOISE_SUDDEN_UNCOMFORTABLE  | <ul><li>'When I insert the hearing aid, it starts making a squealing sound.'</li><li>'Can the feedback whistle in my hearing aid be adjusted?'</li><li>'The whistling from my hearing device is irritating.'</li></ul>                  |
| LOUDNESS_SUDDEN_SOUNDS_LOUD | <ul><li>'When a door slams, the sound is much too loud.'</li><li>'Door slams are startlingly loud when I wear the device.'</li><li>'Why are sudden sounds like a door slam so loud?'</li></ul>                                          |
| LOUDNESS_SOFT_TOO_SOFT      | <ul><li>'Soft sounds are too faint in both hearing aids.'</li><li>'I miss quiet parts of conversations because they sound too soft.'</li><li>'I am missing quiet sounds even with the hearing aid turned on.'</li></ul>                 |
| SPEECH_ECHOIC               | <ul><li>"People's voices seem to echo when they talk to me."</li><li>'Speech sounds echoey through my hearing aid.'</li><li>'I hear an echo on conversations even when nobody else notices one.'</li></ul>                              |
| SPEECH_WHISPER_DISTANT_SOFT | <ul><li>'can u make whispers louder on my aid pls'</li><li>'I can tell someone is whispering, but I cannot make out the words.'</li><li>'People speaking from a distance sound too quiet with my device.'</li></ul>                     |
| SPEECH_TOO_LOUD_BOOMY       | <ul><li>'everything sounds boomy and bassy with the aid'</li><li>'Could the speech settings be changed to reduce the booming sound?'</li><li>'Deep voices are especially loud and boomy.'</li></ul>                                     |
| LOUDNESS_ALARMS_CRYING_LOUD | <ul><li>"I find a child's loud crying uncomfortable with my hearing aid."</li><li>'The alarm at work is overwhelming when my hearing aid is on.'</li><li>'When a baby cries nearby, the sound becomes too loud.'</li></ul>              |
| SPEECH_TOO_SOFT             | <ul><li>'People talking nearby make it hard to focus on the person with me.'</li><li>'Please reduce the restaurant noise without making speech too quiet.'</li><li>'the dishes clattering um its too much'</li></ul>                    |
| SOUND_TOO_MUCH_BASS         | <ul><li>'I want less bass and a more balanced sound.'</li><li>'My hearing aid makes music sound overly bassy.'</li><li>'I hear a heavy, booming sound when listening.'</li></ul>                                                        |

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
preds = model("People's voices sound altered and unclear.")
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
| Training set | Min | Median | Max |
|:-------------|:----|:-------|:----|
| Word count   | 3   | 9.9515 | 17  |

| Label                       | Training Sample Count |
|:----------------------------|:----------------------|
| FEEDBACK_WHISTLING          | 38                    |
| LOUDNESS_ALARMS_CRYING_LOUD | 117                   |
| LOUDNESS_CHEWING_TOO_LOUD   | 114                   |
| LOUDNESS_LOUD_DISTORTED     | 118                   |
| LOUDNESS_LOUD_TOO_LOUD      | 353                   |
| LOUDNESS_NEARBY_TOO_LOUD    | 207                   |
| LOUDNESS_SOFT_TOO_SOFT      | 120                   |
| LOUDNESS_SUDDEN_SOUNDS_LOUD | 41                    |
| LOUDNESS_WATER_FAN_LOUD     | 89                    |
| NOISE_AMBIENT_NOTICEABLE    | 128                   |
| NOISE_CROWD_TOO_LOUD        | 277                   |
| NOISE_SUDDEN_UNCOMFORTABLE  | 123                   |
| NOISE_WIND_TOO_LOUD         | 199                   |
| OWN_VOICE_BOOMY_LOUD        | 86                    |
| OWN_VOICE_HOLLOW            | 81                    |
| OWN_VOICE_MUFFLED_DULL      | 236                   |
| OWN_VOICE_SHARP_METALLIC    | 83                    |
| OWN_VOICE_THIN              | 121                   |
| SOUND_ARTIFICIAL_MECHANICAL | 76                    |
| SOUND_DISTORTED             | 122                   |
| SOUND_ECHOIC                | 46                    |
| SOUND_MUFFLED_DULL          | 109                   |
| SOUND_TINY_METALLIC_SHARP   | 78                    |
| SOUND_TOO_MUCH_BASS         | 34                    |
| SPEECH_ECHOIC               | 79                    |
| SPEECH_TOO_LOUD_BOOMY       | 126                   |
| SPEECH_TOO_SOFT             | 74                    |
| SPEECH_WHISPER_DISTANT_SOFT | 45                    |

### Training Hyperparameters
- batch_size: (16, 16)
- num_epochs: (2, 2)
- max_steps: -1
- sampling_strategy: oversampling
- num_iterations: 2
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
| 0.0012 | 1    | 0.2531        | -               |
| 0.0602 | 50   | 0.2173        | -               |
| 0.1205 | 100  | 0.1960        | -               |
| 0.1807 | 150  | 0.1926        | -               |
| 0.2410 | 200  | 0.1696        | -               |
| 0.3012 | 250  | 0.1622        | -               |
| 0.3614 | 300  | 0.1602        | -               |
| 0.4217 | 350  | 0.1607        | -               |
| 0.4819 | 400  | 0.1481        | -               |
| 0.5422 | 450  | 0.1442        | -               |
| 0.6024 | 500  | 0.1303        | -               |
| 0.6627 | 550  | 0.1274        | -               |
| 0.7229 | 600  | 0.1276        | -               |
| 0.7831 | 650  | 0.1339        | -               |
| 0.8434 | 700  | 0.1292        | -               |
| 0.9036 | 750  | 0.1322        | -               |
| 0.9639 | 800  | 0.1283        | -               |
| 1.0    | 830  | -             | 0.1208          |
| 1.0241 | 850  | 0.1169        | -               |
| 1.0843 | 900  | 0.1105        | -               |
| 1.1446 | 950  | 0.1120        | -               |
| 1.2048 | 1000 | 0.1040        | -               |
| 1.2651 | 1050 | 0.1005        | -               |
| 1.3253 | 1100 | 0.1035        | -               |
| 1.3855 | 1150 | 0.1082        | -               |
| 1.4458 | 1200 | 0.1071        | -               |
| 1.5060 | 1250 | 0.0932        | -               |
| 1.5663 | 1300 | 0.0985        | -               |
| 1.6265 | 1350 | 0.0928        | -               |
| 1.6867 | 1400 | 0.0940        | -               |
| 1.7470 | 1450 | 0.1030        | -               |
| 1.8072 | 1500 | 0.0963        | -               |
| 1.8675 | 1550 | 0.0937        | -               |
| 1.9277 | 1600 | 0.0985        | -               |
| 1.9880 | 1650 | 0.0895        | -               |
| 2.0    | 1660 | -             | 0.1162          |

### Framework Versions
- Python: 3.11.0
- SetFit: 1.2.0
- Sentence Transformers: 6.2.0
- Transformers: 5.21.0
- PyTorch: 2.14.1+cpu
- Datasets: 5.1.0
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