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
- **Number of Classes:** 30 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label                       | Examples                                                                                                                                                                                                                                              |
|:----------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| LOUDNESS_SOFT_TOO_SOFT      | <ul><li>'All the quiet sounds are too quiet for me.'</li><li>'Even the little sounds seem too low.'</li><li>'Everything on the softer end is hard to hear through my aid.'</li></ul>                                                                  |
| LOUDNESS_WATER_FAN_LOUD     | <ul><li>'Running water sounds too loud through my hearing aid.'</li><li>'The tap running sounds much louder than it should.'</li><li>'Water flowing into the sink is too loud when I wear my device.'</li></ul>                                       |
| NOISE_SUDDEN_UNCOMFORTABLE  | <ul><li>'Sudden noises feel really uncomfortable when I wear my hearing aid.'</li><li>'A sudden sound makes me jump; it feels unpleasant through the aid.'</li><li>'Why do unexpected noises bother me so much with my hearing machine on?'</li></ul> |
| NOISE_BACKGROUND_RESTAURANT | <ul><li>'The background noise is too loud through my hearing aid.'</li><li>'I can hear the room noise more loudly than I want to.'</li><li>'My hearing machine makes the background sound too loud.'</li></ul>                                        |
| NOISE_WIND_TOO_LOUD         | <ul><li>'Wind noise is too loud through my hearing aid.'</li><li>'When I step outside, the wind sounds much too loud in my hearing aids.'</li><li>'The sound of wind is overpowering with my hearing machine on.'</li></ul>                           |
| FEEDBACK_WHISTLING          | <ul><li>'My hearing aid keeps whistling.'</li><li>'There is a high-pitched whistle coming from my hearing aid.'</li><li>'My hearing aid makes a whistling sound when I wear it.'</li></ul>                                                            |
| LOUDNESS_LOUD_DISTORTED     | <ul><li>'Loud sounds are distorted through my hearing aid.'</li><li>'When a sound gets loud, it starts to sound distorted.'</li><li>'My hearing aid makes loud noises sound crackly and unclear.'</li></ul>                                           |
| LOUDNESS_ALARMS_CRYING_LOUD | <ul><li>"Alarms and children's crying sound too loud through my hearing aid."</li><li>'The alarm is painfully loud when I wear my hearing aid.'</li><li>'Children crying sounds much too loud with my hearing aid on.'</li></ul>                      |
| LOUDNESS_SUDDEN_SOUNDS_LOUD | <ul><li>'Sudden sounds are too loud with my hearing aid.'</li><li>'A door slamming sounds painfully loud when I wear my hearing aid.'</li><li>'Unexpected noises are much too loud through my hearing device.'</li></ul>                              |
| LOUDNESS_CHEWING_TOO_LOUD   | <ul><li>'My chewing sounds are too loud with my hearing aid.'</li><li>'I can hear my own chewing very loudly when I wear my hearing aid.'</li><li>'The sound of me chewing is overpowering through my hearing device.'</li></ul>                      |
| OWN_VOICE_MUFFLED_DULL      | <ul><li>'My own voice sounds muffled when I wear my hearing aid.'</li><li>'My voice sounds dull and unclear through the hearing aid.'</li><li>'When I speak, my voice sounds blocked up.'</li></ul>                                                   |
| OWN_VOICE_SHARP_METALLIC    | <ul><li>'My own voice sounds sharp and metallic when I wear my hearing aid.'</li><li>'My voice has a tinny sound through the hearing aid.'</li><li>'When I speak, my voice sounds too sharp and unnatural.'</li></ul>                                 |
| SPEECH_WHISPER_DISTANT_SOFT | <ul><li>'Whispering sounds too soft through my hearing aid.'</li><li>'I cannot hear whispers clearly because they sound too faint.'</li><li>'People speaking from a distance sound too quiet with my device.'</li></ul>                               |
| SPEECH_TOO_LOUD_BOOMY       | <ul><li>'Speech sounds too loud and boomy through my hearing aid.'</li><li>"People's voices have too much bass and sound booming."</li><li>'Voices sound unnaturally loud when I use my hearing aid.'</li></ul>                                       |
| SOUND_MUFFLED_DULL          | <ul><li>'Sounds are muffled and dull through my hearing aid.'</li><li>'Everything sounds dull, as if something is covering the speaker.'</li><li>'The sounds around me lack clarity and seem muffled.'</li></ul>                                      |
| SOUND_ARTIFICIAL_MECHANICAL | <ul><li>'Sounds from my hearing aid feel artificial and mechanical.'</li><li>'Everything sounds like it is coming from a machine.'</li><li>'The sound quality does not feel natural; it sounds processed.'</li></ul>                                  |
| SOUND_TOO_MUCH_BASS         | <ul><li>'There is too much bass coming through my hearing aid.'</li><li>'The sound is very bass-heavy and lacks balance.'</li><li>'Low sounds are overpowering everything else.'</li></ul>                                                            |
| SOUND_DISTORTED             | <ul><li>'Sounds from my hearing aid are distorted.'</li><li>'The audio sounds crackly and unclear when I wear my device.'</li><li>'Everything sounds warped through my hearing aid.'</li></ul>                                                        |
| NOISE_AMBIENT_NOTICEABLE    | <ul><li>'I notice the background ambience too much with my hearing aid.'</li><li>'Soft environmental noise is very noticeable when I wear the device.'</li><li>'Even quiet background sounds seem too prominent.'</li></ul>                           |
| LOUDNESS_LOUD_TOO_LOUD      | <ul><li>'All loud sounds feel too loud when I use my hearing aid.'</li><li>'Sudden or strong sounds are uncomfortably loud through my device.'</li><li>'My hearing aid makes loud noises seem even louder than they should.'</li></ul>                |
| LOUDNESS_NEARBY_TOO_LOUD    | <ul><li>'Sounds from people or objects close to me are too loud through my hearing aid.'</li><li>'Nearby sounds feel much louder than they should.'</li><li>'When someone speaks right beside me, their voice is too loud.'</li></ul>                 |
| OWN_VOICE_HOLLOW            | <ul><li>'My voice sounds as if I am speaking inside a tunnel.'</li><li>'When I talk, my voice has an empty, hollow quality.'</li><li>'My voice sounds echoey and hollow to me with the device on.'</li></ul>                                          |
| OWN_VOICE_THIN              | <ul><li>'My voice sounds weak and thin with the device switched on.'</li><li>'When I speak, my voice lacks its usual fullness.'</li><li>'My own voice sounds light and hollow instead of normal.'</li></ul>                                           |
| OWN_VOICE_BOOMY_LOUD        | <ul><li>'My voice sounds boomy with the device switched on.'</li><li>'When I speak, my voice seems much louder than normal.'</li><li>'My own voice feels amplified inside my ears.'</li></ul>                                                         |
| SPEECH_TOO_SOFT             | <ul><li>'Speech sounds too soft through my hearing aid.'</li><li>"People's voices are too quiet when I wear the device."</li><li>'I struggle to hear what people are saying because speech is too low.'</li></ul>                                     |
| SPEECH_ECHOIC               | <ul><li>"People's voices seem to echo when they talk to me."</li><li>'Conversation has an echo effect with my hearing aid on.'</li><li>'Voices sound as if they are bouncing around in a room.'</li></ul>                                             |
| SPEECH_POOR_QUALITY         | <ul><li>'Speech sounds poor through my hearing aid.'</li><li>"People's voices do not sound clear or natural."</li><li>'The overall quality of speech is bad when I wear the device.'</li></ul>                                                        |
| SOUND_TINY_METALLIC_SHARP   | <ul><li>'Sounds from my hearing aid are too tiny and sharp.'</li><li>'Everything sounds metallic when I wear my hearing aid.'</li><li>'Voices have a thin, tinny sound through the device.'</li></ul>                                                 |
| SOUND_ECHOIC                | <ul><li>'Everything sounds echoey through my hearing aid.'</li><li>'Sounds seem to bounce back when I listen with the device on.'</li><li>'My hearing aid makes everyday sounds echo.'</li></ul>                                                      |
| NOISE_CROWD_TOO_LOUD        | <ul><li>'The background noise is too loud when I am in a crowd.'</li><li>'Crowd noise becomes too loud through my hearing aid.'</li><li>'The sound of many people around me is overwhelming.'</li></ul>                                               |

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
| Word count   | 3   | 10.0719 | 17  |

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
| NOISE_AMBIENT_NOTICEABLE    | 50                    |
| NOISE_BACKGROUND_RESTAURANT | 150                   |
| NOISE_CROWD_TOO_LOUD        | 50                    |
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
- num_iterations: 4
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
| 0.0012 | 1    | 0.2642        | -               |
| 0.0609 | 50   | 0.1996        | -               |
| 0.1218 | 100  | 0.1675        | -               |
| 0.1827 | 150  | 0.1458        | -               |
| 0.2436 | 200  | 0.1312        | -               |
| 0.3045 | 250  | 0.1033        | -               |
| 0.3654 | 300  | 0.1053        | -               |
| 0.4263 | 350  | 0.0899        | -               |
| 0.4872 | 400  | 0.0936        | -               |
| 0.5481 | 450  | 0.0877        | -               |
| 0.6090 | 500  | 0.0769        | -               |
| 0.6699 | 550  | 0.0727        | -               |
| 0.7308 | 600  | 0.0717        | -               |
| 0.7917 | 650  | 0.0821        | -               |
| 0.8526 | 700  | 0.0702        | -               |
| 0.9135 | 750  | 0.0614        | -               |
| 0.9744 | 800  | 0.0708        | -               |
| 1.0    | 821  | -             | 0.0703          |

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