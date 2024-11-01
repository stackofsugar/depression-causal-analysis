<a name="readme-top"></a>

<div align="center">
    <img src="https://github.com/user-attachments/assets/cd947efc-b1b7-4372-a98c-75729280dfa1" alt="Project hero" height="250"  />
    <h1>Depression Causal Analysis</h1>
</div>

<!-- Badges -->
<p align="center">
    <!-- Project Status: Completed -->
    <img src="https://img.shields.io/badge/status-completed-orange?style=for-the-badge&labelColor=black" />
    <!-- License -->
    <a href="https://github.com/stackofsugar/depression-causal-analysis/blob/main/LICENSE">
        <img src="https://img.shields.io/github/license/stackofsugar/depression-causal-analysis?style=for-the-badge&labelColor=black&color=green" />
    </a>
    <!-- View Model Card -->
    <a href="https://huggingface.co/stackofsugar/mentallongformer-cams-finetuned">
        <img src="https://img.shields.io/badge/Model_Card-%F0%9F%A4%97%20Huggingface-yellow?style=for-the-badge&labelColor=black" />
    </a>
    <!-- Try at Huggingface Spaces -->
    <a href="https://huggingface.co/spaces/stackofsugar/depression-causal-analysis">
        <img src="https://img.shields.io/badge/Try_at-%F0%9F%A4%97%20Huggingface_Spaces-yellow?style=for-the-badge&labelColor=black" />
    </a>
</p>

<p align="center">
    Fine-tuned Longformer model to detect root cause of depression in long texts.
</p>

<div align="center">
    <h2>Coming soon!</h2>
</div>

## 📝 Table of Contents

-   [About The Project](#about)
-   [Getting Started](#getting-started)
-   [Author(s)](#authors)
-   [Appendix](#appendix)

## 💭 About The Project <a name="about"></a>

This project aims to fine-tune a [MentalLongformer](https://arxiv.org/abs/2304.10447) model for a depression causal analysis downstream task using CAMS dataset ([paper](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.686.pdf) | [GitHub](https://github.com/drmuskangarg/CAMS/)). This model reaches SOTA performance (read below for details) for depression causal analysis task.

This project is developed as a part of my undegraduate thesis research.

### Technologies Used

-   **Language**: Python
-   **Libraries**: [PyTorch](https://pytorch.org/), [HuggingFace family of libraries](https://huggingface.co/)
-   **Hyperparameter Tuning**: [Optuna](https://optuna.org/)

### Performance

I measured my model using F1-score and accuracy to maintain comparability with other researches. Other metrics and training data are available, please see the [appendix](#appendix).

With [CAMS Dataset](https://github.com/drmuskangarg/CAMS/):

| **Author**                                               | **Model Used**       | **F1 Score** | **Accuracy** |
|----------------------------------------------------------|----------------------|--------------|--------------|
| [Garg et al. (2022)](https://arxiv.org/abs/2207.04674v1) | CNN + LSTM           | 0.4633       | 0.4778       |
| [Saxena et al. (2022)](http://arxiv.org/abs/2210.08430)  | BiLSTM               | 0.4700       | 0.5054       |
| [Ji et al. (2023)](https://arxiv.org/abs/2304.10447v1)   | **MentalLongformer** | 0.4874       | 0.4920       |
| [Ji et al. (2023)](https://arxiv.org/abs/2304.10447v1)   | MentalXLNet          | 0.5008       | 0.5080       |
| **Mine**                                                 | **MentalLongformer** | **0.5524**   | **0.6064**   |

If I miss a research, please [let me know](https://github.com/stackofsugar/depression-causal-analysis/issues/new)!

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 🛫 Getting Started <a name="getting-started"></a>

### Run Inference on HuggingFace Spaces

This is the easiest way to try my model, as you don't need to setup anything. Just head to my [HuggingFace Spaces for this project](https://huggingface.co/spaces/stackofsugar/depression-causal-analysis) and try typing some depressive text and let my model do the magic of analyzing the reason of its depression. You can also load some long example text as `MentalLongformer` is better for long texts.

### Loading The Model using HuggingFace Python Library

If you wish to use my model to infer your dataset or maybe pre-train it further, you can import my model in a Python script/notebook.

```py
from transformers import LongformerTokenizer, LongformerForSequenceClassification

tokenizer = LongformerTokenizer.from_pretrained("aimh/mental-longformer-base-4096")
model = LongformerForSequenceClassification.from_pretrained("stackofsugar/mentallongformer-cams-finetuned")  
```

If you prefer to use the high-level HuggingFace pipeline to make predictions, you can also do it in a Python script/notebook.

```py
from transformers import pipeline

pipe = pipeline("text-classification", model="stackofsugar/mentallongformer-cams-finetuned")     
```

If you're not sure yet, you might want to read [HuggingFace's Course on NLP](https://huggingface.co/learn/nlp-course/chapter1/1).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 📚 Author(s) <a name="authors"></a>

-   [@stackofsugar](https://github.com/stackofsugar) (Myself)

See also a list of [contributors](https://github.com/stackofsugar/personal-website/graphs/contributors) who has participated in this project.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 📋 Appendix <a name="appendix"></a>

### Experiment Setup

- **CPU**: Intel Xeon Silver 4216
- **RAM**: 128GB DDR4 ECC
- **GPU**: 16GB RTX A4000

With the setup above, it took me ~4 hours to complete a training run and ~36 hours to run hyperparameter tuning at 20 runs with pruning enabled.

I would like to acknowledge Universitas Sebelas Maret for the computational resources provided for this project.

### Full Performance Metrics

| **Metric**    | **Score** |
|---------------|-----------|
| **F1 Score**  | 0.5524    |
| **Accuracy**  | 0.6064    |
| **Precision** | 0.6020    |
| **Recall**    | 0.5385    |

### Hyperparameters

| **Hyperparameter** | **Value** |
|--------------------|-----------|
| **Learning Rate**  | 3.04e-5   |
| **Warmup Steps**   | 75        |
| **Weight Decay**   | 2.692e-5  |
| **Train Epochs**   | 5         |
| **fp16**           | True      |

Total number of steps trained: 620
