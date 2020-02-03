# AM-FM-PM

This repo is for the paper, 'Deep AM-FM: Toolkit For Automatic Dialogue Evaluation', IWSDS 2020 Submission and we are continuously improving this repo to make it a better platform for dialogue evaluation.

## The Deep AM-FM-PM Framework

### Adequacy Metric

This component aims to assess the semantic aspect of system responses, more specifically, how much source information is preserved by the dialogue generation with reference to human-written responses. The continuous space model is adopted for evaluating adequacy where good word-level or sentence-level embedding techniques are studied to measure the semantic closessness of system responses and human references in the continous vector space.

### Fluency Metric

This component aims to assess the syntactic validity of system responses. It tries to compare the system hypotheses against human references in terms of their respective sentence-level normalized log probabilities based on the assumption that sentences that are of similar syntactic validity should share similar perplexity level given by a language model. Hence, in this component, various language model techniques are explored to accurately estimate the sentence-level probability distribution.

### Pragmatics Metric

To be added


## Evaluation Procedure

### Toolkit Requirements

1. python 3.x
2. emoji=0.5.4
3. jsonlines=1.2.0
4. tensorflow-gpu=1.14.0

### Adequacy Metric

