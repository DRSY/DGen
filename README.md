# Knowledge-Driven Distractor Generation
AAAI 2021 paper ["Knowledge-Driven Distractor Generation for Cloze-style Multiple-choice Questions"](https://arxiv.org/abs/2004.09853)

# Web API for generating distractors with WordNet CSG
URL: http://202.120.38.146:10090/wordnet_dgen

Due to security issue, this service is now shut down. We will reopen it very soon!

## Example post request
**input**:
{
    "sentence": "The attraction between all objects in the universe is known as **blank**",
    "answer": "gravity",
    "num": 3
}


**result**
{
    "requested_k": 3,
    "source": "WordNet",
    "sent": "The attraction between all objects in the universe is known as **blank**",
    "ans": "gravity",
    "distractors": [
        "magnetism",
        "tension",
        "stress"
    ]
}


