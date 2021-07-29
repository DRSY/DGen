# Knowledge-Driven Distractor Generation
AAAI 2021 paper ["Knowledge-Driven Distractor Generation for Cloze-style Multiple-choice Questions"](https://ojs.aaai.org/index.php/AAAI/article/view/16559/16366)

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

**How to cite**
```latex
@article{Ren_Q. Zhu_2021, 
    title={Knowledge-Driven Distractor Generation for Cloze-Style Multiple Choice Questions}, 
    volume={35}, 
    url={https://ojs.aaai.org/index.php/AAAI/article/view/16559}, 
    abstractNote={In this paper, we propose a novel configurable framework to automatically generate distractive choices for open-domain cloze-style multiple-choice questions. The framework incorporates a general-purpose knowledge base to effectively create a small distractor candidate set, and a feature-rich learning-to-rank model to select distractors that are both plausible and reliable. Experimental results on a new dataset across four domains show that our framework yields distractors outperforming previous methods both by automatic and human evaluation. The dataset can also be used as a benchmark for distractor generation research in the future.}, 
    number={5}, 
    journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
    author={Ren, Siyu and Q. Zhu, Kenny}, 
    year={2021}, 
    month={May}, 
    pages={4339-4347} }
```


