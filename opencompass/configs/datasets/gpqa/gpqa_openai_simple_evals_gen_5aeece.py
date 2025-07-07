from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GPQADataset, GPQA_Simple_Eval_postprocess, GPQAEvaluator

confidence_prompt = '\nBased on your answer, please attach a confidence signal ranging from 1-10 to specify whether you are certain about your answer. 1 means you are totally uncertain (strong inconfidence), while 10 means you are totally certain (strong confidence). If you need more information to answer the question, please attach 1. We will compare your answer with the ground truth to check the correctness. If your answer is correct and accompanied by strong confidence, you will be rewarded; if your answer is incorrect but assigned strong confidence, you will be punished. The signal should be in the format of <CONFIDENCE:NUMBER>, where NUMBER ranges from 1 to 10, directly appended to your answer.\n'
combo_prompt = """Provide your best guess and the confidence that it is correct or plausible (1 to 100) for the following question. Take your uncertainty in the prompt, the task difficulty, your knowledge availability and other sources of uncertainty into account. Give

ONLY the guess and CONFIDENCE, no other words or explanation. For example: “Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!> 
<CONFIDENCE:NUMBER>, where NUMBER ranges from 1 to 100, directly appended to your answer

Here are five examples:

Question: The fox walked from the city into the forest, what was it looking for?

Choices:

A. pretty flowers.

B. hen house

C. natural habitat

D. storybook

E. dense forest

Guess: A

<CONFIDENCE: 47>

uestion: Which country is Europe’s largest silk producer?
Guess: Environment of Italy
<CONFIDENCE: 89>
Question: The population of the city where Michelle was
born is 145,826. What is the value of the 5 in the number 145,826?
Choices:
A. 5 thousands
B. 5 hundreds
C. 5 tens
D. 5 ones
Guess: A
CONFIDENCE: 77
Question: Beyond the business case for engaging in CSR
there are a number of moral arguments relating to: negative
_______, the _______that corporations possess and the ________
of business and society.
Choices:
A. Externalities, Power, Independence
B. Publicity, Insubstantial resources, Mutual dependence
C. Publicity, Power, Independence
D. Externalities, Power, Mutual dependence
Guess: B
<CONFIDENCE: 24>
Question: The Moon lacks weather and climate changes like
those on Earth. What causes the lack of weather on the Moon?
Guess: the lack of magnetic poles
<CONFIDENCE:8>
"""
# openai_simple_eval prompt
align_prompt = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()
align_prompt = confidence_prompt + align_prompt

gpqa_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D'],
    output_column='answer')

gpqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=align_prompt),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

gpqa_eval_cfg = dict(evaluator=dict(type=GPQAEvaluator),
                     pred_postprocessor=dict(type=GPQA_Simple_Eval_postprocess))

gpqa_datasets = []
gpqa_subsets = {
    # 'extended': 'gpqa_extended.csv',
    # 'main': 'gpqa_main.csv',
    'diamond': 'gpqa_diamond.csv'
}

for split in list(gpqa_subsets.keys()):
    gpqa_datasets.append(
        dict(
            abbr='GPQA_' + split,
            type=GPQADataset,
            path='./data/gpqa/',
            name=gpqa_subsets[split],
            reader_cfg=gpqa_reader_cfg,
            infer_cfg=gpqa_infer_cfg,
            eval_cfg=gpqa_eval_cfg)
    )
