from langsmith import evaluate, traceable
from src.pipeline import pipeline_enforced
from langsmith import Client
from pathlib import Path
import pandas as pd
from langchain_ollama import ChatOllama
from langsmith.utils import LangSmithConflictError
from pydantic import BaseModel
from typing import List
import ollama
import json


def valid_reasoning(inputs: dict, outputs: dict) -> bool:
    """
    Evaluates the logical validity of a given answer and its reasoning.

    :param inputs: A dictionary containing the question.
                   Example: {'question': 'What is 2 + 2?'}
    :param outputs: A dictionary containing the answer and reasoning.
                     Example: {'answer': '4', 'reasoning': '2 plus 2 equals 4.'}
    :return: True if the reasoning is logically valid, False otherwise.
    """
    instructions = """
    Given the following question, answer, and reasoning, determine if the reasoning for the answer is logically valid
    and consistent with the question and the answer.
    """

    class Response(BaseModel):
        reasoning_is_valid: bool

    msg = f"Question: {inputs.get('question', '')}\nAnswer & Reasoning: {outputs.get('answer', '')}"

    # Call the local Ollama LLM to judge the output using the Pydantic schema
    response = ollama.chat(
        model="deepseek-r1",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": msg}
        ],
        format=Response.model_json_schema(),
        options={"temperature": 0.0}
    )

    try:
        parsed_data = json.loads(response['message']['content'])
        return parsed_data.get("reasoning_is_valid", False)
    except json.JSONDecodeError:
        print("Failed to parse JSON from judge.")
        return False


def check_faithfulness(inputs: dict, outputs: dict) -> float:
    """
    :summary: Checks the faithfulness of an LLM's answer to a given context.

    :param inputs: A dictionary containing the 'context' and 'answer' keys,
                   where 'context' is the source text and 'answer' is the LLM's response.
                   The keys are strings.
    :param outputs: A dictionary containing the 'context' and 'answer' keys.
                   The keys are strings.
    :return: A float representing the faithfulness score, ranging from 0.0 to 1.0.
              A score of 1.0 indicates perfect faithfulness, while 0.0 indicates no faithfulness.
    """

    class ClaimVerification(BaseModel):
        claim: str
        is_supported: bool

    class FaithfulnessResponse(BaseModel):
        extracted_claims: List[ClaimVerification]

    instructions = """You are an expert grading a system's factual accuracy.
    Step 1: Extract all individual, verifiable claims made in the 'Answer'.
    Step 2: For each claim, check if it can be directly inferred from the provided 'Context'.

    Output a list of these claims. For each, set 'is_supported' to true ONLY if the context directly backs it up. Set it to false if it contains hallucinations or outside knowledge."""

    msg = f"Context: {outputs.get('context', '')}\nAnswer: {outputs.get('answer', '')}"

    response = ollama.chat(
        model="llama3.1",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": msg}
        ],
        format=FaithfulnessResponse.model_json_schema(),
        options={"temperature": 0.0}
    )

    try:
        parsed_data = json.loads(response['message']['content'])
        claims_list = parsed_data.get("extracted_claims", [])

        if not claims_list:
            return 0.0

        supported_count = sum(1 for item in claims_list if item.get("is_supported") is True)
        total_claims = len(claims_list)

        faithfulness_score = supported_count / total_claims

        return faithfulness_score

    except json.JSONDecodeError:
        print("Failed to parse LLM JSON output.")
        return 0.0

@traceable
def eval_model(inputs: dict) -> dict:
    """Evaluates a model based on the provided input.

    :param inputs: A dictionary containing the input data for the model.
                   It is expected to contain a 'question' key, which
                   represents the question to be answered by the model.
    :return: A dictionary containing the model's response to the question.
             The dictionary has one key, 'answer', which holds the
             model's response.
    """
    response = chain.invoke(inputs.get('question'))
    return {"answer": response.answer}


def evaluate_rag(data):
    """
    Evaluates a Retrieval-Augmented Generation (RAG) system.

    This function orchestrates the evaluation process by calling the
    `evaluate` function with the appropriate model and data. It utilizes
    the `valid_reasoning` evaluator to assess the generated output.

    :param data:
        The data to be processed by the RAG system.
        The type of this parameter is unspecified.
    :return:
        The evaluation result.
        The type of this return value is unspecified.
    """
    res = evaluate(
        eval_model,
        data=data,
        evaluators=[valid_reasoning, check_faithfulness],
        num_repetitions=2
    )

    return res


if __name__ == "__main__":
    parent_dir = Path(__name__).parent.resolve()
    client = Client()

    # The path to your local CSV file
    csv_file = str(parent_dir.parent / "evaluation.csv")
    df = pd.read_csv(csv_file)
    chain = pipeline_enforced(model=ChatOllama(model='gemma3', temperature=0))

    ds = []
    for entry in df.iterrows():
        question = entry[1]['question']
        answer = entry[1]['answer']
        context = entry[1]['contexts']
        res = {
            'inputs': {'question': question},
            'outputs': {'answer': answer, 'contexts': context},

        }

        ds.append(res)
    try:
        dataset = client.create_dataset(
            dataset_name='Statistics Evaluation',
            description='Statistical questions for RAG'
        )

        client.create_examples(
            dataset_id=dataset.id,
            examples=ds
        )
    except LangSmithConflictError:
        dataset = client.read_dataset(
            dataset_name='Statistics Evaluation'
        )
        client.create_examples(
            dataset_id=dataset.id,
            examples=ds
        )
        print(f"Dataset '{dataset.name}' already exists. Loaded from LangSmith!")

    print(f"Dataset created! ID: {dataset.id}")



    evaluate_rag(dataset) # Use LangSmith UI to see evaluation