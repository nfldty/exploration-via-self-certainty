"""
Preprocess the MATH dataset to parquet format for RL training.
"""

import re
import os
import datasets
import argparse

from verl.utils.hdfs_io import copy, makedirs


def extract_boxed_answer(solution_str):
    """Extract the answer from \\boxed{...} in the solution."""
    boxed_matches = list(re.finditer(r'\\boxed\{', solution_str))
    if not boxed_matches:
        return None
    last_match = boxed_matches[-1]
    start = last_match.end()
    depth = 1
    i = start
    while i < len(solution_str) and depth > 0:
        if solution_str[i] == '{':
            depth += 1
        elif solution_str[i] == '}':
            depth -= 1
        i += 1
    if depth == 0:
        return solution_str[start:i - 1]
    return None


def make_prefix(problem, template_type):
    if template_type == 'base':
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: {problem}
Please solve this problem step by step. Put your final answer within \\boxed{{}}.
Assistant: Let me solve this step by step.
"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system
You are a helpful assistant that solves math problems. You first think about the reasoning process and then provide the answer.<|im_end|>
<|im_start|>user
{problem}
Please solve this problem step by step. Put your final answer within \\boxed{{}}.<|im_end|>
<|im_start|>assistant
"""
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/math')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')
    parser.add_argument('--train_size', type=int, default=None,
                        help='Max number of training samples (default: use all)')
    parser.add_argument('--test_size', type=int, default=500)

    args = parser.parse_args()

    data_source = 'math'
    dataset = datasets.load_dataset('hendrycks/competition_math')

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    if args.train_size is not None and args.train_size < len(train_dataset):
        train_dataset = train_dataset.select(range(args.train_size))
    if args.test_size is not None and args.test_size < len(test_dataset):
        test_dataset = test_dataset.select(range(args.test_size))

    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example['problem']
            solution = example['solution']
            answer = extract_boxed_answer(solution)
            if answer is None:
                answer = ""

            question = make_prefix(problem, template_type=args.template_type)
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer,
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'level': example.get('level', ''),
                    'type': example.get('type', ''),
                    'solution': solution,
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    os.makedirs(local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    print(f"Saved {len(train_dataset)} train and {len(test_dataset)} test samples to {local_dir}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
