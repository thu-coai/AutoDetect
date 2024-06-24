import requests
import json
import os
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import argparse


# TODO change the api key and url
API_KEY = '<API KEY>'
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}
API_URL = "<API URL>"

device = 'cuda:0'

# TODO update your model path, we implement the inference function for llama2 (llama2_generate) and llama3 (llama3_generate)
model_path = '<MODEL PATH>'


def gpt4_turbo_generate(text, temp=None, presence_penalty=None):
    # print(text)
    num = 50
    res = ""
    messages = [{"role": "user", "content": text}]
    while num > 0 and len(res)==0:
        try:
            if temp:
                data = json.dumps({"model": "gpt-4-1106-preview", "messages": 
                    [{"role": "user", "content": text}],
                    'temperature': temp
                })
            else:
                data = json.dumps({"model": "gpt-4-1106-preview", "messages": 
                    [{"role": "user", "content": text}]
                })
            response = requests.post(API_URL, headers=HEADERS, data=data)
            response_json = response.json()
            res = response_json['choices'][0]['message']['content']
        except Exception as e:
            print(e)
            num -= 1
    
    return res


def chatgpt_generate(text, temp=None, presence_penalty=None):
    # print(text)
    num = 50
    res = ""
    messages = [{"role": "user", "content": text}]
    while num > 0 and len(res)==0:
        try:
            if temp:
                data = json.dumps({"model": "gpt-3.5-turbo", "messages": 
                    [{"role": "user", "content": text}],
                    'temperature': temp
                })
            else:
                data = json.dumps({"model": "gpt-3.5-turbo", "messages": 
                    [{"role": "user", "content": text}]
                })
            response = requests.post(API_URL, headers=HEADERS, data=data)
            response_json = response.json()
            res = response_json['choices'][0]['message']['content']
        except Exception as e:
            print(e)
            num -= 1
    
    return res


def gpt4_generate(text, temp=1.0):
    # print(text)
    num = 50
    res = ""
    while num > 0 and len(res)==0:
        try:
            if temp:
                data = json.dumps({"model": "gpt-4", "messages": 
                    [{"role": "user", "content": text}],
                    'temperature': temp
                })
            else:
                data = json.dumps({"model": "gpt-4", "messages": 
                    [{"role": "user", "content": text}]
                })
            response = requests.post(API_URL, headers=HEADERS, data=data)
            response_json = response.json()
            res = response_json['choices'][0]['message']['content']
        except Exception as e:
            print(e)
            num -= 1
    
    return res


def get_gpt4_score(question, answer, ref_ans):
    prompt = {"name": "single-v1-ref", "type": "single", "system_prompt": "You are a helpful assistant.", "prompt_template": "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the correctness(high priority), helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. You will be given a high-quality reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer and identify the mistakes in the assistant's answer, then provide a short explanation. Be as objective as possible. Please be very careful in giving a 10. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n[Question]\n{question}\n\n[The Start of Reference Answer]\n{ref_answer}\n[The End of Reference Answer]\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]", "description": "Prompt for general questions", "category": "math", "output_format": "[[rating]]"}
    judge_prompt = prompt["prompt_template"].replace('{question}', question).replace('{ref_answer}', ref_ans).replace('{answer}', answer)
    score_res = gpt4_turbo_generate(judge_prompt, temp=0)
    return score_res


model = AutoModelForCausalLM.from_pretrained(model_path).half().eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def llama2_generate(text):
    input_text = "[INST] {} [/INST]".format(text)
    model_inputs = tokenizer(input_text, return_tensors="pt").to(device)    
    output = model.generate(**model_inputs, max_new_tokens=1024, do_sample=True, top_p=0.9, temperature=0.6, num_beams=1)
    resp = tokenizer.decode(output[0], skip_special_tokens=True).split('[/INST]')[1].strip()
    return resp


def llama3_generate(text):
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    input_text = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n".format(text)
    model_inputs = tokenizer(input_text, return_tensors="pt").to(device)    
    output = model.generate(**model_inputs, max_new_tokens=1024, do_sample=True, top_p=0.9, temperature=0.6, num_beams=1, eos_token_id=terminators)
    resp = tokenizer.decode(output[0][model_inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return resp


def deep_search(task_name, seed_prompts):
    history = []
    steps = []

    score_func = get_gpt4_score
    optimize_func = gpt4_turbo_generate

    final_data['search_optimize_func'] = str(optimize_func)
    final_data['score_func'] = str(score_func)

    for idx in range(len(seed_prompts)):
        i = seed_prompts[idx]
        ref_ans = optimize_func(i['prompt'])
        gen_res = llama2_generate(i['prompt'])
        seed_prompts[idx]['answer'] = gen_res
        seed_prompts[idx]['ref_ans'] = ref_ans
        i = seed_prompts[idx]
        for _ in range(3):
            try:
                score_res = score_func((i['prompt']).strip(), i['answer'], i['ref_ans'])
                i['comparison'] = score_res
                i['score'] = float(re.findall(r'\[\[.*?\]\]', score_res.strip())[-1].replace('[[', '').replace(']]', ''))
                break
            except Exception as e:
                print(e)

        seed_prompts[idx]['score'] = i['score']
        seed_prompts[idx]['comparison'] = i['comparison']
        history.append(i)
        steps.append(i)

    show_num = 5
    # for i in trange(10):
    while len(steps) < 15:
        print('current step:', len(steps))
        optimized_prompt = """This task involves generating test cases for the instruction-following task. Instruction following is an important topic in LLM research, where LLM should strcitly follow the instructions from human to complete a certain task.

Previous Prompts:\n\n"""

        history = sorted(history, key=lambda k: k['score'])

        tmp_h = []

        for j in history:
            # max_score = 0
            exact_match_start = False
            for k in tmp_h:
                if j['prompt'].split()[:3] == k['prompt'].split()[:3]:
                    exact_match_start = True
                    break
            if not exact_match_start:
                tmp_h.append(j)

        history = tmp_h[::-1]

        sample_his = []
        good_cases = [item for item in history]
        bad_cases = [item for item in history if item['score'] <= 3]
        
        if len(good_cases) < 5 or len(bad_cases) < 2:
            sample_his = history[-show_num:]
        else:
            sample_his = random.sample(bad_cases, 2)
            while len(sample_his) < 5:
                good_case = random.choice(good_cases)
                if good_case not in sample_his:
                    sample_his.append(good_case)
            sample_his = sorted(sample_his, key=lambda k: k['score'], reverse=True)


        for j in sample_his[-show_num:]:
            optimized_prompt += "Prompt: {}\nKey Point: {}\nScore: {}\n\n".format(j['prompt'], j['key_point'], j['score'])
        

        optimized_prompt += """The objective is to create new prompts that are challenging for the language model, with a focus on diverse types of instructions about "{task_name}". Each prompt should be solvable by a language model, complete, and aimed at achieving a lower score (indicating be difficult and complex).

Guidelines for Creating New Prompts:

1. Each prompt should be solvable by a language model (no visual task) and should contain all necessary information.
2. Aim for prompts that would result in a low score (less than 3.0), indicating a high level of complexity and difficulty.
3. Do not repeat verbs across different instructions to maximize diversity.
4. The point should be no more than 15 words and summarizes the key points of prompt.
5. Please focus on "{task_name}" constraints. And express the constraints explicitly in prompts.

Please generate a new test case. Output in json format, {"key_point": ..., "prompt": ...}"""

        try:
            optimized_prompt = optimized_prompt.replace(r"{task_name}", task_name)
            optimized_res = optimize_func(optimized_prompt)
            # from IPython import embed; embed() 
            pattern = r'```json\n(.+?)```'
            test_case = json.loads(re.search(pattern, optimized_res, re.DOTALL).group(1))
            new_prompt = test_case['prompt']
            key_point = test_case['key_point']
            ref_ans = optimize_func(new_prompt)
            gen_res = llama2_generate(new_prompt)

            score_res = score_func((new_prompt).strip(), gen_res, ref_ans)
            if len(re.findall(r'\[\[.*?\]\]', score_res.strip())) == 0:
                print("score invalid")
                continue
            score = float(re.findall(r'\[\[.*?\]\]', score_res.strip())[-1].replace('[[', '').replace(']]', ''))

            history.append({
                'prompt': new_prompt,
                'answer': gen_res,
                'ref_ans': ref_ans,
                'comparison': score_res,
                'key_point': key_point,
                'score': score
            })

            steps.append({
                'prompt': new_prompt,
                'answer': gen_res,
                'ref_ans': ref_ans,
                'comparison': score_res,
                'key_point': key_point,
                'score': score
            })

            if 'optimize_prompt' not in final_data:
                final_data['optimize_prompt'] = optimized_prompt
            final_data[task_name]['steps'] = steps
            
            with open(f'{output_path}/log.json', 'w', encoding='utf-8') as f:
                json.dump(final_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(e)
            continue


def gen_seed(task_name, categories):
    prompt_template = """Instruction following is an important topic in LLM research, where LLM should strictly follow the instructions from human to complete a certain task. The task types of instruction following include generation, openqa, rewrite, brainstorming and so on.
Here is a taxonomy for instruction-following task:
{categories}

Based on this, please generate 5 test case of "{task_name}" category to test if language models can follow prompts with "{task_name}" constraint. Key point is a short sentence summarizes the key point you want to test the language model. The constraints on "{task_name}" should be explicitly expressed. Besides, your test cases should cover different task types mentioned before to increase prompt diversity. Please be as diverse as you can but focus on "{task_name}" and ensure the prompt is text-only (no multimodal). The answer of these test cases are expected not to be too long.

You should ONLY output the test cases in json format, {"test_case1": {"key_point": ..., "prompt": ...}, ...}"""
    res = []
    for _ in range(5):
        try:
            prompts = gpt4_generate(prompt_template.replace(r"{task_name}", task_name).replace(r"{categories}", json.dumps(categories)))
            prompts = json.loads(prompts)
            for k in prompts.keys():
                assert ("key_point" in prompts[k])
                assert ("prompt" in prompts[k])
                res.append(prompts[k])
            break
        except Exception as e:
            print(e)
            continue
    return res
    
def analysis(task_names):
    prompt_template = """Instruction following is an important topic in LLM research, where LLM should strictly follow the instructions from human to complete a certain task. The task types of instruction following include generation, openqa, rewrite, brainstorming and so on.

Here is a sub task's taxonomy as well as the averaged score on these tasks(lower means worse performance):
{taxonomy}

And here is some bad cases:
{bad_cases}
Based on the given information, please judge if the taxonomy is comprehensive, if so please just output [[Stop]]. 

If not, please give me a new possible issue you inferred from present taxonomy and bad cases. Plaese focus on {main_task}. Ensure the new task is text-only (no multimodal). Also give a brief explanation of how you find the issue. Please output in json format, {"task_name": ..., "explanation":...}"""
    
    bad_cases = {}
    main_task = task_names[0].split(':')[0]
    sub_tax = {}
    for i in task_names:
        task_name = i
        scores = [float(j['score']) for j in final_data[task_name]['steps']]
        sub_task_name = task_name.split(':')[1]
        sub_tax[sub_task_name] = sum(scores) / len(scores)
        bad_cases[sub_task_name] = []
        for j in final_data[task_name]['steps']:
            if float(j['score']) <= 3.0:
                bad_cases[sub_task_name].append(j)
    
    bad_cases_str = ""
    for k in bad_cases.keys():
        if len(bad_cases[k]) == 0:
            continue
        samples = random.sample(bad_cases[k], min(2, len(bad_cases[k])))
        bad_cases_str += f"Task Name: {k}\nSamples:\n"
        for i in samples:
            bad_cases_str += "Prompt: {}\nResponse: {}\nScore: {}\n\n".format(i['prompt'], i['answer'], i['score'])
        
    tax = {main_task: sub_tax}
    for _ in range(3):
        try:
            new_task = gpt4_generate(prompt_template.replace(r"{taxonomy}", json.dumps(tax)).replace(r"{bad_cases}", bad_cases_str).replace(r"{main_task}", main_task))
            if "[[Stop]]" in new_task:
                return "[[Stop]]"
            new_task = json.loads(new_task)
            final_data['new_points'].append(new_task)
            return new_task['task_name']
        except Exception as e:
            print(e)


def judge_new_task(task_names, new_point):
    prompt_template = """Instruction following is an important topic in LLM research, where LLM should strictly follow the instructions from human to complete a certain task. The task types of instruction following include generation, openqa, rewrite, brainstorming and so on.

Here is a sub task's taxonomy on the task "{main_task}":
{taxonomy}

Based on the given taxonomy, please judge whether the new test point "{new_point}" is suitable as a sub task on the task "{main_task}". The judge criteria are as following:
1. The new test point should precisely cover an important and meaningful part of the main task.
2. The new test point should be sufficiently different from the existing test points.
3. The new test point should be text-only (no multimodal).

If the new test point "{new_point}" is suitable as a sub task on the task "{main_task}", please ONLY output [[Yes]]. If not, please first output [[No]], and then provide the reason why it's not suitable as a sub task on the task "{main_task}"."""
    main_task = task_names[0].split(':')[0]
    sub_tax = []
    for i in task_names:
        task_name = i
        sub_task_name = task_name.split(':')[1]
        sub_tax.append(sub_task_name)

        
    tax = {main_task: sub_tax}
    for i in range(3):
        try:
            judge_res = gpt4_generate(prompt_template.replace(r"{taxonomy}", json.dumps(tax)).replace(r"{new_point}", new_point).replace(r"{main_task}", main_task), temp=0)
            if "[[Yes]]" in judge_res or ("Yes" in judge_res and "[[No]]" not in judge_res):
                return True
            print(judge_res)
            return False
        except Exception as e:
            print(e)




if __name__ == '__main__':
    with open('../data/if_cat.json', 'r') as f:
        categories = json.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument('--category', default=None, type=str)
    parser.add_argument('--task', default=None, type=str)
    args = parser.parse_args()

    main_cat = args.category
    main_task = args.task.replace('_', ' ')
    points = list(categories['instruction_following'][main_cat][main_task].keys())
    test_points = [f'{main_task}:{point}' for point in points]
    
    output_dir = f'../result/autodetect/{args.task}/'

    num = 0
    output_path = ''
    while True:
        folder_name = f'version_{num}'
        output_path = f'{output_dir}{folder_name}'
        num += 1
        if os.path.exists(output_path):
            continue
        else:
            break
    
    os.makedirs(output_path, exist_ok=True)

    final_data = {'init_points': test_points, 'new_points': []}
    idx = 0
    while idx < len(test_points) and idx <= 5:
        
        task = test_points[idx]
        print(f'Begin gen seed: {task}')
        seeds = gen_seed(task, categories)

        final_data[task] = {
            'seed_prompts': seeds
        }
        with open(f'{output_path}/log.json', 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)
        
        deep_search(task, seeds)
        
        if idx == len(test_points) - 1:
            for x in range(3):
                new_task = analysis(test_points)
                if new_task == '[[Stop]]':
                    print('Encounter stop. End circuit.')
                    exit(0)
                if not judge_new_task(test_points, new_task):
                    if x < 2: continue
                    print('Reject three times. End circuit.')
                    exit(0)
                categories['instruction_following'][main_cat][main_task][new_task] = {}
                test_points.append(main_task+':'+new_task)
                break

        
        idx += 1    