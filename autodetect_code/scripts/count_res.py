import json

file_dir = f"../result/autodetect_gpt35"

categories = [
    "string_processing",
    "data_container_operations",
    "logic_and_control_flow",
    "functions_and_modules",
    "mathematics_and_algorithms",
    "data_structures",
    "file_handling",
]

res = {}
totall = 0
totall_scores = []
succcess = 0
tot_score_per_round = [[] for _ in range(15)]
for cat in categories:
    print(cat)
    res[cat] = {}
    res[cat]['cat'] = {}
    ver = 'version_0'
    with open(f'{file_dir}/{cat}/{ver}/log.json', 'r') as f:
        data = json.load(f)

    init_points = data['init_points']
    tot = 0
    success = 0
    tot_scores = []
    for point in init_points:
        point_data = data[point]['steps']
        point = point.split(':')[-1]
        res[cat]['cat'][point] = {}
        total_data = len(point_data)
        tot += total_data
        success_data = len([p for p in point_data if p['score'] < 3.1])
        success += success_data
        res[cat]['cat'][point]['total_data'] = total_data
        res[cat]['cat'][point]['success_data'] = success_data
        res[cat]['cat'][point]['asr'] = float(f'{success_data / total_data * 100:.2f}')
        scores = [p['score'] for p in point_data]
        tot_scores += scores
        for i in range(len(scores)):
            tot_score_per_round[i].append(scores[i])
        res[cat]['cat'][point]['avg_score'] = float(f'{sum(scores) / total_data:.2f}')

    res[cat]['total_data'] = tot
    res[cat]['success_data'] = success
    res[cat]['asr'] = float(f'{success / tot * 100:.2f}')
    res[cat]['avg_score'] = float(f'{sum(tot_scores) / tot:.2f}')
    totall += tot
    succcess += success
    totall_scores += tot_scores

res['overall_asr'] = float(f'{succcess / totall * 100:.2f}')
res['overall_avg_score'] = float(f'{sum(totall_scores) / totall:.2f}')
res['overall_avg_score_per_iter'] = [float(f'{sum(score) / len(score):.2f}') for score in tot_score_per_round]
res['total_data'] = totall

with open(f'../result/autodetect_gpt35/overall_data.json', 'w') as f:
    json.dump(res, f, ensure_ascii=False, indent=4)
