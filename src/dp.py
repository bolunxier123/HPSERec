import math
import pandas as pd
from tqdm import tqdm
from src.argument import parse_args


def creat_list(path):
    df = pd.read_csv(path, sep=' ', header=None)
    df.columns = ['userId', 'movieId', 't']
    movie_counts = df['movieId'].value_counts()

    sorted_movie_counts = movie_counts.sort_values(ascending=True).reset_index()
    sorted_movie_counts.columns = ['movieId', 'count']

    # 将结果转换为列表，每个元素是一个包含 (movieId, count) 的元组
    result_list = list(sorted_movie_counts.itertuples(index=False, name=None))

    return df, result_list


def save_partitions(partitions, original_df, output_dir):
    for idx, partition in enumerate(partitions):
        partition_movie_ids = [movie[0] for movie in partition]

        partition_df = original_df[original_df['movieId'].isin(partition_movie_ids)]

        output_file = f"{output_dir}/{args.dataset}{idx}.txt"

        partition_df.to_csv(output_file, sep=' ', index=False, header=False)


def preprocess_prefix_sums(movie_list):
    prefix_sum = [0] * len(movie_list)
    prefix_log_sum = [0] * len(movie_list)

    for i, (_, count) in enumerate(movie_list):
        prefix_sum[i] = prefix_sum[i - 1] + count if i > 0 else count
        prefix_log_sum[i] = prefix_log_sum[i - 1] + count * math.log(count) if i > 0 and count > 0 else 0

    return prefix_sum, prefix_log_sum


def score_shang(st, ed, prefix_sum, prefix_log_sum, total_sum, num_partitions, gamma=1.0):
    if st == 0:
        total_count = prefix_sum[ed]
        log_sum = prefix_log_sum[ed]
    else:
        total_count = prefix_sum[ed] - prefix_sum[st - 1]
        log_sum = prefix_log_sum[ed] - prefix_log_sum[st - 1]

    if total_count == 0:
        return float('inf')

    expected_count = total_sum / num_partitions

    entropy = -log_sum / total_count + math.log(total_count)

    balance_regularization = gamma * ((total_count - expected_count) ** 2 / expected_count)

    score = entropy + balance_regularization
    return score


def min_score_partition(arr, T, score):
    n = len(arr)
    dp = [[float('inf')] * (T + 1) for _ in range(n + 1)]
    split_point = [[-1] * (T + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    for t in range(1, T + 1):
        for i in tqdm(range(1, n + 1)):
            for j in range(t - 1, i):
                score_val = dp[j][t - 1] + score(j, i - 1)
                if score_val < dp[i][t]:
                    dp[i][t] = score_val
                    split_point[i][t] = j

    partitions = []
    i = n
    t = T
    while t > 0:
        j = split_point[i][t]
        partitions.append(arr[j:i])
        i = j
        t -= 1

    partitions.reverse()
    return dp[n][T], partitions


args = parse_args()
original_df, list = creat_list(f'../dataset/{args.dataset}.txt')

prefix_sum, prefix_log_sum = preprocess_prefix_sums(list)

mark, partitions = min_score_partition(list, args.num_experts,
                                       lambda st, ed: score_shang(st, ed, prefix_sum, prefix_log_sum,
                                                                  prefix_sum[len(list) - 1], 4,))

for i in range(1, len(partitions)):
    partitions[i] += partitions[i - 1]

save_partitions(partitions, original_df, f'../dataset/{args.dataset}')
