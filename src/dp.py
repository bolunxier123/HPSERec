import math
import pandas as pd
from tqdm import tqdm
from src.argument import parse_args


def creat_list(path):
    df = pd.read_csv(path, sep=' ', header=None)
    df.columns = ['userId', 'movieId','t']
    # 对 movieId 进行计数
    movie_counts = df['movieId'].value_counts()

    # 将计数结果从高到低排序，并转换为列表
    sorted_movie_counts = movie_counts.sort_values(ascending=True).reset_index()
    sorted_movie_counts.columns = ['movieId', 'count']

    # 将结果转换为列表，每个元素是一个包含 (movieId, count) 的元组
    result_list = list(sorted_movie_counts.itertuples(index=False, name=None))

    return df, result_list


def save_partitions(partitions, original_df, output_dir):
    """将分割的子数据集保存为 txt 文件，格式与原始文件一致"""
    for idx, partition in enumerate(partitions):
        # 获取当前子数组的 movieId 列表
        partition_movie_ids = [movie[0] for movie in partition]

        # 筛选原始数据中 movieId 在子数组中的行
        partition_df = original_df[original_df['movieId'].isin(partition_movie_ids)]

        # 为每个子数据集生成文件路径
        output_file = f"{output_dir}/{args.dataset}{idx}.txt"

        # 保存为无表头的 txt 文件
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
        return float('inf')  # 无数据，返回无穷大

    # 期望的平均交互量
    expected_count = total_sum / num_partitions

    # 基本熵计算
    entropy = -log_sum / total_count + math.log(total_count)


    # 均匀性正则化项（平方差）
    balance_regularization = gamma * ((total_count - expected_count) ** 2 / expected_count)

    # 最小化总得分
    score = entropy + balance_regularization
    return score


def min_score_partition(arr, T, score):
    n = len(arr)
    dp = [[float('inf')] * (T + 1) for _ in range(n + 1)]
    split_point = [[-1] * (T + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    # 填充 dp 表
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

    partitions.reverse()  # 反转顺序
    return dp[n][T], partitions


# 示例：使用参数解析器获取文件路径等配置
args = parse_args()
original_df, list = creat_list(f'../dataset/{args.dataset}.txt')

# 预处理前缀和
prefix_sum, prefix_log_sum = preprocess_prefix_sums(list)

# 使用快速熵计算函数
mark, partitions = min_score_partition(list, args.num_experts,
                                       lambda st, ed: score_shang(st, ed, prefix_sum, prefix_log_sum,
                                                                  prefix_sum[len(list) - 1], 4, alpha=5))  # 调整 alpha 参数

# save_partitions(partitions, original_df, f'../dataset/{args.dataset}0')

for i in range(1, len(partitions)):
    partitions[i] += partitions[i - 1]

save_partitions(partitions, original_df, f'../dataset/{args.dataset}')
