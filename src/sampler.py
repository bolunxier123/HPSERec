from collections import Counter
import numpy as np


class NegativeSampler(object):
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.usernum = dataset[4]  # User num
        self.itemnum = dataset[5]  # Item num

        self.negative_samples = self.get_random_negative('test')
        self.negative_samples_valid = self.get_random_negative('valid')

    def __call__(self, user, is_valid):
        if is_valid == 'test':
            return self.negative_samples[user]
        else:
            return self.negative_samples_valid[user]



    def get_random_negative(self, valid_or_test):
        np.random.seed(self.args.seed)
        negative_samples = {}

        for user in np.arange(1, self.usernum + 1):
            user_data = self.dataset[2].get(user, [])
            if len(user_data) < 1:
                continue

            seen = set(self.dataset[0].get(user, []))
            seen.add(0)

            samples = []

            if valid_or_test == 'test':
                seen.add(self.dataset[1].get(user, [0])[0])
                samples.append(user_data[0])
            else:
                samples.append(self.dataset[1].get(user, [0])[0])

            for _ in range(self.args.n_negative_samples):
                t = np.random.randint(1, self.itemnum + 1)
                while t in seen:
                    t = np.random.randint(1, self.itemnum + 1)
                samples.append(t)
                seen.add(t)

            negative_samples[user] = samples

        return negative_samples

