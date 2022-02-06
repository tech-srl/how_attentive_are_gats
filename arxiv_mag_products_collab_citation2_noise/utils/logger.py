import torch
import time


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]
        self.start_time = [None for _ in range(runs)]

    def reset(self, run):
        assert len(self.results) > run >= 0
        self.start_time[run] = None
        self.results[run] = []

    def set_time(self, run):
        self.start_time[run] = time.time()

    def add_result(self, run, result):
        assert len(result) == 3
        assert len(self.results) > run >= 0
        if self.start_time[run] is None:
            self.start_time[run] = time.time()
        self.results[run].append(result)

    def get_time_elapsed(self, run=0):
        start = self.start_time[run]
        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{int(hours):0>2}:{int(minutes):0>2}:{int(seconds):0>2}"

    def print_csv(self, results):
        print("split, model, dataset, metric, run")
        for i, (_, val, train, test) in enumerate(results):

            print(f"train, {self.info.type}, {self.info.dataset}, {train}, {i + 1}")
            print(f"val, {self.info.type}, {self.info.dataset}, {val}, {i + 1}")
            print(f"test, {self.info.type}, {self.info.dataset}, {test}, {i + 1}")

    def print_statistics(self, run=None):
        if run is not None:
            time_elapsed = self.get_time_elapsed(run)
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Time Elapsed:  {time_elapsed}')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            time_elapsed = self.get_time_elapsed()
            best_result_list = []
            for r in self.results:
                r = 100 * torch.tensor(r)
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_result_list.append((train1, valid, train2, test))

            best_result = torch.tensor(best_result_list)
            print(f'{self.info.dataset} {self.info.type} All runs:')
            print(self.info)
            print(f'All runs:')
            print(f'Time Elapsed:  {time_elapsed}')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
            print('')
            self.print_csv(best_result_list)
