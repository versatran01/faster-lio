#!/usr/bin/env python

from scipy.spatial.transform import Rotation
import numpy as np

from pathlib import Path
from dataclasses import dataclass
from evo.core import metrics, sync
from evo.tools import file_interface, plot
import matplotlib.pyplot as plt
import pickle


def compute_traj_len(xyzs: np.ndarray) -> float:
    assert xyzs.shape[1] == 3
    assert xyzs.ndim == 2
    diff = np.diff(xyzs, axis=0)
    square = diff ** 2
    sqnorm = np.sum(square, axis=1)
    return np.sum(sqnorm ** 0.5)


@dataclass
class Results:
    """Class for book keeping results"""
    method: str
    ape_rmse: float
    rpe_rmse: float
    ape_rmse_rot: float
    rpe_rmse_rot: float
    gt_len: int
    method_len: int

    def __init__(self, method: str, ape_rmse: float, rpe_rmse: float, ape_rmse_rot: float, rpe_rmse_rot: float,
                 gt_len: int, method_len: int):
        self.method = method
        self.ape_rmse = ape_rmse
        self.rpe_rmse = rpe_rmse
        self.ape_rmse_rot = ape_rmse_rot
        self.rpe_rmse_rot = rpe_rmse_rot
        self.gt_len = gt_len
        self.method_len = method_len


class ResultAccumulator:

    def __init__(self):
        self.ape_rmse_tr = []
        self.ape_rmse_rot = []
        self.rpe_rmse_tr = []
        self.rpe_rmse_rot = []

    def add_results(self, dataset: str, method: str, results: dict, threshold, constant_error):
        for seq_k in results.keys():
            print(f'Key: {seq_k}')
            sequence = results[seq_k]
            print(f'Sequence: {sequence}\n')
            if sequence.method_len / sequence.gt_len >= threshold:
                # if sequence.method_len == sequence.gt_len:
                self.ape_rmse_tr.append(sequence.ape_rmse)
                self.ape_rmse_rot.append(sequence.ape_rmse_rot)
                self.rpe_rmse_tr.append(sequence.rpe_rmse)
                self.rpe_rmse_rot.append(sequence.rpe_rmse_rot)
            else:
                print(f"{method} {seq_k} was incomplete. GT: {sequence.gt_len} Method: {sequence.method_len}")
                self.ape_rmse_tr.append(constant_error[0])
                self.ape_rmse_rot.append(constant_error[1])
                self.rpe_rmse_tr.append(constant_error[2])
                self.rpe_rmse_rot.append(constant_error[3])


class CumulativePlotter:

    def __init__(self, traj_norm: bool):
        fig, axs = plt.subplots(2, 2)
        self.axs = axs
        self.fig = fig
        self.traj_norm = traj_norm

    def add_data(self, name: str, result_data: ResultAccumulator, color: str, error_list):
        error_less_count_ape_rmse_t = np.sum(np.array(result_data.ape_rmse_tr)[:, None] < error_list[0], axis=0)
        if self.traj_norm:
            error_less_count_ape_rmse_t = (error_less_count_ape_rmse_t / len(result_data.ape_rmse_tr)) * 100
            # self.axs[0, 0].set_xlabel('Error (%)')
        else:
            error_less_count_ape_rmse_t = error_less_count_ape_rmse_t / len(result_data.ape_rmse_tr)
            # self.axs[0, 0].set_xlabel('Error (m)')
        self.axs[0, 0].plot(error_list[0] * 100, error_less_count_ape_rmse_t, color, label=name)
        self.axs[0, 0].set_title('Translational APE')
        self.axs[0, 0].set_ylabel('Percentage of runs')
        self.axs[0, 0].legend(loc='lower right')

        error_less_count_ape_rmse_r = np.sum(np.array(result_data.ape_rmse_rot)[:, None] < error_list[1], axis=0)
        self.axs[0, 1].plot(error_list[1], (error_less_count_ape_rmse_r / len(result_data.ape_rmse_rot)) * 100, color,
                            label=name)
        self.axs[0, 1].set_title('Rotational APE')
        # self.axs[0, 1].set_xlabel('Error (deg)')
        # self.axs[0, 1].set_ylabel('Percentage of runs')
        self.axs[0, 1].legend(loc='lower right')

        error_less_count_rpe_rmse_t = np.sum(np.array(result_data.rpe_rmse_tr)[:, None] < error_list[2], axis=0)
        if self.traj_norm:
            error_less_count_rpe_rmse_t = (error_less_count_rpe_rmse_t / len(result_data.rpe_rmse_tr)) * 100
            self.axs[1, 0].set_xlabel('Error (%)')
        else:
            error_less_count_rpe_rmse_t = error_less_count_rpe_rmse_t / len(result_data.rpe_rmse_tr)
            self.axs[1, 0].set_xlabel('Error (m)')
        self.axs[1, 0].plot(error_list[2] * 100, error_less_count_rpe_rmse_t, color, label=name)
        self.axs[1, 0].set_title('Translational RPE')
        self.axs[1, 0].set_ylabel('Percentage of runs')
        self.axs[1, 0].legend(loc='lower right')

        error_less_count_rpe_rmse_r = np.sum(np.array(result_data.rpe_rmse_rot)[:, None] < error_list[3], axis=0)
        self.axs[1, 1].plot(error_list[3], (error_less_count_rpe_rmse_r / len(result_data.rpe_rmse_rot)) * 100, color,
                            label=name)
        self.axs[1, 1].set_title('Rotational RPE')
        self.axs[1, 1].set_xlabel('Error (deg)')
        # self.axs[1, 1].set_ylabel('Percentage of runs')
        self.axs[1, 1].legend(loc='lower right')

    def plot_figure(self, save_fig: bool):
        plt.tight_layout()
        if save_fig:
            plt.savefig("/tmp/cumulative_plot.pdf", bbox_inches='tight')
        else:
            plt.show()


class EvalMethod:

    def __init__(self, base_dir: str, dataset: str, method: str):
        self.base_dir = base_dir
        self.dataset = dataset
        self.method = method
        self.method_path = Path(f'{self.base_dir}/{self.dataset}/{self.method}')
        self.gt_path = Path(f'{self.base_dir}/{self.dataset}/gt')
        self.gt_list = []
        self.method_list = []
        self.results_dict = {}

        # Threshold for maximum difference in timestamps for sync
        self.evo_config_t_max_diff = 0.05

        # Threshold for minimum ratio of valid poses after sync
        self.evo_config_pose_ratio = 0.8

        self.save_dir = Path(f"{self.base_dir}/{self.dataset}/{self.method}/eval")
        self.save_dir.mkdir(exist_ok=True, parents=True)

        for gt_p in self.gt_path.iterdir():
            if gt_p.is_file():
                self.gt_list.append(gt_p)
        self.gt_list.sort()

        for m_p in self.method_path.iterdir():
            if m_p.is_file():
                self.method_list.append(m_p)
        self.method_list.sort()

        print(f"Identified {len(self.gt_list)} GT files and {len(self.method_list)} {self.method} files.")

    def evaluate_pair(self, method_path: Path, gt_path: Path, normalize_trajectory: bool) -> Results:
        gt_in = np.loadtxt(gt_path)
        method_in = np.loadtxt(method_path)

        traj_ref = file_interface.read_tum_trajectory_file(gt_path)
        traj_method = file_interface.read_tum_trajectory_file(method_path)

        # Trajectories are associated using timestamps and a threshold
        traj_ref, traj_method = sync.associate_trajectories(traj_ref, traj_method, self.evo_config_t_max_diff)

        # Alignment is calculated in closed form using Umeyama's method [Umeyama-1991]
        traj_method.align(traj_ref, correct_scale=False, correct_only_scale=False)

        assert (traj_ref.num_poses / traj_method.num_poses > self.evo_config_pose_ratio)

        ape_metric_method = metrics.APE(metrics.PoseRelation.translation_part)
        ape_metric_method.process_data((traj_ref, traj_method))
        ape_rmse_method = ape_metric_method.get_statistic(metrics.StatisticsType.rmse)

        rpe_metric_method = metrics.RPE(metrics.PoseRelation.translation_part)
        rpe_metric_method.process_data((traj_ref, traj_method))
        rpe_rmse_method = rpe_metric_method.get_statistic(metrics.StatisticsType.rmse)

        if normalize_trajectory:
            traj_len = compute_traj_len(gt_in[:, 1:4])
            ape_rmse_method /= traj_len
            rpe_rmse_method /= traj_len

        ape_metric_method_rot = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
        ape_metric_method_rot.process_data((traj_ref, traj_method))
        ape_rmse_method_rot = ape_metric_method_rot.get_statistic(metrics.StatisticsType.rmse)

        rpe_metric_method_rot = metrics.RPE(metrics.PoseRelation.rotation_angle_deg)
        rpe_metric_method_rot.process_data((traj_ref, traj_method))
        rpe_rmse_method_rot = rpe_metric_method_rot.get_statistic(metrics.StatisticsType.rmse)

        res = Results(method=self.method, ape_rmse=ape_rmse_method, rpe_rmse=rpe_rmse_method,
                      ape_rmse_rot=ape_rmse_method_rot,
                      rpe_rmse_rot=rpe_rmse_method_rot, gt_len=gt_in.shape[0], method_len=method_in.shape[0])

        return res

    def evaluate_all(self, normalize_trajectory: bool):
        for method in self.method_list:
            gt = self.gt_path / f"{method.stem.replace(self.method + '_', '')}_tum.csv"
            assert (gt.is_file())
            res_pair = self.evaluate_pair(method, gt, normalize_trajectory)
            self.results_dict[gt.stem] = res_pair

    def save_results(self):
        if self.results_dict:
            save_path = self.save_dir / "evaluate.pkl"
            outfile = open(save_path, 'wb')
            pickle.dump(self.results_dict, outfile)
            outfile.close()

    def load_results(self):
        load_path = self.save_dir / "evaluate.pkl"
        with open(load_path, 'rb') as f:
            p_data = pickle.load(f)
        if not self.results_dict:
            self.results_dict = p_data

    def viz_matrix_ape_rmse(self, thresh_complete, constant_error):
        ape_rmse_list = []
        for seq_k in self.results_dict.keys():
            print(f'Key: {seq_k}')
            sequence = self.results_dict[seq_k]
            print(f'Sequence: {sequence}\n')
            if sequence.method_len / sequence.gt_len >= thresh_complete:
                ape_rmse_list.append(sequence.ape_rmse)
            else:
                ape_rmse_list.append(constant_error)

        ape_rmse_viz_vector = np.tile(np.asarray(ape_rmse_list), (10, 1))
        fig = plt.figure()
        plt.imshow(ape_rmse_viz_vector, cmap='jet', interpolation='nearest')
        plt.colorbar()
        plt.grid(None)
        plt.show()

    def viz_rmse(self):
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot([data.ape_rmse for data in self.results_dict.values()])
        axs[0, 0].set_title('Translational APE')
        axs[0, 0].set_ylabel('RMSE (m)')
        axs[0, 0].set_xlabel('Sequence')
        axs[0, 1].plot([data.ape_rmse_rot for data in self.results_dict.values()])
        axs[0, 1].set_title('Rotational APE')
        axs[0, 1].set_ylabel('RMSE (deg)')
        axs[0, 1].set_xlabel('Sequence')
        axs[1, 0].plot([data.rpe_rmse for data in self.results_dict.values()])
        axs[1, 0].set_title('Translational RPE')
        axs[1, 0].set_ylabel('RMSE (m)')
        axs[1, 0].set_xlabel('Sequence')
        axs[1, 1].plot([data.rpe_rmse_rot for data in self.results_dict.values()])
        axs[1, 1].set_title('Rotational RPE')
        axs[1, 1].set_ylabel('RMSE (deg)')
        axs[1, 1].set_xlabel('Sequence')
        plt.show()


if __name__ == "__main__":
    arr = np.array([[0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 5]])
    print(compute_traj_len(arr))
