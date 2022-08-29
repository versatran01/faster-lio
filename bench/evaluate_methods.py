from eval import EvalMethod, CumulativePlotter, ResultAccumulator
import numpy as np
import matplotlib.pyplot as plt


def load_evaluation(basedir: str, dataset: str, method: str, normalize_traj: bool) -> EvalMethod:
    evaluator = EvalMethod(basedir, dataset, method)
    if (evaluator.save_dir / "evaluate.pkl").is_file():
        print(f"Loading {method} - {dataset} data from pickle.")
        evaluator.load_results()
    else:
        print(f"Generating {method} - {dataset} data and saving to pickle.")
        evaluator.evaluate_all(normalize_trajectory=normalize_traj)
        evaluator.save_results()
    return evaluator


if __name__ == "__main__":
    traj_norm = False

    # LOAD DATA

    fasterlio_eval_os0 = load_evaluation('/tmp/rofl_results', 'os0', 'fasterlio', normalize_traj=traj_norm)
    # fasterlio_eval_os0.viz_rmse()
    fasterlio_eval_os1 = load_evaluation('/tmp/rofl_results', 'os1', 'fasterlio', normalize_traj=traj_norm)
    # fasterlio_eval_os1.viz_rmse()

    # rofl_eval_os0= load_evaluation('/tmp/rofl_results', 'os0', 'rofl', normalize_traj=traj_norm)
    # rofl_eval_os0.viz_rmse()
    # rofl_eval_os1 = load_evaluation('/tmp/rofl_results', 'os1', 'rofl', normalize_traj=traj_norm)
    # rofl_eval_os1.viz_rmse()

    # ACCUMULATE DATA

    traj_thresh = 0.8
    error_caps = [0.5, 90, 0.5, 45]

    fasterlio_all = ResultAccumulator()
    fasterlio_all.add_results('os0', 'fasterlio', fasterlio_eval_os0.results_dict, traj_thresh, error_caps)
    fasterlio_all.add_results('os1', 'fasterlio', fasterlio_eval_os1.results_dict, traj_thresh, error_caps)

    # dsol_all = ResultAccumulator()
    # dsol_all.add_results('os0', 'rofl', rofl_eval_os0.results_dict, traj_thresh, error_caps)
    # dsol_all.add_results('os1', 'rofl', rofl_eval_os1.results_dict, traj_thresh, error_caps)

    # #key_list = [*sdso_eval_vkitti.results_dict.keys()]

    # PLOT CUMULATIVE DATA

    error_list_ape_t = np.arange(0, 0.1, 0.00001)
    error_list_ape_r = np.arange(0, 90, 0.01)
    error_list_rpe_t = np.arange(0, 0.1, 0.00001)
    error_list_rpe_r = np.arange(0, 5, 0.01)
    error_list = [error_list_ape_t, error_list_ape_r, error_list_rpe_t, error_list_rpe_r]

    plotter = CumulativePlotter(traj_norm=traj_norm)
    plotter.add_data('fasterlio', fasterlio_all, 'r', error_list)
    # plotter.add_data('rofl', rofl_all, 'b', error_list)

    for ax in plotter.axs.ravel():
        ax.set_ylim([0, 101])
    ylimb = 0
    plotter.axs[0, 0].set_xlim([ylimb, 10])
    plotter.axs[0, 1].set_xlim([ylimb, 60])
    plotter.axs[1, 0].set_xlim([ylimb, 1])
    plotter.axs[1, 1].set_xlim([ylimb, 3])

    plotter.plot_figure(save_fig=True)
