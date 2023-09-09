from abc import abstractmethod
import numpy as np
from scipy.optimize import dual_annealing

from pi0fit.pi0_model import Pi0Model
from pi0fit.fitter_utilities import FitResults


class Pi0MinimizerBase(FitResults):
    """
    Base class defining the structure of the concrete minimizer classes.
    """
    def __init__(self, config):

        self.config = config["pi0_minimizers"]

        self.debug = self.config["debug"]
        self.pi0_model = Pi0Model(config)

        try:
            self.out_file = self.config["result_file"]
        except KeyError:
            self.out_file = None

    @abstractmethod
    def minimize(self, pi0_points, truth_values):
        """
        Base class top level function
        :return:
        """
        pass

    @abstractmethod
    def show_results(self, minimizer, truth_values):
        """
        Base class to print minimization results.
        :param truth_values:
        :param minimizer: minimizer object
        :return:
        """
        pass


class DualAnnealingMinimizer(Pi0MinimizerBase):

    def __init__(self, config):
        super().__init__(config)

        self.config = config["pi0_minimizers"]["dual_annealing"]

        self.anealing_temp = self.config["annealing_temp"] #10k
        self.maxiter = self.config["maxiter"] # 2k
        self.use_scan_start = self.config["use_scan_start"]

    def minimize(self, pi0_points, truth_values):

        epi0_bound = tuple(self.config["epi0_limits"])  #(240, 2500)
        cos_pi0_bound = tuple(self.config["cos_pi0_limits"]) #(0., 1)
        eg1_bound = tuple(self.config["eg1_limits"]) #(120, 2200)
        c1_bound = tuple(self.config["c1_limits"])   #(0, 80)
        c2_bound = tuple(self.config["c2_limits"])   #(0, 80)
        bounds = [epi0_bound, cos_pi0_bound, eg1_bound, c1_bound, c2_bound]

        if self.use_scan_start:
            print("Starting Scan")
            _, scan_params = self.brute_force_scan(shower_pts=pi0_points, conv_dist=None, out_file=out_file, full_fit=True)
            start_pt = np.array([scan_params[0], scan_params[1], scan_params[2], scan_params[3], scan_params[4]])
        else:
            start_pt = np.array([1200, 0.9, 600, 12, 18])

        if self.out_file is None:
            print("Start Point:", start_pt)
        else:
            print("Start Point:", start_pt, file=self.out_file)

        print("Starting Minimization")
        min_result = dual_annealing(self.model_interface, args=np.array([pi0_points]), bounds=bounds,
                                    maxiter=self.maxiter, x0=start_pt, initial_temp=self.anealing_temp)

        self.show_results(minimizer=min_result, truth_values=truth_values)

    def show_results(self, minimizer, truth_values):
        """
        Print reco fit results and compare to truth values.
        :param minimizer:
        :param truth_values:
        :return:
        """
        self.set_event_values(epi0=minimizer.x[0], cos_pi0=minimizer.x[1], eg1=minimizer.x[2],
                              c1=minimizer.x[3], c2=minimizer.x[4], is_truth=False)

        self.print_comparison_table(fit_result=truth_values)
        self.calculate_metrics(fit_result=truth_values)

    def model_interface(self, x, pi0_points):

        epi0, cos_pi0, eg1, c1, c2 = x

        if self.debug:
            print("x", x)
            print("epi0, cos_pi0, eg1", epi0, cos_pi0, eg1)
            print("shower_pts", pi0_points)
            print("shower_pts.shape", pi0_points.shape)

        # Energy constraint
        if (epi0 - eg1) < 120.:
            epi0 += abs(epi0 - eg1 - 120) / 2  # 250, 120
            eg1 -= abs(epi0 - eg1 - 120) / 2
            tmp_res = self.pi0_model.pi0_model(pi0_points, epi0=epi0, cos_pi0=cos_pi0, eg1=eg1, c1=c1, c2=c2,)
            res = tmp_res * np.exp((epi0 - eg1) - 120)
        else:
            res = self.pi0_model.pi0_model(pi0_points, epi0=epi0, cos_pi0=cos_pi0, eg1=eg1, c1=c1, c2=c2)

        log_lh = np.log(np.nan_to_num(res, nan=1.e-200) + 1.e-200)
        return -np.sum(log_lh)


class LikelihoodScan(Pi0MinimizerBase):

    def __init__(self, config):
        super().__init__(config)

        self.config = config["pi0_minimizers"]["likelihood_scan"]

        def brute_force_scan(self, shower_pts, conv_dist, out_file, full_fit=False):
            """
            Warning this is slowww!
            :param shower_pts:
            :param conv_dist:
            :param out_file:
            :param full_fit:
            :return:
            """
            return_result_list = False

            num_pts = 20
            # a_list = np.linspace(0,1, num_pts)
            a_list = np.linspace(120, 1880, num_pts)
            epi_list = np.linspace(240, 2000, num_pts)
            cos_list = np.linspace(0.5, 1, int(num_pts / 2))
            cdist_list = np.linspace(0, 60, 10)

            if full_fit:
                print("Total points:", num_pts * num_pts * int(num_pts / 2) * 10 * 10, file=out_file)
                total_iter = num_pts * num_pts * int(num_pts / 2) * 10 * 10
            else:
                print("Total points:", num_pts * num_pts * int(num_pts / 2), file=out_file)
                total_iter = num_pts * num_pts * int(num_pts / 2)

            if return_result_list:
                res_list = []
                param_list = []

            j = 0

            if full_fit:
                min_val = 1.e6
                min_val_params = tuple()
                # t = tq.tqdm(total=total_iter)
                for epi in epi_list:
                    for cos in cos_list:
                        for a in a_list:
                            for c1 in cdist_list:
                                for c2 in cdist_list:
                                    params = (epi, cos, a)
                                    res = self.log_fit_pi0_probability_v2(theta=params, shower_pts=shower_pts,
                                                                          conv_dist=(c1, c2))
                                    if abs(res) == np.inf:
                                        res = 1.e4
                                    else:
                                        res = -res
                                    if res < min_val:
                                        # print("Eπ0,cosθ,a,res: ", epi, ", ", cos, ", ", a, ",", -res)
                                        min_val = res
                                        min_val_params = (epi, cos, a, c1, c2)
                                    if return_result_list:
                                        res_list.append(res)
                                        param_list.append((epi, cos, a, c1, c2))
                                    # t.update(1)
                                    j += 1
                                    if not j % 50000:
                                        print("Iteration:", j, file=out_file)
                # t.close()
            else:
                min_val = 1.e6
                min_val_params = tuple()
                # t = tq.tqdm(total=total_iter)
                for epi in epi_list:
                    for cos in cos_list:
                        for a in a_list:
                            params = (epi, cos, a)
                            res = self.log_fit_pi0_probability_v2(theta=params, shower_pts=shower_pts,
                                                                  conv_dist=conv_dist)
                            if abs(res) == np.inf:
                                res = 1.e4
                            else:
                                res = -res
                            if res < min_val:
                                # print("Eπ0,cosθ,a,res: ", epi, ", ", cos, ", ", a, ",", -res)
                                min_val = res
                                min_val_params = (epi, cos, a)
                            if return_result_list:
                                res_list.append(res)
                                param_list.append((epi, cos, a))
                            # t.update(1)
                            j += 1
                            if not j % 50000:
                                print("Iteration:", j, file=out_file)
                # t.close()

            if full_fit:
                print("Brute Force Min: (Eπ0,cosθ,a,c1,c2),res =", min_val_params, min_val, file=out_file)
            else:
                print("Brute Force Min: (Eπ0,cosθ,a),res =", min_val_params, min_val, file=out_file)

            if return_result_list:
                return min_val_params, res_list, param_list
            else:
                return min_val, min_val_params
