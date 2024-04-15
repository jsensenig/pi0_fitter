from abc import abstractmethod
import numpy as np
from scipy.optimize import NonlinearConstraint, dual_annealing, differential_evolution

from pi0fit.pi0_model import Pi0Model, BinnedPi0Model
from pi0fit.fitter_utilities import FitResults, spherical_dot


class Pi0MinimizerBase(FitResults):
    """
    Base class defining the structure of the concrete minimizer classes.
    """
    def __init__(self, config):

        self.config = config["pi0_minimizers"]

        self.debug = self.config["debug"]
        self.pi0_model = BinnedPi0Model(config)
        self.use_pi0_var = self.config["fit_pi0_var"]

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
    def show_results(self, minimizer, pts, truth_values):
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
        self.restart_temp_ratio = self.config["restart_temp_ratio"]
        self.use_scan_start = self.config["use_scan_start"]
        self.minimizer_workers = self.config["minimizer_workers"]

        self.total_event_charge = 0.

    def minimize(self, pi0_points, truth_values=None):

        epi0_bound = tuple(self.config["epi0_limits"])  #(140, 2400)
        cos_pi0_bound = tuple(self.config["cos_pi0_limits"]) #(0., 1)
        eg1_bound = tuple(self.config["eg1_limits"])  #(120, 2200)
        eg2_bound = tuple(self.config["eg2_limits"])  # (120, 2200)
        theta1_bound = tuple(self.config["theta1_limits"])
        theta2_bound = tuple(self.config["theta2_limits"])
        phi1_bound = tuple(self.config["phi1_limits"])
        phi2_bound = tuple(self.config["phi2_limits"])
        c1_bound = tuple(self.config["c1_limits"])   #(0, 80)
        c2_bound = tuple(self.config["c2_limits"])   #(0, 80)

        bounds = [eg1_bound, eg2_bound, eg2_bound, theta1_bound, theta2_bound, theta2_bound, phi1_bound, phi2_bound, phi2_bound]
        start_pt = np.array([500, 800, 200, 30, 60, 110, 50, -20, 100])

        print("Bounds:", bounds)

        if self.out_file is None:
            print("Start Point:", start_pt)
        else:
            print("Start Point:", start_pt, file=self.out_file)

        charge_hist, dir_hist = self.pi0_model.construct_event_hists(pi0_pts=pi0_points)
        self.total_event_charge = np.sum(charge_hist)

        bounds = [epi0_bound, theta1_bound, theta2_bound, phi1_bound, phi2_bound]
        start_pt = np.asarray([800, 30, 60, 50, 40])

        two_shower = True
        nlc_energy_up = NonlinearConstraint(self.total_energy_constraint_up, 0.0, np.inf)
        nlc_energy_down = NonlinearConstraint(self.total_energy_constraint_down, 0.0, np.inf)
        nlc_oa_min = NonlinearConstraint(self.open_angle_constraint, 0.01, np.inf)
        nlc_inv_mass = NonlinearConstraint(self.invariant_mass_constraint, 130., 140.)

        energy_from_calo = self.pi0_model.calo_to_energy(charge=np.sum(charge_hist))
        dir_norm = np.sum(dir_hist)

        min_fval_list = []
        min_list = []
        print("Starting 2Shower Minimization!")
        #for n in range(3): # use tol=1e-5 and mutation=(0.1,0.35)
        for n, mup in enumerate([0.55, 0.55, 0.55]):
            min_res2 = differential_evolution(self.model_interface,
                                              args=(charge_hist, dir_hist, energy_from_calo, dir_norm, two_shower),
                                              bounds=bounds, popsize=60, tol=1.e-5, mutation=(0.01, mup), x0=start_pt,
                                              constraints=[nlc_oa_min], maxiter=25000, workers=self.minimizer_workers)
                                              #constraints = [nlc_oa_min, nlc_inv_mass, nlc_energy_down, nlc_energy_up],
            min_fval_list.append(min_res2.fun)
            min_list.append(min_res2)
            print("Iter:", n, "Mutation_upper:", mup, "Fmin1:", min_res2.fun)

        min_fit_res = min_list[np.argmin(min_fval_list)]
        print("Selected Min:", min_fit_res.fun)

        if truth_values is not None:
            self.show_results(minimizer=min_fit_res, pts=pi0_points, truth_values=truth_values)
        else:
            return min_fit_res

    @staticmethod
    def open_angle_constraint(x):

        # e1, e2, a1, a2, p1, p2 = x
        epi0, a1, a2, p1, p2 = x
        a1_rad, a2_rad = np.radians(a1), np.radians(a2)
        p1_rad, p2_rad = np.radians(p1), np.radians(p2)

        open_angle_12 = np.arccos(spherical_dot(np.array([[1, a1_rad, p1_rad]]), np.array([[1, a2_rad, p2_rad]])))
        min_angle_12 = 2. * np.arcsin(135. / epi0)
        angle_diff_12 = open_angle_12 - min_angle_12 if not np.isnan(min_angle_12) else -1.

        adiff = angle_diff_12  # must be >0

        return adiff

    @staticmethod
    def invariant_mass_constraint(x):

        e1, e2, a1, a2, p1, p2 = x
        open_angle_cos = spherical_dot(np.array([[1, np.radians(a1), np.radians(p1)]]),
                                       np.array([[1, np.radians(a2), np.radians(p2)]]))
        inv_mass = np.sqrt(2. * e1 * e2 * (1 - open_angle_cos))
        return inv_mass

    def total_energy_constraint_up(self, x):
        e1, e2, a1, a2, p1, p2 = x
        _, energy_up = self.energy_limits(charge=self.total_event_charge)
        return energy_up - (e1 + e2)

    def total_energy_constraint_down(self, x):
        e1, e2, a1, a2, p1, p2 = x
        energy_down, _ = self.energy_limits(charge=self.total_event_charge)
        return (e1 + e2) - energy_down

    @staticmethod
    def energy_limits(charge):
        low_lim = 0.055 * charge - 1200.
        high_lim = 0.045 * charge + 200.

        return np.max([low_lim, 135.]), np.min([high_lim, 2.e3])

    def select_showers(self, min_res1, charge_hist, dir_hist):

        tmp_charge_hist = np.ones_like(charge_hist) * charge_hist
        tmp_dir_hist = np.ones_like(dir_hist) * dir_hist

        s1, s2, energy = self.choose_shower_pair(fit_res=min_res1, set_pair=None)
        print("Shower1", s1)
        print("Shower2", s2)
        print("Energy", energy)

        reco_a1, reco_p1 = s1
        reco_a2, reco_p2 = s2
        reco_e1, reco_e2 = energy
        c1, c2 = 5, 5

        fit_hcomp_charge_3d = self.pi0_model.create_pi0_pdf(hist_lo=self.pi0_model.get_charge_hist(energy=reco_e1),
                                                            hist_hi=self.pi0_model.get_charge_hist(energy=reco_e2),
                                                            hist_pro=None,
                                                            xyzpos_lo=(int(c1), int(reco_a1), int(reco_p1)),
                                                            xyzpos_hi=(int(c2), int(reco_a2), int(reco_p2)),
                                                            xyzpos_pro=None,
                                                            bins=self.pi0_model.charge_dict['bins'],
                                                            xyz_origin=(0, 90, 90), fill_value=0)
        fit_hcomp_dir_unorm = self.pi0_model.create_pi0_pdf(hist_lo=self.pi0_model.get_direction_hist(),
                                                            hist_hi=self.pi0_model.get_direction_hist(),
                                                            hist_pro=None,
                                                            xyzpos_lo=(int(c1), int(reco_a1), int(reco_p1)),
                                                            xyzpos_hi=(int(c2), int(reco_a2), int(reco_p2)),
                                                            xyzpos_pro=None,
                                                            bins=self.pi0_model.direction_dict['bins'],
                                                            xyz_origin=(0, 90, 90), fill_value=0)

        fit_hcomp_dir_3d = fit_hcomp_dir_unorm / self.pi0_model.normalize_3d_hist(hist=fit_hcomp_dir_unorm,
                                                                                  bins=self.pi0_model.direction_dict['bins'])
        charge_mask = fit_hcomp_charge_3d < 2.
        dir_mask = fit_hcomp_dir_3d < 1.e-8

        tmp_charge_hist[charge_mask] = 0.
        tmp_dir_hist[dir_mask] = 0.

        print("New Hists Q/D", np.sum(tmp_charge_hist), "/", np.sum(tmp_dir_hist))

        return tmp_charge_hist, tmp_dir_hist

    @staticmethod
    def choose_shower_pair(fit_res, set_pair=None):

        reco_a1, reco_p1 = fit_res.x[3], fit_res.x[6]
        reco_a2, reco_p2 = fit_res.x[4], fit_res.x[7]
        reco_a3, reco_p3 = fit_res.x[5], fit_res.x[8]
        e1, e2, e3 = fit_res.x[0], fit_res.x[1], fit_res.x[2]

        angle_check_12 = spherical_dot(np.array([[1, np.radians(reco_a1), np.radians(reco_p1)]]),
                                       np.array([[1, np.radians(reco_a2), np.radians(reco_p2)]]))[0]
        angle_check_13 = spherical_dot(np.array([[1, np.radians(reco_a1), np.radians(reco_p1)]]),
                                       np.array([[1, np.radians(reco_a3), np.radians(reco_p3)]]))[0]
        angle_check_23 = spherical_dot(np.array([[1, np.radians(reco_a2), np.radians(reco_p2)]]),
                                       np.array([[1, np.radians(reco_a3), np.radians(reco_p3)]]))[0]

        mgg_12 = np.sqrt(2. * e1 * e2 * (1. - angle_check_12))
        mgg_13 = np.sqrt(2. * e1 * e3 * (1. - angle_check_13))
        mgg_23 = np.sqrt(2. * e2 * e3 * (1. - angle_check_23))

        print("mgg 12/13/23", mgg_12, "/", mgg_13, "/", mgg_23)

        shower_pair = np.argmin(
            [abs(mgg_12 - 135), abs(mgg_13 - 135), abs(mgg_23 - 135)]) if set_pair is None else set_pair

        if shower_pair == 0:
            print("Selected Showers 1&2 (Mgg =", round(mgg_12, 2), ")")
            return (reco_a1, reco_p1), (reco_a2, reco_p2), (e1, e2)
        elif shower_pair == 1:
            print("Selected Showers 1&3 (Mgg =", round(mgg_13, 2), ")")
            return (reco_a1, reco_p1), (reco_a3, reco_p3), (e1, e3)
        else:
            print("Selected Showers 2&3 (Mgg =", round(mgg_23, 2), ")")
            return (reco_a2, reco_p2), (reco_a3, reco_p3), (e2, e3)

    @staticmethod
    def calculate_shower_energy(epi0, open_angle):
        mgg2 = 135. * 135.
        num = np.sqrt(2. * mgg2 * (np.cos(open_angle) - 1.) + epi0 * epi0 * (1. - np.cos(open_angle)) ** 2) \
              + epi0 * (np.cos(open_angle) - 1.)
        denom = 2. * (np.cos(open_angle) - 1.)
        calculated_e1 = num / denom

        if np.isnan(calculated_e1):
            return epi0 / 2., epi0 / 2.

        return calculated_e1, epi0 - calculated_e1

    def energy_from_shower_direction(self, pts, epi0, open_angle, theta1, theta2, phi1, phi2, print_stuff=False):

        ones = np.ones(len(pts), dtype=bool)

        tmp_pts = np.vstack((ones, np.radians(pts[:, 1]), np.radians(pts[:, 2]))).T
        shower1 = np.vstack((ones, ones * np.radians(theta1), ones * np.radians(phi1))).T
        shower2 = np.vstack((ones, ones * np.radians(theta2), ones * np.radians(phi2))).T

        shower_point_cos1 = spherical_dot(shower1, tmp_pts)
        shower_point_cos2 = spherical_dot(shower2, tmp_pts)

        charge_save_mask1 = shower_point_cos1 > 0.92
        charge_save_mask2 = shower_point_cos2 > 0.92

        if print_stuff: print("Selected Count:", np.count_nonzero(charge_save_mask1), "/",
                              np.count_nonzero(charge_save_mask2))

        qsum1 = np.sum(pts[:, 3][charge_save_mask1])
        qsum2 = np.sum(pts[:, 3][charge_save_mask2])
        qtotal = np.sum(pts[:, 3][charge_save_mask1 | charge_save_mask2])

        calo_epi0 = self.pi0_model.calo_to_energy(charge=qtotal)
        print("Epi0 fit/calo:", epi0, "/", calo_epi0)

        tmp_e1, tmp_e2 = self.calculate_shower_energy(epi0=calo_epi0, open_angle=open_angle)
        #tmp_e1, tmp_e2 = self.calculate_shower_energy(epi0=epi0, open_angle=open_angle)
        calc_energy1, calc_energy2 = (tmp_e1, tmp_e2) if qsum1 > qsum2 else (tmp_e2, tmp_e1)

        if print_stuff:
            print("S1: E", int(calc_energy1), "MeV")
            print("S2: E", int(calc_energy2), "MeV")

        return calo_epi0, calc_energy1, calc_energy2
        #return epi0, calc_energy1, calc_energy2

    def show_results(self, minimizer, pts, truth_values):
        """
        Print reco fit results and compare to truth values.
        :param minimizer:
        :param truth_values:
        :return:
        """
        shift = -1
        true_dir1 = np.array([[1., np.radians(truth_values.theta1), np.radians(truth_values.phi1)]])
        true_dir2 = np.array([[1., np.radians(truth_values.theta2), np.radians(truth_values.phi2)]])
        reco_dir1 = np.array([[1., np.radians(minimizer.x[shift+2]), np.radians(minimizer.x[shift+4])]])
        reco_dir2 = np.array([[1., np.radians(minimizer.x[shift+3]), np.radians(minimizer.x[shift+5])]])

        flipped = spherical_dot(reco_dir2, true_dir1) > spherical_dot(reco_dir1, true_dir1)
        flipped &= spherical_dot(reco_dir1, true_dir2) > spherical_dot(reco_dir2, true_dir2)

        open_angle = np.arccos(spherical_dot(reco_dir1, reco_dir2))[0]

        # epi0 = minimizer.x[0]
        # eg1, eg2 = self.calculate_shower_energy(epi0=epi0, open_angle=open_angle)
        epi0, eg1, eg2 = self.energy_from_shower_direction(pts=pts, epi0=minimizer.x[0], open_angle=open_angle, theta1=minimizer.x[shift+2],
                                                           theta2=minimizer.x[shift+3], phi1=minimizer.x[shift+4],
                                                           phi2=minimizer.x[shift+5])

        if flipped:
            self.set_event_values_shower(eg1=eg1, eg2=eg2, theta1=minimizer.x[shift+3],
                                         theta2=minimizer.x[shift+2], phi1=minimizer.x[shift+5], phi2=minimizer.x[shift+4],
                                         c1=5, c2=5, is_truth=False)
        else:
            self.set_event_values_shower(eg1=eg2, eg2=eg1, theta1=minimizer.x[shift+2],
                                         theta2=minimizer.x[shift+3], phi1=minimizer.x[shift+4], phi2=minimizer.x[shift+5],
                                         c1=5, c2=5, is_truth=False)

        self.print_comparison_table(fit_result=truth_values)
        self.calculate_metrics(fit_result=truth_values)

    def model_interface(self, x, charge_hist, dir_hist, energy_from_calo, dir_norm, two_shower):

        if self.debug:
            print("x", x)

        if two_shower:
            epi0, a1, a2, p1, p2 = x
            # e1, e2, a1, a2, p1, p2 = x
            e1, e2 = 0., 0.
            e3, a3, p3, c3 = 0., 0., 0., 0.
        else:
            e1, e2, e3, a1, a2, a3, p1, p2, p3 = x

        return self.pi0_model.pi0_model_nll(hevt_charge=charge_hist, hdir_charge=dir_hist,
                                            energy_from_calo=energy_from_calo, dir_norm=dir_norm, epi0=epi0,
                                            e1=e1, e2=e2, e3=e3, a1=a1, a2=a2, a3=a3, p1=p1, p2=p2, p3=p3,
                                            c1=5, c2=5, c3=5, two_shower=two_shower)

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
