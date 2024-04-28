import numpy as np
import numba as nb
import cupy as cp
import pickle

from pi0fit.shower_model import AnalyticShowerModel
from pi0fit.shower_model import ShowerModelBase, AnalyticShowerModel, BinnedShowerModel
import pi0fit.fitter_utilities as futil


class Pi0Model:
    def __init__(self, config):
        self.config = config["pi0_model"]

        self.debug = self.config["debug"]

        '''
            Radiation length and Moliere Radius taken from:
            https://pdg.lbl.gov/2020/AtomicNuclearProperties/HTML/liquid_argon.html
        '''
        self.moliere_divisor = 9.043
        self.rad_len_divisor = 14.

        #self.shower_model = AnalyticShowerModel(config)

        model = self.config["shower_model"]
        self.shower_model = futil.get_class(selected_class=model, base_class=ShowerModelBase, config=config)
        self.ana_model = AnalyticShowerModel(config=config)
        self.binned_model = BinnedShowerModel(config=config)

    def pi0_model(self, shower_pts, epi0, cos_pi0, eg1, c1, c2):
        """

        :param shower_pts:
        :param epi0:
        :param cos_pi0:
        :param eg1:
        :param c1:
        :param c2:
        :return:
        """
        eg2 = epi0 - eg1
        a = eg1 / epi0

        alpha, theta1, theta2 = futil.pi0_angles(epi0=epi0, cos_pi0=cos_pi0, a=a)
        print("alpha, theta1, theta2:", round(np.degrees(alpha), 2), ",", round(np.degrees(theta1), 2),
              ",", round(np.degrees(theta2), 2))
        result = self.pi0_probability(shower_pts=shower_pts, eg1=eg1, eg2=eg2, phi=0., theta1=theta1,
                                                theta2=theta2, conv_dist1=c1, conv_dist2=c2)
        result[result <= 0.] = 1.e-200

        if self.debug:
            log_f = np.log(np.nan_to_num(result, nan=1.e-200) + 1.e-200)
            print("epi,cosθ,a,e1,e2,t1,t2,res ", round(epi0, 4), " | ", round(cos_pi0, 4), " | ", round(a, 4), " | ",
                  round(eg1, 2), " | ", round(eg2, 2), " | ", round(np.degrees(theta1), 2),
                  " | ", round(np.degrees(theta2), 2), " | ", (np.sum(log_f)))

        return result

    def tr_transform(self, pi0_pts, theta, conv_dist, phi=0):
        tshift = 0.01

        dir_hypothesis = np.array([[1., theta, phi]])  # test shower 1 direction
        angle = np.arccos(futil.spherical_dot(dir_hypothesis, pi0_pts[:, 0:3]))

        t = pi0_pts[:, 0] * np.abs(np.cos(angle)) - conv_dist  # subtract before dividing by rad_len
        r = pi0_pts[:, 0] * np.abs(np.sin(angle))
        r /= self.moliere_divisor
        t /= self.rad_len_divisor
        tmask = t >= tshift

        return t, r, tmask, dir_hypothesis

    def pi0_probability(self, shower_pts, eg1, eg2, phi, theta1, theta2, conv_dist1, conv_dist2):
        """
        Assume we've rotated the points into the plane of the decay gammas
        """

        t1, r1, tmask1, dir1 = self.tr_transform(pi0_pts=shower_pts, theta=theta1, conv_dist=conv_dist1)
        t2, r2, tmask2, dir2 = self.tr_transform(pi0_pts=shower_pts, theta=theta2, conv_dist=conv_dist2)

        alpha = np.arccos(futil.spherical_dot(dir1, dir2))
        open_angle_prob = self.dn_dalpha_distribution(alpha=alpha, epi0=eg1 + eg2)

        shower_lh1 = np.ones_like(r1)
        shower_lh2 = np.ones_like(r2)

        shower_lh1[tmask1] = self.shower_model.evaluate_shower_pdf(t=t1[tmask1], r=r1[tmask1], charge=shower_pts[:,3][tmask1],
                                                                   energy=eg1)
        shower_lh2[tmask2] = self.shower_model.evaluate_shower_pdf(t=t2[tmask2], r=r2[tmask2], charge=shower_pts[:,3][tmask2],
                                                                   energy=eg2)

        res = (np.nan_to_num(shower_lh1, nan=1.e-100, posinf=1.e10, neginf=1.e-100) +
               np.nan_to_num(shower_lh2, nan=1.e-100, posinf=1.e10, neginf=1.e-100)) * open_angle_prob  # convert nan -> 0.0

        ana_res = (np.nan_to_num(shower_lh1, nan=1.e-100, posinf=1.e10, neginf=1.e-100) *
                   np.nan_to_num(shower_lh2, nan=1.e-100, posinf=1.e10, neginf=1.e-100))
        ana_nll = -np.sum(np.log(np.nan_to_num(ana_res, nan=1.e-200) + 1.e-200)) / len(shower_pts)

        if self.debug:  # np.any(np.isnan(res)) or np.any(res < 0.):
            print("NaN", np.count_nonzero(np.isnan(res)), " <0", np.count_nonzero(res < 0))
            return np.ones_like(res) * -np.inf

        return ana_nll - np.log(open_angle_prob)

    def pi0_probability_binned(self, shower_pts, eg1, eg2, theta1, theta2, conv_dist1, conv_dist2, f1=1000, f2=1000):
        """
        Assume we've rotated the points into the plane of the decay gammas
        """

        fake_eg1, fake_eg2 = f1, f2

        t1, r1, tmask1, dir1 = self.tr_transform(pi0_pts=shower_pts, theta=theta1, conv_dist=conv_dist1)
        t2, r2, tmask2, dir2 = self.tr_transform(pi0_pts=shower_pts, theta=theta2, conv_dist=conv_dist2)

        alpha = np.arccos(futil.spherical_dot(dir1, dir2))
        open_angle_prob = self.dn_dalpha_distribution(alpha=alpha, epi0=eg1 + eg2)

        ana_s1 = np.ones_like(r1)
        ana_s2 = np.ones_like(r2)

        excess_energy1 = np.sum(shower_pts[:, 3][~tmask1 & (shower_pts[:, 3] > -1.)])
        excess_energy2 = np.sum(shower_pts[:, 3][~tmask2 & (shower_pts[:, 3] > -1.)])

        """
        Use the Analytic shower model to estimate the shower direction and conversion distance
        """

        ana_s1[tmask1] = self.ana_model.evaluate_shower_pdf(t=t1[tmask1], r=r1[tmask1], charge=shower_pts[:,3][tmask1],
                                                                   energy=fake_eg1)
        ana_s2[tmask2] = self.ana_model.evaluate_shower_pdf(t=t2[tmask2], r=r2[tmask2], charge=shower_pts[:,3][tmask2],
                                                                   energy=fake_eg2)

        """
        Now use the Binned shower model to estimate the shower energy
        """

        binned_s1, q3p_s1 = self.binned_model.evaluate_shower_pdf(t=t1[tmask1], r=r1[tmask1],
                                                          charge=shower_pts[:, 3][tmask1], energy=eg1,
                                                          energy2=eg2, excess_energy=excess_energy1, cutoff=20.)
        binned_s2, q3p_s2 = self.binned_model.evaluate_shower_pdf(t=t2[tmask2], r=r2[tmask2],
                                                          charge=shower_pts[:, 3][tmask2], energy=eg2,
                                                          energy2=eg1, excess_energy=excess_energy2, cutoff=20.)

        ana_res = (np.nan_to_num(ana_s1, nan=1.e-100, posinf=1.e10, neginf=1.e-100) *
                   np.nan_to_num(ana_s2, nan=1.e-100, posinf=1.e10, neginf=1.e-100)) # convert nan -> 0.0

        binned_nll = -np.sum(np.log(binned_s1 * binned_s2 * q3p_s1 * q3p_s2 * open_angle_prob + 1.e-200))
        ana_nll = -np.sum(np.log(np.nan_to_num(ana_res, nan=1.e-200) + 1.e-200)) / len(shower_pts)

        """
        Use exponential model to fit each gammas conversion distance
        """

        flatline = 1.e-6
        tshift1 = (t1[r1 < 0.45] * self.rad_len_divisor) - conv_dist1
        tshift2 = (t2[r2 < 0.45] * self.rad_len_divisor) - conv_dist2

        res1 = np.where(tshift1 > -5., self.binned_model.shower_func(x=tshift1, A=1, k=0.1, mu=(0 + 4), sigma=1,
                                                                     reverse=True, normalize=True), flatline)
        res2 = np.where(tshift2 > -5., self.binned_model.shower_func(x=tshift2, A=1, k=0.1, mu=(0 + 4), sigma=1,
                                                                     reverse=True, normalize=True), flatline)

        s1_nll = -np.sum(np.log(res1)) / len(shower_pts)
        s2_nll = -np.sum(np.log(res2)) / len(shower_pts)

        if self.debug:
            print("NLL Ana/Bin", ana_nll, "/", binned_nll)

        return np.nan_to_num(binned_nll + ana_nll + s1_nll + s2_nll, nan=1.e20, neginf=1.e20)

    @staticmethod
    def dn_dalpha_distribution(alpha, epi0):
        offset = 0.1
        min_angle = 2. * np.arcsin(135. / epi0)
        momentum = np.sqrt(epi0 ** 2 - 135. * 135.)
        beta = (momentum / 135.) * np.sqrt(1 / (1 + (momentum / 135.) ** 2))
        gamma = 1 / np.sqrt(1 - beta ** 2)

        diff_angle = 2 * (1 / (4. * gamma * beta)) * (np.cos(alpha / 2.) / np.sin(alpha / 2.) ** 2) * (
                    1 / np.sqrt(gamma ** 2 * np.sin(alpha / 2.) ** 2 - 1))

        if alpha < (min_angle + np.radians(offset)):
            min_alpha = min_angle + np.radians(offset)
            trans_point = 2 * (1 / (4. * gamma * beta)) * (np.cos(min_alpha / 2.) / np.sin(min_alpha / 2.) ** 2) * (
                        1 / np.sqrt(gamma ** 2 * np.sin(min_alpha / 2.) ** 2 - 1))
            diff_angle = trans_point * np.exp(50.*(alpha - min_alpha))

        return diff_angle

    @staticmethod
    def conversion_pdf(energy, dist):
        conv_dist = -0.0033 * energy + 24.2  # from D. Rivera thesis
        # profile_res = longitudinal_profile(r=0.01, t=x/rad_len_divisor(), E0=E)
        return np.exp(dist / conv_dist) / conv_dist


class BinnedPi0Model:
    def __init__(self, config):
        self.config = config["binned_pi0_model"]

        self.debug = self.config["debug"]
        self.variance_divisor = self.config["variance_divisor"]  # use 6
        self.charge_hist_file = self.config["charge_hist_file"]
        self.direction_hist_file = self.config["direction_hist_file"]
        self.charge_upper_range = self.config["charge_upper_range"]
        self.sigma_e_factor = self.config["sigma_e_factor"]
        self.epi0_charge_model_divisor = self.config["epi0_charge_model_divisor"]

        self.charge_dict, self.direction_dict, self.random_hist = self.load_histograms()
        self.fit_elist = np.asarray(list(self.charge_dict.keys())[:-3])
        self.charge_bins = self.get_charge_hist(energy=100).shape[0]
        self.direction_bins = self.get_direction_hist().shape[0]

    def energy_key(self, energy):
        return self.fit_elist[np.argmin(abs(self.fit_elist - energy))]

    def get_charge_hist(self, energy):
        return self.charge_dict[self.energy_key(energy)][0]

    def get_variance_hist(self, energy):
        return self.charge_dict[self.energy_key(energy)][1]

    def get_direction_hist(self):
        return self.direction_dict["shower_dir"]

    @staticmethod
    def beam_cut(spherical_pts):
        beam_theta, beam_phi = 162., 50.

        the_one = np.ones(len(spherical_pts[:, 1]))
        pts = np.vstack((the_one, np.radians(spherical_pts[:, 1]), np.radians(spherical_pts[:, 2]))).T
        beam_center = np.vstack((the_one, the_one * np.radians(beam_theta), the_one * np.radians(beam_phi))).T

        beam_point_cos = futil.spherical_dot(beam_center, pts)

        return ~(beam_point_cos > 0.8)

    def construct_event_hists(self, pi0_pts, return_precut=False):

        print("Event Charge Fill")
        print("Pre-Qhist", np.sum(pi0_pts[:, 3]))
        hist, bins = np.histogramdd(pi0_pts[:, :3], bins=self.charge_bins, range=((0, 450), (0, 180), (-180, 180)))
        charge_hist_tmp = self.set_bin_charge_3d(hist=hist, X=pi0_pts[:, 0], Y=pi0_pts[:, 1],
                                             Z=pi0_pts[:, 2], Q=pi0_pts[:, 3], bins=bins)

        charge_hist = charge_hist_tmp if return_precut else self.distance_cut_2d(hist=charge_hist_tmp, X=pi0_pts[:, 0],
                                                                         Y=pi0_pts[:, 1], Z=pi0_pts[:, 2],
                                                                         Q=pi0_pts[:, 3], bins=bins,
                                                                         dist_lower_cut=15, dist_upper_cut=80,
                                                                         charge_cut=80)

        print("Qhist", np.sum(charge_hist), "Calo Energy", self.calo_to_energy(charge=np.sum(charge_hist)))

        hist, bins = np.histogramdd(pi0_pts[:, :3], bins=self.direction_bins, range=((0, 450), (0, 180), (-180, 180)))
        dir_hist_tmp = self.set_bin_charge_3d(hist=hist, X=pi0_pts[:, 0], Y=pi0_pts[:, 1],
                                          Z=pi0_pts[:, 2], Q=pi0_pts[:, 3], bins=bins)

        dir_hist = dir_hist_tmp if return_precut else self.distance_cut_2d(hist=dir_hist_tmp, X=pi0_pts[:, 0],
                                                                           Y=pi0_pts[:, 1], Z=pi0_pts[:, 2],
                                                                           Q=pi0_pts[:, 3], bins=bins, dist_lower_cut=15,
                                                                           dist_upper_cut=80, charge_cut=25)

        print("Charge Bins", charge_hist.shape, " Dir Bins", dir_hist.shape)

        return charge_hist, dir_hist

    def pi0_model_nll(self, hevt_charge, hdir_charge, energy_from_calo, dir_norm, epi0, e1, e2, e3, a1, a2, a3, p1, p2, p3, c1, c2, c3, two_shower):

        a1_rad, a2_rad, a3_rad = np.radians(a1), np.radians(a2), np.radians(a3)
        p1_rad, p2_rad, p3_rad = np.radians(p1), np.radians(p2), np.radians(p3)

        open_angle_cos_12 = futil.spherical_dot(np.array([[1, a1_rad, p1_rad]]), np.array([[1, a2_rad, p2_rad]]))
        oa_divisor = 5.  # 75#40
        oa_prob_12 = (np.nan_to_num(self.dn_dalpha_distribution_mod(alpha=np.arccos(open_angle_cos_12),
                                                                    epi0=epi0), nan=1.e-300, posinf=1.e-300))
        oa_nll = -(np.log(oa_prob_12)) #/ oa_divisor)
        oa_nll = np.clip(oa_nll, a_max=np.inf, a_min=-5./oa_divisor)
        # oa_nll = np.where(oa_nll_shifted <= -2, np.exp((-oa_nll_shifted) / 2.)-3, oa_nll_shifted) / oa_divisor

        #charge_bins = self.charge_dict['bins']
        direction_bins = self.direction_dict['bins']


        ## 3D
        # hcomp_charge = self.create_pi0_pdf(hist_lo=self.get_charge_hist(energy=e1),
        #                                    hist_hi=self.get_charge_hist(energy=e2),
        #                                    hist_pro=proton_charge_hist,
        #                                    xyzpos_lo=(int(c1), int(a1), int(p1)),
        #                                    xyzpos_hi=(int(c2), int(a2), int(p2)),
        #                                    xyzpos_pro=proton_pos,
        #                                    bins=charge_bins, xyz_origin=(0, 90, 90), fill_value=0)
        #
        # hcomp_var = self.create_pi0_variance_hist(hist_lo=self.get_variance_hist(energy=e1),
        #                                           hist_hi=self.get_variance_hist(energy=e2),
        #                                           hist_pro=proton_var_hist,
        #                                           xyzpos_lo=(int(c1), int(a1), int(p1)),
        #                                           xyzpos_hi=(int(c2), int(a2), int(p2)),
        #                                           xyzpos_pro=proton_pos,
        #                                           bins=charge_bins, xyz_origin=(0, 90, 90), fill_value=1)

        hcomp_dir_unnorm = self.create_pi0_pdf(hist_lo=self.get_direction_hist(),
                                               hist_hi=self.get_direction_hist(),
                                               hist_pro=None,
                                               xyzpos_lo=(int(c1), int(a1), int(p1)),
                                               xyzpos_hi=(int(c2), int(a2), int(p2)),
                                               xyzpos_pro=None,
                                               bins=direction_bins, xyz_origin=(0, 90, 90), fill_value=0)

        hcomp_dir_unnorm[hcomp_dir_unnorm == 0] =  1.6981788794088674e-11 #np.min(hcomp_dir_unnorm[hcomp_dir_unnorm > 0])
        hcomp_dir = hcomp_dir_unnorm / self.normalize_3d_hist(hist=hcomp_dir_unnorm, bins=direction_bins)
        print("Min:", np.min(hcomp_dir_unnorm[hcomp_dir_unnorm > 0]))

        ## Direction Likelihood
        # dir_norm = np.sum(hdir_charge) if two_shower else np.prod(hcomp_dir.shape)
        
        #charge_dir_nll = -np.sum(hdir_charge * np.log(hcomp_dir + 1.e-200)) / dir_norm

        ## GPU
        hcomp_gpu = cp.asarray(hcomp_dir)
        #hdir_gpu = cp.asarray(hdir_charge)
        nll = -cp.sum(hdir_charge * cp.log(hcomp_gpu + 1e-200))
        charge_dir_nll = cp.asnumpy(nll)
        charge_dir_nll /= dir_norm

        ## Charge Likelihood
        # charge_nll = self.test_charge_model_nll(hevt_charge=hevt_charge, hcomp=hcomp_charge, hcomp_var=hcomp_var)
        charge_nll = self.total_charge_model(epi0=epi0, energy_from_calo=energy_from_calo)

        ## Conversion Likelihood
        #conv_nll = (c1 / 100.) + (c2 / 100.)

        if self.debug:
            print("NLL Q/Dir/OA/C", charge_nll, "/", charge_dir_nll, "/", oa_nll)

        return charge_nll + charge_dir_nll + oa_nll  # Total NLL

    def charge_integral_vec(self, hcomp, hcomp_var):

        exponent = -0.5 * ((self.random_hist - hcomp) / hcomp_var) * ((self.random_hist - hcomp) / hcomp_var)
        exponential = np.exp(exponent)

        bins = self.charge_dict['bins']
        binwidth2 = bins[2][1] - bins[2][0]
        binwidth1 = bins[1][1] - bins[1][0]
        binwidth0 = bins[0][1] - bins[0][0]

        norm = np.sum(np.sum(np.sum(exponential * binwidth2, axis=3) * binwidth1, axis=2) * binwidth0, axis=1)

        return self.charge_upper_range * (np.sum(norm) / self.random_hist.shape[0])

    @staticmethod
    def charge_model(hevt_charge, hcomp, hcomp_var):
        mean_dist = (hevt_charge - hcomp) / hcomp_var
        return np.exp(-0.5 * mean_dist * mean_dist)

    @staticmethod
    def test_charge_model_nll(hevt_charge, hcomp, hcomp_var, return_prob=False):

        # Average over R axis
        # NO! must sum over unwanted axis to conserve total charge
        hevt_charge_angle = np.sum(hevt_charge, axis=0)
        hcomp_angle = np.sum(hcomp, axis=0)
        hcomp_var_angle = np.sum(hcomp_var, axis=0)

        hcomp_angle[hcomp_angle < 2] = 2. #np.min(hcomp_angle[hcomp_angle > 0])
        mask = hevt_charge_angle > 0.

        mean_dist = (hevt_charge_angle - hcomp_angle) / hcomp_var_angle
        norm = np.log(np.sqrt(2. * np.pi) * hcomp_var_angle + 1.e-200)

        # 3D R,θ,ϕ
        # mean_dist = (hevt_charge - hcomp) / hcomp_var
        # norm = np.log(np.sqrt(2. * np.pi) * hcomp_var + 1.e-200)

        if return_prob:
            return (np.exp(-0.5 * mean_dist * mean_dist) / (np.sqrt(2. * np.pi) * hcomp_var_angle))[mask], mask

        charge_norm = 150. * len(mean_dist) #hcomp[mask].shape[0]
        return (np.sum((0.5 * mean_dist * mean_dist)[mask]) + np.sum(norm[mask])) / charge_norm

    def total_charge_model(self, epi0, energy_from_calo):
        sigma_e = self.sigma_e_factor * np.sqrt(energy_from_calo)
        pdf_norm = np.log(np.sqrt(2. * np.pi) * sigma_e)
        charge_nll = pdf_norm + (((epi0 - energy_from_calo) * (epi0 - energy_from_calo)) / (2. * sigma_e * sigma_e))
        return charge_nll / self.epi0_charge_model_divisor

    @staticmethod
    def calo_to_energy(charge):
        energy = (charge - 72.17) / 24.90
        return energy if energy > 135. else 135.

    @staticmethod
    def normalize_3d_hist(hist, bins):
        binwidth2 = bins[2][1] - bins[2][0]
        binwidth1 = bins[1][1] - bins[1][0]
        binwidth0 = bins[0][1] - bins[0][0]

        norm = np.sum(np.sum(np.sum(hist * binwidth2, axis=2) * binwidth1, axis=1) * binwidth0, axis=0)
        return norm

    @staticmethod
    @nb.njit(error_model="numpy")
    def set_bin_charge_3d(hist, X, Y, Z, Q, bins):
        for x in range(len(bins[0]) - 1):
            for y in range(len(bins[1]) - 1):
                for z in range(len(bins[2]) - 1):
                    xmask = (X >= bins[0][x]) & (X < bins[0][x + 1])
                    ymask = (Y >= bins[1][y]) & (Y < bins[1][y + 1])
                    zmask = (Z >= bins[2][z]) & (Z < bins[2][z + 1])
                    qtot = np.sum(Q[xmask & ymask & zmask])
                    hist[x, y, z] = qtot
        return hist

    @staticmethod
    @nb.njit(error_model="numpy", fastmath=True)
    def distance_cut_2d(hist, X, Y, Z, Q, bins, dist_lower_cut, dist_upper_cut, charge_cut):  # θ,ϕ
        for y in range(len(bins[1]) - 1):
            for z in range(len(bins[2]) - 1):
                ymask = (Y >= bins[1][y]) & (Y < bins[1][y + 1])
                zmask = (Z >= bins[2][z]) & (Z < bins[2][z + 1])
                xmask = (X > dist_lower_cut) & (X < dist_upper_cut)
                qup = np.sum(Q[xmask & ymask & zmask]) if len(Q[xmask & ymask & zmask]) > 0 else 0.
                if qup < charge_cut:
                    hist[:, y, z] = 0.
        return hist

    @staticmethod
    def shift_index(array, shift, axis, upper):

        idx_side = slice(shift, array.shape[axis]) if upper else slice(0, shift)

        if axis == 0:
            return (idx_side, ...)
        elif axis == 1:
            return (slice(0, array.shape[axis]), idx_side, ...)
        elif axis == 2:
            return (..., idx_side)
        else:
            raise ValueError

    def shift_array_3d(self, arr, num, axis, fill_value=0):
        result = np.empty_like(arr)
        result_shift_up = self.shift_index(array=result, shift=num, axis=axis, upper=True)
        result_shift_down = self.shift_index(array=result, shift=num, axis=axis, upper=False)
        array_shift_up = self.shift_index(array=arr, shift=-num, axis=axis, upper=True)
        array_shift_down = self.shift_index(array=arr, shift=-num, axis=axis, upper=False)
        if num > 0:
            if axis == 2: # wrap the phi axis
                result[result_shift_down] = arr[array_shift_up]
                result[result_shift_up] = arr[array_shift_down]
            else:
                result[result_shift_down] = fill_value
                result[result_shift_up] = arr[array_shift_down]
        elif num < 0:
            if axis == 2: # wrap the phi axis
                result[result_shift_up] = arr[array_shift_down]
                result[result_shift_down] = arr[array_shift_up]
            else:
                result[result_shift_up] = fill_value
                result[result_shift_down] = arr[array_shift_up]
        else:
            result = arr

        return result

    @staticmethod
    def shift_idx(bins, shifted_pos, shower_origin):
        shift_sign = np.sign(shifted_pos - shower_origin)
        origin_idx = np.argmin(abs(shower_origin - bins))
        shifted_idx = np.argmin(abs(shifted_pos - bins))
        num = abs(origin_idx - shifted_idx)

        return shift_sign * num

    def position_shower_pdf(self, hist, bins, xyz_pos, xyz_origin, fill_value=0):

        hshift = hist.copy()

        for i in range(len(bins)):
            if i == 0: continue # skip the R axis
            num = self.shift_idx(bins=bins[i], shifted_pos=xyz_pos[i], shower_origin=xyz_origin[i])
            hshift = self.shift_array_3d(arr=hshift, num=num, axis=i, fill_value=fill_value)

        return hshift

    def create_pi0_pdf(self, hist_lo, hist_hi, hist_pro, xyzpos_lo, xyzpos_hi, xyzpos_pro, bins,
                       xyz_origin=(0, 90, 90), fill_value=0):

        hist_pi0 = self.position_shower_pdf(hist=hist_lo, bins=bins, xyz_pos=xyzpos_lo, xyz_origin=xyz_origin,
                                            fill_value=fill_value)
        hist_pi0 += self.position_shower_pdf(hist=hist_hi, bins=bins, xyz_pos=xyzpos_hi, xyz_origin=xyz_origin,
                                             fill_value=fill_value)
        if hist_pro is not None:
            hist_pi0 += self.position_shower_pdf(hist=hist_pro, bins=bins, xyz_pos=xyzpos_pro, xyz_origin=xyz_origin,
                                                 fill_value=fill_value)
        return hist_pi0

    def create_pi0_variance_hist(self, hist_lo, hist_hi, hist_pro, xyzpos_lo, xyzpos_hi, xyzpos_pro, bins,
                                 xyz_origin=(0, 90, 90), fill_value=1):
        var_hist_pi0_1 = self.position_shower_pdf(hist=hist_lo, bins=bins, xyz_pos=xyzpos_lo, xyz_origin=xyz_origin,
                                                  fill_value=fill_value) / self.variance_divisor
        var_hist_pi0_2 = self.position_shower_pdf(hist=hist_hi, bins=bins, xyz_pos=xyzpos_hi, xyz_origin=xyz_origin,
                                                  fill_value=fill_value) / self.variance_divisor
        if hist_pro is not None:
            var_hist_pi0_3 = self.position_shower_pdf(hist=hist_pro, bins=bins, xyz_pos=xyzpos_pro, xyz_origin=xyz_origin,
                                                 fill_value=fill_value) / self.variance_divisor
            return np.sqrt(var_hist_pi0_1 * var_hist_pi0_1 + var_hist_pi0_2 * var_hist_pi0_2 + var_hist_pi0_3 * var_hist_pi0_3)

        return np.sqrt(var_hist_pi0_1 * var_hist_pi0_1 + var_hist_pi0_2 * var_hist_pi0_2)

    def load_histograms(self):
        with open(self.charge_hist_file, 'rb') as ifile:
            charge_dict = pickle.load(ifile)

        print("Loaded QHist file:", self.charge_hist_file)
        charge_hist_shape = charge_dict[100][0].shape
        print("QHist Bins", charge_hist_shape)

        with open(self.direction_hist_file, 'rb') as ifile:
            direction_dict = pickle.load(ifile)

        print("Loaded DirHist file:", self.direction_hist_file)
        print("DirHist Bins", direction_dict["shower_dir"].shape)

        rshape = tuple((self.config["n_random_hist"],)) + charge_hist_shape
        random_hist = np.random.uniform(0, self.charge_upper_range, size=rshape)
        print("Random Hist", random_hist.shape)

        return charge_dict, direction_dict, random_hist

    @staticmethod
    def dn_dalpha_distribution(alpha, epi0):
        momentum = np.sqrt(epi0 ** 2 - 135. * 135.)
        beta = (momentum / 135.) * np.sqrt(1 / (1 + (momentum / 135.) ** 2))
        gamma = 1 / np.sqrt(1 - beta ** 2)

        diff_angle = 2 * (1 / (4. * gamma * beta)) * (np.cos(alpha / 2.) / np.sin(alpha / 2.) ** 2) * (
                1 / np.sqrt(gamma ** 2 * np.sin(alpha / 2.) ** 2 - 1))

        return diff_angle

    @staticmethod
    def dn_dalpha_distribution_mod(alpha, epi0):
        offset = 0.1
        min_angle = 2. * np.arcsin(135. / epi0)
        momentum = np.sqrt(epi0 ** 2 - 135. * 135.)
        beta = (momentum / 135.) * np.sqrt(1 / (1 + (momentum / 135.) ** 2))
        gamma = 1 / np.sqrt(1 - beta ** 2)

        diff_angle = 2 * (1 / (4. * gamma * beta)) * (np.cos(alpha / 2.) / np.sin(alpha / 2.) ** 2) * (
                1 / np.sqrt(gamma ** 2 * np.sin(alpha / 2.) ** 2 - 1))

        if alpha < (min_angle + np.radians(offset)):
            min_alpha = min_angle + np.radians(offset)
            trans_point = 2 * (1 / (4. * gamma * beta)) * (np.cos(min_alpha / 2.) / np.sin(min_alpha / 2.) ** 2) * (
                    1 / np.sqrt(gamma ** 2 * np.sin(min_alpha / 2.) ** 2 - 1))
            diff_angle = trans_point * np.exp(50. * (alpha - min_alpha))

        return diff_angle

    # @staticmethod
    # def dn_dalpha_distribution(alpha, epi0):
    #     offset = 0.1
    #     min_angle = 2. * np.arcsin(135. / epi0)
    #     momentum = np.sqrt(epi0 ** 2 - 135. * 135.)
    #     beta = (momentum / 135.) * np.sqrt(1 / (1 + (momentum / 135.) ** 2))
    #     gamma = 1 / np.sqrt(1 - beta ** 2)
    #
    #     diff_angle = 2 * (1 / (4. * gamma * beta)) * (np.cos(alpha / 2.) / np.sin(alpha / 2.) ** 2) * (
    #                 1 / np.sqrt(gamma ** 2 * np.sin(alpha / 2.) ** 2 - 1))
    #
    #     if alpha < (min_angle + np.radians(offset)):
    #         min_alpha = min_angle + np.radians(offset)
    #         trans_point = 2 * (1 / (4. * gamma * beta)) * (np.cos(min_alpha / 2.) / np.sin(min_alpha / 2.) ** 2) * (
    #                     1 / np.sqrt(gamma ** 2 * np.sin(min_alpha / 2.) ** 2 - 1))
    #         diff_angle = trans_point * np.exp(50.*(alpha - min_alpha))
    #
    #     return diff_angle

