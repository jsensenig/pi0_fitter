import numpy as np
import math

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

    def pi0_probability(self, shower_pts, eg1, eg2, phi, theta1, theta2, conv_dist1, conv_dist2):
        """
        Assume we've rotated the points into the plane of the decay gammas
        """

        Rs1 = np.array([[1., theta1, phi]])  # test shower 1 direction
        Rs2 = np.array([[1., theta2, phi]])  # test shower 2 direction
        alpha = np.arccos(futil.spherical_dot(Rs1, Rs2))
        #eg2 = etotal  # - eg1

        open_angle_lh = self.dn_dalpha_distribution(alpha=alpha, epi0=(eg1 + eg2))

        if self.debug:
            print("cosα:", np.cos(alpha), "α:", np.degrees(alpha), "LH:", open_angle_lh)
            print("E2:", eg2)

        angle1 = np.arccos(futil.spherical_dot(Rs1, shower_pts[:, 0:3]))
        angle2 = np.arccos(futil.spherical_dot(Rs2, shower_pts[:, 0:3]))

        t1 = shower_pts[:, 0] * np.abs(np.cos(angle1)) - conv_dist1  # subtract before dividing by rad_len
        r1 = shower_pts[:, 0] * np.abs(np.sin(angle1))

        r1 /= self.moliere_divisor
        t1 /= self.rad_len_divisor

        t2 = shower_pts[:, 0] * np.abs(np.cos(angle2)) - conv_dist2  # subtract before dividing by rad_len
        r2 = shower_pts[:, 0] * np.abs(np.sin(angle2))

        r2 /= self.moliere_divisor
        t2 /= self.rad_len_divisor

        prob_r1 = 1.#self.conversion_pdf(energy=eg1, dist=conv_dist1)
        prob_r2 = 1.#self.conversion_pdf(energy=eg2, dist=conv_dist2)

        tshift = 0.01

        tmask1 = t1 >= tshift  # (conv_dist1 + 0.1)
        tmask2 = t2 >= tshift  # (conv_dist2 + 0.1)

        shower_lh1 = np.ones_like(r1)
        shower_lh2 = np.ones_like(r2)

        shower_lh1[tmask1] = self.shower_model.evaluate_shower_pdf(t=t1[tmask1], r=r1[tmask1], charge=shower_pts[:,3][tmask1],
                                                                   energy=eg1) * prob_r1
        shower_lh2[tmask2] = self.shower_model.evaluate_shower_pdf(t=t2[tmask2], r=r2[tmask2], charge=shower_pts[:,3][tmask2],
                                                                   energy=eg2) * prob_r2

        e1conv_dist = -0.0033 * eg1 + 24.2
        e2conv_dist = -0.0033 * eg2 + 24.2

        exten_point1 = prob_r1 * self.shower_model.evaluate_shower_pdf(t=tshift, r=0.1, charge=shower_pts[0, 3],
                                                                       energy=eg1)
        exten_point2 = prob_r2 * self.shower_model.evaluate_shower_pdf(t=tshift, r=0.1, charge=shower_pts[0, 3],
                                                                       energy=eg2)

        exp1 = np.exp(1.8 * (t1[~tmask1] * self.rad_len_divisor) / e1conv_dist) / (e1conv_dist / 1.8)
        exp2 = np.exp(1.8 * (t2[~tmask2] * self.rad_len_divisor) / e2conv_dist) / (e2conv_dist / 1.8)
        #shower_lh1[~tmask1] = exten_point1 * exp1 * self.shower_model.evaluate_shower_pdf(t=tshift, r=r1[~tmask1],
        #                                                                                  charge=shower_pts[:,3][~tmask1],
        #                                                                                  energy=eg1)
        #shower_lh2[~tmask2] = exten_point2 * exp2 * self.shower_model.evaluate_shower_pdf(t=tshift, r=r2[~tmask2],
        #                                                                                  charge=shower_pts[:,3][~tmask2],
        #                                                                                  energy=eg2)

        res = (np.nan_to_num(shower_lh1, nan=1.e-100, posinf=1.e10, neginf=1.e-100) +
               np.nan_to_num(shower_lh2, nan=1.e-100, posinf=1.e10, neginf=1.e-100)) * open_angle_lh  # convert nan -> 0.0

        ana_res = (np.nan_to_num(shower_lh1, nan=1.e-100, posinf=1.e10, neginf=1.e-100) *
                   np.nan_to_num(shower_lh2, nan=1.e-100, posinf=1.e10, neginf=1.e-100))
        ana_nll = -np.sum(np.log(np.nan_to_num(ana_res, nan=1.e-200) + 1.e-200)) / len(shower_pts)

        if self.debug:  # np.any(np.isnan(res)) or np.any(res < 0.):
            print("NaN", np.count_nonzero(np.isnan(res)), " <0", np.count_nonzero(res < 0))
            return np.ones_like(res) * -np.inf

        #return res
        return ana_nll - np.log(open_angle_lh)

    def pi0_probability_binned(self, shower_pts, eg1, eg2, theta1, theta2, conv_dist1, conv_dist2, f1=1000, f2=1000):
        """
        Assume we've rotated the points into the plane of the decay gammas
        """

        fake_eg1, fake_eg2 = f1,f2 #1000., 1000.
        phi = 0.

        Rs1 = np.array([[1., theta1, phi]])  # test shower 1 direction
        Rs2 = np.array([[1., theta2, phi]])  # test shower 2 direction
        alpha = np.arccos(futil.spherical_dot(Rs1, Rs2))
        #eg2 = etotal  # - eg1

        open_angle_lh = self.dn_dalpha_distribution(alpha=alpha, epi0=(eg1 + eg2))

        if self.debug:
            print("Theta 1/2", np.degrees(theta1), "/", np.degrees(theta2), " [deg]")
            print("cosα:", np.cos(alpha), "α:", np.degrees(alpha), "LH:", open_angle_lh)
            print("E1/E2:", eg1, "/", eg2)

        angle1 = np.arccos(futil.spherical_dot(Rs1, shower_pts[:, 0:3]))
        angle2 = np.arccos(futil.spherical_dot(Rs2, shower_pts[:, 0:3]))

        t1 = shower_pts[:, 0] * np.abs(np.cos(angle1)) - conv_dist1  # subtract before dividing by rad_len
        r1 = shower_pts[:, 0] * np.abs(np.sin(angle1))

        r1 /= self.moliere_divisor
        t1 /= self.rad_len_divisor

        t2 = shower_pts[:, 0] * np.abs(np.cos(angle2)) - conv_dist2  # subtract before dividing by rad_len
        r2 = shower_pts[:, 0] * np.abs(np.sin(angle2))

        r2 /= self.moliere_divisor
        t2 /= self.rad_len_divisor

        tshift = 0.01

        tmask1 = t1 >= tshift
        tmask2 = t2 >= tshift

        ana_s1 = np.ones_like(r1)
        ana_s2 = np.ones_like(r2)

        excess_energy1 = np.sum(shower_pts[:, 3][~tmask1 & (shower_pts[:, 3] > -1.)])
        excess_energy2 = np.sum(shower_pts[:, 3][~tmask2 & (shower_pts[:, 3] > -1.)])

        """
        Use the Analytic shower model to estimate the shower direction and conversion distance
        """

        ana_s1[tmask1] = self.ana_model.evaluate_shower_pdf(t=t1[tmask1], r=r1[tmask1], charge=shower_pts[:,3][tmask1],
                                                                   energy=fake_eg1) * prob_r1
        ana_s2[tmask2] = self.ana_model.evaluate_shower_pdf(t=t2[tmask2], r=r2[tmask2], charge=shower_pts[:,3][tmask2],
                                                                   energy=fake_eg2) * prob_r2

        #e1conv_dist = -0.0033 * eg1 + 24.2
        #e2conv_dist = -0.0033 * eg2 + 24.2


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

        binned_nll = -np.sum(np.log(binned_s1 * binned_s2 * q3p_s1 * q3p_s2 * open_angle_lh + 1.e-200))
        # binned_nll = -np.sum(np.log(binned_s1 * binned_s2 * open_angle_lh + 1.e-200))
        ana_nll = -np.sum(np.log(np.nan_to_num(ana_res, nan=1.e-200) + 1.e-200)) / len(shower_pts)

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

    # @staticmethod
    # def dn_dalpha_distribution(alpha, epi0):
    #     momentum = np.sqrt(epi0**2 - 135. * 135.)
    #     beta = (momentum / 135.) * np.sqrt(1 / (1 + (momentum / 135.) ** 2))
    #     gamma = 1 / np.sqrt(1 - beta ** 2)
    #
    #     diff_angle = 2 * (1 / (4. * gamma * beta)) * (np.cos(alpha / 2.) / np.sin(alpha / 2.)**2) * (
    #                 1 / np.sqrt(gamma ** 2 * np.sin(alpha / 2.) ** 2 - 1))
    #
    #     return diff_angle

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
