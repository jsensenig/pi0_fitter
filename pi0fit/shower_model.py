import numpy as np
import math
import pickle


class AnalyticShowerModel:

    """
        Electromagnetic shower model taken from paper:
        "The Parameterized Simulation of Electromagnetic Showers
        in Homogeneous and Sampling Calorimeters"
        https://arxiv.org/pdf/hep-ex/0001020v1.pdf
    """

    def __init__(self, config):

        self.config = config["analytic_shower_model"]
        self.Z = 18

        '''
            Values from: https://pdg.lbl.gov/2009/AtomicNuclearProperties/HTML_PAGES/289.html
            E_c = 32.38: Avg. of -> e- = 32.84 and e+ = 31.91 for Ar
        '''
        self.e_critical = 32.375

        topo_norm_file = self.config["normalization_splines"]
        with open(topo_norm_file, 'rb') as f:
            self.topo_normalization = pickle.load(f)

    def longitudinal_radial_profile(self, r, t, e0):
        tmp_radial = self.radial_profile(r, t, e0)
        tmp_long = self.longitudinal_profile(t, e0)

        # should be 1/sqrt(r)
        return (tmp_long * tmp_radial * (1. / np.sqrt(r))) / self.topo_normalization(e0)

    def radial_profile(self, r, t, e0):
        tau = t / self.tmax(e0)
        p_arg = (0.401 + 0.00187 * self.Z - tau) / (1.313 - 0.0686 * np.log(e0))
        p = np.sqrt((2.632 - 0.00094 * self.Z) * np.exp(p_arg - np.exp(p_arg)))

        tmp_radial = p * self.radial_rc(r, t, e0) + (1. - p) * self.radial_rt(r, t, e0)

        return tmp_radial

    def radial_p(self, t, e0):
        tau = t / self.tmax(e0)
        p_arg = (0.401 + 0.00187 * self.Z - tau) / (1.313 - 0.0686 * np.log(e0))
        return (2.632 - 0.00094 * self.Z) * np.exp(p_arg - np.exp(p_arg))

    def radial_rc(self, r, t, e0):
        tau = t / self.tmax(e0)
        Rc = 0.0251 + 0.00319 * np.log(e0) + (0.1162 - 0.000381 * self.Z) * tau

        A = ((2. * (Rc / np.sqrt(3)) * Rc ** 2) / ((Rc / np.sqrt(3)) ** 2 + Rc ** 2) ** 2)  # max at r = Rc/sqrt(3)

        return np.where(r >= Rc / np.sqrt(3), ((2. * r * Rc ** 2) / (r ** 2 + Rc ** 2) ** 2),
                        A * np.exp(-(r - Rc / np.sqrt(3))))

    def radial_rt(self, r, t, e0):
        tau = t / self.tmax(e0)
        Rt = (0.659 - 0.00309 * self.Z) * (
                np.exp(-2.59 * (tau - 0.645)) + np.exp((0.3585 + 0.0421 * np.log(e0)) * (tau - 0.645)))

        return (2. * r * Rt**2) / (r**2 + Rt**2)**2

    def longitudinal_profile(self, t, e0):
        # Constants
        a = 0.21 + (0.492 + 2.38 / 18.) * np.log(e0 / self.e_critical)
        b = (a - 1.) / self.tmax(e0)

        yfit = b * ((b * t) ** (a - 1.)) * (np.e ** (-b * t)) / math.gamma(a)
        return yfit

    def tmax(self, e0):
        """
            T is average maximum shower length for a given material
            (encapsulated in the critical energy Ec) as a function of photon energy e0.
            e0: (float) Photon energy (units: [MeV])
        """
        # return np.log(e0/E_c) - 0.858 #LAr
        return np.log(e0 / self.e_critical) - 0.329  # divergence at 50MeV
