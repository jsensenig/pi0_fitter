import numpy as np
import math
import pickle
import cloudpickle
from scipy import stats
from abc import abstractmethod


class ShowerModelBase:

    def __init__(self, config):
        pass

    @abstractmethod
    def evaluate_shower_pdf(self, t, r, charge, energy):
        """
        Interface function to access the shower model. Receives
        points in t in radiation length and r in Moliere units.
        Returns the probability of a hit at points (t,r) for a
        given energy.
        :param charge: array[N]
        :param t: array[N]
        :param r: array[N]
        :param energy: float
        :return: array[N]
        """
        pass
    

class AnalyticShowerModel(ShowerModelBase):

    """
        Electromagnetic shower model taken from paper:
        "The Parameterized Simulation of Electromagnetic Showers
        in Homogeneous and Sampling Calorimeters"
        https://arxiv.org/pdf/hep-ex/0001020v1.pdf
    """

    def __init__(self, config):
        super().__init__(config=config)
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

    def evaluate_shower_pdf(self, t, r, charge, energy):
        tmp_radial = self.radial_profile(r, t, energy)
        tmp_long = self.longitudinal_profile(t, energy)

        # should be 1/sqrt(r)
        return (tmp_long * tmp_radial * (1. / np.sqrt(r))) / self.topo_normalization(energy)

    def radial_profile(self, r, t, e0):
        tau = t / self.tmax(e0)
        p_arg = (0.401 + 0.00187 * self.Z - tau) / (1.313 - 0.0686 * np.log(e0))
        p = np.sqrt((2.632 - 0.00094 * self.Z) * np.exp(p_arg - np.exp(p_arg)))

        tmp_radial = p * self.radial_rc(r, t, e0) + (1. - p) * self.radial_rt(r, t, e0)

        return tmp_radial # normalizing function to 1

    def test_radial_profile(self, r, t, e0, cutoff):
        #cutoff = 5.
        if type(t) == np.float64:
            if t > cutoff: t = cutoff
        else:
            t[t > cutoff] = cutoff

        tau = t / self.tmax(e0)
        p_arg = (0.401 + 0.00187 * self.Z - tau) / (1.313 - 0.0686 * np.log(e0))
        p = np.sqrt((2.632 - 0.00094 * self.Z) * np.exp(p_arg - np.exp(p_arg)))

        tmp_radial = p * self.radial_rc(r=r, t=(t / 1.), e0=e0) + (1. - p) * self.radial_rt(r=r, t=t, e0=e0)

        return tmp_radial / 5.6

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


class BinnedShowerModel(ShowerModelBase):

    def __init__(self, config):
        super().__init__(config=config)
        self.config = config["binned_shower_model"]
        self.analytic_model = AnalyticShowerModel(config=config)
        self.hit_selection_threshold = self.config["hit_selection_threshold"] # 0.1
        self.kde_file = self.config["kde_file"]
        self.charge_kde = None

        self.charge_pdf_just_calo_kde = None

        self.q3_fit_params = None
        self.qtotal_fit_params = None

        self.load_kde()
        self.energy_list = [] if self.charge_kde is None else np.array(list(self.charge_kde.keys())[0:-1], dtype=np.float64)

        # normal qsum
        # self.energy_norm = {800.0: 0.8397880242330674, 850.0: 0.857005859758964, 900.0: 0.8697732139752832,
        #                     950.0: 0.8802144605290095, 1000.0: 0.8935034134139729, 1050.0: 0.9004120457537446,
        #                     1100.0: 0.9077056398971941, 1150.0: 0.9183093604994915, 1200.0: 0.924934842721561}

        # 3D norm
        self.energy_norm = {800.0: 0.8287630492826533, 850.0: 0.8404425752693955, 900.0: 0.8498204381095625,
                            950.0: 0.8587290289079585, 1000.0: 0.8693138036576731, 1050.0: 0.8746628930374565,
                            1100.0: 0.8797030259308104, 1150.0: 0.8888759600933305, 1200.0: 0.8951575268051254}

        # weighted_avg*
        #self.energy_norm = {800.0: 0.9805671258123179, 850.0: 0.9863479585493605, 900.0: 0.9898481407888383,
        #                    950.0: 0.9923859640941197, 1000.0: 0.9947234869598383, 1050.0: 0.9955517119464418,
        #                    1100.0: 0.9968522645847565, 1150.0: 0.997797606853569, 1200.0: 0.9983132356693118}

        self.calo_energy_mean = {800.0: 47500.0, 850.0: 50500.0, 900.0: 53500.0, 950.0: 56500.0, 1000.0: 59500.0,
                                 1050.0: 63500.0, 1100.0: 66500.0, 1150.0: 69500.0, 1200.0: 72500.0}

    @staticmethod
    def calo_params(e0, pdict):
        k = pdict["k"][0] * e0 + pdict["k"][1]
        mu = pdict["mu"][0] * e0 + pdict["mu"][1]
        sigma = pdict["sigma"][0] * e0 + pdict["sigma"][1]
        return k, mu, sigma

    @staticmethod
    def get_q3_mpv(energy, pdict):
        return pdict["mu"][0] * energy + pdict["mu"][1] # mx + b

    @staticmethod
    def norm_func(x, kl, kh, mu, sigma):
        """
        Taken from https://arxiv.org/pdf/1603.08591.pdf
        """
        sqrt2 = np.sqrt(2.)

        condlist = [(x > 0.) & (x <= (mu - kl * sigma)),
                    (x > (mu - kl * sigma)) & (x <= (mu - kh * sigma)),
                    (x > (mu - kh * sigma))]  # & (x < (mu + kh * sigma))]

        choicelist = [(sigma / kl) * np.exp(0.5 * kl ** 2) * (np.exp(-kl ** 2) - np.exp(kl * mu / sigma)),  # I1
                      (sigma * sqrt2) * (math.erf(kl / sqrt2) - math.erf(kh / sqrt2)),  # I2
                      (sigma / kh) * np.exp(0.5 * kh ** 2) * (np.exp(3) - np.exp(kh * mu / sigma))]  # I3

        return np.select(condlist, choicelist, default=0.)

    @staticmethod
    def shower_func(x, A, k, mu, sigma, reverse=False, normalize=False):
        """
        Taken from https://arxiv.org/pdf/1603.08591.pdf
        """
        exp_arg = (x - mu) / sigma

        if reverse:
            if normalize:
                a1 = sigma * np.sqrt(np.pi / 2.) * (math.erf(k / np.sqrt(2)) + math.erf(mu / (sigma * np.sqrt(2))))
                a2 = (sigma / k) * np.exp(-0.5 * k ** 2)
                norm = a1 + a2
            else:
                norm = 1.
            return A * np.where(x <= mu + k * sigma, np.exp(-0.5 * exp_arg ** 2),
                                np.exp(0.5 * (k ** 2) - k * (exp_arg))) / norm
        else:  # original
            if normalize:
                a1 = -(sigma / k) * np.exp(-0.5 * k ** 2) * (np.exp(k * (k - (mu / sigma))) - 1.)
                a2 = sigma * np.sqrt(np.pi / 2.) * (math.erf(k / np.sqrt(2)) + 1.)
                norm = a1 + a2
            else:
                norm = 1.
            return A * np.where(x >= mu - k * sigma, np.exp(-0.5 * exp_arg ** 2),
                                np.exp(0.5 * (k ** 2) + k * (exp_arg))) / norm

    def calo_energy(self, qtotal, epi0):
        e1, e2 = epi0 / 2., epi0 / 2.
        k1, mu1, sigma1 = self.calo_params(e0=e1, pdict=self.qtotal_fit_params)
        k2, mu2, sigma2 = self.calo_params(e0=e2, pdict=self.qtotal_fit_params)

        sigma_tot = np.sqrt((sigma1 * mu1) ** 2 + (sigma2 * mu2) ** 2)
        q_prob = self.shower_func(x=qtotal, A=1., k=k1 + k2, mu=mu1 + mu2, sigma=sigma_tot,
                                  reverse=False, normalize=True)
        return q_prob

    @staticmethod
    def shower_func_2side(x, A, kl, kh, mu, sigma):
        """
        Taken from https://arxiv.org/pdf/1603.08591.pdf
        """
        exp_arg = (x - mu) / sigma

        sqrt2 = np.sqrt(2.)
        a1 = (sigma / kl) * np.exp(0.5 * kl ** 2 - (kl * mu) / sigma) * (np.exp((kl * mu) / sigma - kl ** 2) - 1.)
        a2 = sigma * np.sqrt(np.pi / 2.) * (math.erf(kl / sqrt2) + math.erf(kh / sqrt2))
        a3 = (sigma / kh) * np.exp(-0.5 * kh ** 2)
        norm = a1 + a2 + a3

        condlist = [x <= mu - kl * sigma, (x > mu - kl * sigma) & (x <= mu + kh * sigma), x > mu + kh * sigma]
        choicelist = [np.exp(0.5 * (kl ** 2) + kl * exp_arg), np.exp(-0.5 * exp_arg ** 2),
                      np.exp(0.5 * (kh ** 2) - kh * exp_arg)]

        return A * np.select(condlist, choicelist, default=0) / norm

    def convolve_q3(self, q3_prime, energy, energy2):
        kl, mu_qtot, sig_qtot = self.calo_params(e0=energy2, pdict=self.qtotal_fit_params)
        kh, mu_q3, sig_q3 = self.calo_params(e0=energy, pdict=self.q3_fit_params)

        sig_qtot = sig_qtot * mu_qtot / 2.
        sig_q3 = sig_q3 * mu_q3
        sig_quad = np.sqrt(sig_qtot**2 + sig_q3**2)

        return self.shower_func_2side(x=q3_prime, A=1., kl=kl, kh=kh, mu=(mu_qtot + mu_q3), sigma=sig_quad) # A=7.e-5

    def evaluate_shower_pdf(self, t, r, charge, energy, energy2, excess_energy, cutoff):

        if type(t) == float:
            print("Returning 1e-100")
            return 1.e-100

        # Evaluate the correlated charge deposition pdf for each hit
        result = self.charge_tr_pdf(t=t, r=r, charge=charge, energy=energy, energy2=energy2, excess_energy=excess_energy)

        return result

    def charge_tr_pdf(self, t, r, charge, energy, energy2, excess_energy):
        """

        :param t:
        :param r:
        :param charge:
        :param energy:
        :return:
        """
        cut_list = self.charge_kde["ranges"]

        qlist = []
        for t_lower, t_upper, r_lower, r_upper in cut_list:
            t_mask = (t_lower <= t) & (t < t_upper)
            r_mask = (r_lower <= r) & (r < r_upper)
            q_mask = r_mask & t_mask

            charge_sum = np.sum(charge[q_mask & (charge > -1)])
            qlist.append(charge_sum)

        # Since we only have 50MeV steps now we find the closest step
        min_idx = np.argmin(abs(self.energy_list - energy))
        e_key = self.energy_list[min_idx]

        # Convolve the 2nd shower with the Q3 from the 1st shower
        qlist = np.asarray(qlist)
        q3_prime = qlist[-1] + excess_energy
        q3_prime_prob = self.convolve_q3(q3_prime=q3_prime, energy=energy, energy2=energy2)

        q3_mpv = self.get_q3_mpv(energy=energy, pdict=self.q3_fit_params)
        qlist[-1] = q3_mpv

        #return self.charge_kde[e_key].evaluate(qlist) / self.energy_norm[e_key], q3_prime_prob

        return self.charge_kde[e_key].evaluate(qlist), q3_prime_prob

    def load_kde(self):

        with open(self.kde_file, 'rb') as ifile:
            self.charge_kde = pickle.load(ifile)

        if type(self.charge_kde[next(iter(self.charge_kde))]) != stats._kde.gaussian_kde:
            raise TypeError

        print("Loaded KDE file for BinnedShowerModel file:", self.kde_file)
        print(len(self.charge_kde["ranges"]), "D Ranges", self.charge_kde["ranges"])

        ###################

        # pdf_hist_file = "/Users/jsen/work/Protodune/analysis/pi0_fitter/etc/charge_pdf_just_calo_kde.pkl"
        pdf_hist_file = "/Users/jsen/work/Protodune/analysis/pi0_fitter/etc/charge_pdf_just_calo_kde_50_1200.pkl"
        with open(pdf_hist_file, 'rb') as ifile:
            self.charge_pdf_just_calo_kde = pickle.load(ifile)

        #####################
        # Dictionary to make Q3'

        pdf_hist_file = "/Users/jsen/work/Protodune/analysis/pi0_fitter/etc/qtotal_fit_params_v1.pkl"
        with open(pdf_hist_file, 'rb') as ifile:
            self.qtotal_fit_params = pickle.load(ifile)

        pdf_hist_file = "/Users/jsen/work/Protodune/analysis/pi0_fitter/etc/q3_fit_params_v1.pkl"
        with open(pdf_hist_file, 'rb') as ifile:
            self.q3_fit_params = pickle.load(ifile)
