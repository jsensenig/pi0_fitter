import numpy as np
from dataclasses import dataclass
from prettytable import PrettyTable


def spherical_dot(x1, x2):
    """
    Take the dot product of 2 vectors in spherical coordinates
    :param x1: array[N,3]
    :param x2: array[N,3]
    :return: array[N]
    """
    xyz1 = spherical_to_cartesian(x1)[0]
    xyz2 = spherical_to_cartesian(x2)[0]

    xyz1 /= np.linalg.norm(xyz1, axis=0)
    xyz2 /= np.linalg.norm(xyz2, axis=0)

    return xyz1 @ xyz2.T


def spherical_to_cartesian(points):  # X = [r,theta,phi]
    """
    Convert points from spherical to cartesian coordinates
    :param points: array[N,3]
    :return: array[N,3]
    """
    if points.ndim == 2:
        x = points[:, 0] * np.cos(points[:, 2]) * np.sin(points[:, 1])
        y = points[:, 0] * np.sin(points[:, 2]) * np.sin(points[:, 1])
        z = points[:, 0] * np.cos(points[:, 1])
        return np.vstack((x, y, z)).T
    elif points.ndim == 1:
        x = points[0] * np.cos(points[2]) * np.sin(points[1])
        y = points[0] * np.sin(points[2]) * np.sin(points[1])
        z = points[0] * np.cos(points[1])
        return np.array([x, y, z])
    else:
        print("Dim=", points.ndim, "not supported!")
        raise TypeError


def single_to_spherical(v):  # [x,y,z]
    x, y, z = v
    # rho
    xy = x ** 2 + y ** 2
    # R,Theta,Phi
    r = np.sqrt(xy + z ** 2)
    theta = np.arctan2(np.sqrt(xy), z)
    phi = np.arctan2(y, x)

    return np.array([r, theta, phi])


def pi0_angles(epi0, cos_pi0, a):

    e1 = a * epi0
    e2 = (1. - a) * epi0
    alpha = np.arccos(1. - (135. ** 2 / (2. * e1 * e2)))
    ppi0 = np.sqrt(e1 ** 2 + e2 ** 2 + 2. * e1 * e2 * np.cos(alpha))

    A = a + (1. - a) * np.cos(alpha)
    B = (1. - a) * np.sin(alpha)
    C = (ppi0 * cos_pi0) / epi0

    theta1 = np.arccos(C / np.sqrt(A ** 2 + B ** 2)) + np.arctan(B / A)

    # The pdf is cyclic every 90deg e.g. +90deg = -90deg
    if abs(theta1) > np.pi / 2.:
        theta1 = np.sign(theta1) * ((abs(theta1) % np.pi) - np.pi)  # +/- (θ1 % π) - π

    #theta2 = theta1 - alpha
    #theta2 = -theta1
    #theta1 = theta2 + alpha

    theta2 = theta1 + alpha

    # if abs(theta2) > np.pi/2.:
    #     theta2 = np.sign(theta2) * ((abs(theta2) % np.pi) - np.pi)

    return alpha, theta1, theta2


@dataclass
class FitResults:

    epi0: float
    cos_pi0: float
    eg1: float
    eg2: float
    theta1: float
    theta2: float
    c1: float
    c2: float
    open_angle: float
    is_truth: bool

    def __str__(self):
        t = PrettyTable(self.class_labels())
        t.add_row(self.values_as_array())
        return t

    @staticmethod
    def class_labels():
        return ["Eπ0", "cosθ_π0", "Eγ1", "Eγ2", "θγ1", "θγ2", "Open Angle", "C1", "C2", "istruth"]

    def set_event_values(self, epi0, cos_pi0, eg1, c1, c2, is_truth):
        self.epi0 = epi0
        self.cos_pi0 = cos_pi0
        self.eg1 = eg1
        self.eg2 = self.epi0 - self.eg1
        self.c1 = c1
        self.c2 = c2
        self.is_truth = is_truth
        self.open_angle, self.theta1, self.theta2 = pi0_angles(epi0=self.epi0, cos_pi0=self.cos_pi0, a=self.eg1/self.epi0)

    def values_as_array(self):
        return np.round(np.array([self.epi0, self.cos_pi0, self.eg1, self.eg2, np.degrees(self.theta1),
                                  np.degrees(self.theta2), np.degrees(self.open_angle), self.c1, self.c2,
                                  self.is_truth]), 2)

    def values_as_dict(self):
        return {"epi0": self.epi0, "cos_pi0": self.cos_pi0, "eg1": self.eg1, "eg2": self.eg2, "theta1": self.theta1,
                "theta2": self.theta2, "c1": self.c1, "c2": self.c2, "open_angle": self.open_angle}

    def print_comparison_table(self, fit_result):

        t = PrettyTable(self.class_labels())
        t.add_row(fit_result.values_as_array())
        t.add_row(self.values_as_array())
        print(t)

    def calculate_metrics(self, fit_result):
        """
        Compare the class instance fit results to another result, e.g. the true values
        :param fit_result: FitResults object
        :return:
        """
        labels = ["Eπ0: reco/true-1", "cosθ_π0: true-reco", "Eγ1: reco/true-1", "C1: true-reco", "C2: true-reco"]
        epi0_bias = self.epi0 / fit_result.epi0 - 1.
        cos_pi0_diff = fit_result.cos_pi0 - self.cos_pi0
        eg1_bias = self.eg1 / fit_result.eg1 - 1.
        c1_diff = fit_result.c1 - self.c1
        c2_diff = fit_result.c2 - self.c2

        t = PrettyTable(labels)
        t.add_row(np.round(np.array([epi0_bias, cos_pi0_diff, eg1_bias, c1_diff, c2_diff]), 3))
        print(t)
