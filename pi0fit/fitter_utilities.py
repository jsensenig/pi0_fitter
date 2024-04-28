import numpy as np
from dataclasses import dataclass
from prettytable import PrettyTable


def get_class(selected_class, base_class, config):
    """
    Find requested class and return configured object.
    :return:
    """
    available_classes = {cls.__name__: cls for cls in base_class.__subclasses__()}

    if selected_class not in available_classes.keys():
        print("Unknown Class", selected_class, "must be one of", list(available_classes.keys()))
        raise KeyError

    return available_classes[selected_class](config)


def spherical_dot(x1, x2, spherical=True):
    """
    Take the dot product of 2 vectors in spherical coordinates
    :param x1: array[N,3]
    :param x2: array[N,3]
    :return: array[N]
    """
    if spherical:
        xyz1 = spherical_to_cartesian(x1)#[0]
        xyz2 = spherical_to_cartesian(x2)#[0]
    else:
        xyz1 = x1
        xyz2 = x2

    norm1 = np.linalg.norm(xyz1, axis=1)
    norm1.reshape(norm1.size, 1)
    xyz1p = (xyz1.T / norm1).T

    norm2 = np.linalg.norm(xyz2, axis=1)
    norm2.reshape(norm2.size, 1)
    xyz2p = (xyz2.T / norm2).T

    return (xyz1p @ xyz2p.T).T[:, 0]


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


def single_to_spherical(v, rotate_polar_axis=False):  # [x,y,z]
    """
    Input: (3,N) = (XYZ,N)
    Default coordinate system sets z to be polar axis
    setting `rotated_polar_axis=True` rotates about x
    and sets y to be the polar axis i.e.,
    y' = z
    z' = -y
    """
    if rotate_polar_axis:
        x, z, y = v
        z *= -1.
    else:
        x, y, z = v
    # rho
    xy = x*x + y*y
    # R,Theta,Phi
    r = np.sqrt(xy + z*z)
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


def mc_accept_reject_1d(x, pdf_func, N, **kwargs):
    """
    1D monte carlo sampler
    :param x:
    :param pdf_func:
    :param N:
    :param kwargs:
    :return:
    """
    # Normalize the pdf to its mode/MPV
    f = pdf_func(x, **kwargs)
    f /= np.max(f)

    # 1D uniform random numbers
    xr = np.random.uniform(0, 1, N)

    # Acceptance/Rejection mask
    accept_reject_mask = xr < f

    return x[accept_reject_mask]


def mc_accept_reject_2d(xmesh, ymesh, pdf_func, N, **kwargs):
    """
    2D monte carlo sampler accepting x and y from np.meshgrid
    :param xmesh:
    :param ymesh:
    :param pdf_func:
    :param N:
    :param kwargs:
    :return:
    """
    # Normalize the pdf to its mode/MPV
    f = pdf_func(xmesh, ymesh, **kwargs)
    f /= np.max(f)

    # 2D uniform random number mesh
    Xr, _ = np.meshgrid(np.random.uniform(0, 1, N), np.random.uniform(0, 1, N))

    # Acceptance/Rejection mask
    accept_reject_mask = Xr < f

    print("Accepted:", np.count_nonzero(accept_reject_mask))

    return xmesh.T[accept_reject_mask], ymesh[accept_reject_mask]


def mc_accept_reject_2d_nonmesh(x, y, pdf_func, N, **kwargs):
    """
    2D monte carlo sampler accepting independent x and y
    :param x:
    :param y:
    :param pdf_func:
    :param N:
    :param kwargs:
    :return:
    """
    # Normalize the pdf to its mode/MPV
    f = pdf_func(x, y, **kwargs)
    f /= np.max(f)

    # 2D uniform random number
    xr, yr = np.random.uniform(0, 1, N), np.random.uniform(0, 1, N)

    # Acceptance/Rejection mask
    accept_reject_mask = yr < f

    print("Accepted:", np.count_nonzero(accept_reject_mask))
    print(accept_reject_mask.shape)

    return x[accept_reject_mask], y[accept_reject_mask]


@dataclass
class FitResults:

    epi0: float
    cos_pi0: float
    eg1: float
    eg2: float
    theta1: float
    theta2: float
    phi1: float
    phi2: float
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
        return ["Eπ0", "cosθ_π0", "Eγ1", "Eγ2", "θγ1", "θγ2", "ϕ1", "ϕ2", "Open Angle", "C1", "C2", "istruth"]

    def set_event_values(self, epi0, cos_pi0, eg1, c1, c2, is_truth):
        self.epi0 = epi0
        self.cos_pi0 = cos_pi0
        self.eg1 = eg1
        self.eg2 = self.epi0 - self.eg1
        self.c1 = c1
        self.c2 = c2
        self.is_truth = is_truth
        self.open_angle, self.theta1, self.theta2 = pi0_angles(epi0=self.epi0, cos_pi0=self.cos_pi0, a=self.eg1/self.epi0)

    def set_event_values_shower(self, eg1, eg2, theta1, theta2, phi1, phi2, c1, c2, is_truth):
        self.eg1 = eg1
        self.c1 = c1
        self.c2 = c2
        self.is_truth = is_truth
        self.theta1 = theta1
        self.theta2 = theta2
        self.phi1 = phi1
        self.phi2 = phi2
        self.open_angle = np.arccos(spherical_dot(np.array([[1., np.radians(theta1), np.radians(phi1)]]),
                                                  np.array([[1., np.radians(theta2), np.radians(phi2)]])))[0]

        self.eg2 = eg2 #(135. * 135.) / (2. * self.eg1 * (1. - np.cos(self.open_angle))) if eg2 is None else eg2
        self.epi0 = eg1 + eg2

        ppi0 = np.sqrt(eg1 ** 2 + self.eg2 ** 2 + 2. * eg1 * self.eg2 * np.cos(self.open_angle))
        #self.cos_pi0 = (eg1 + self.eg2 * np.cos(self.open_angle)) / ppi0
        self.cos_pi0 = (eg1 * np.cos(np.radians(theta1)) + eg2 * np.cos(np.radians(theta2))) / ppi0

    def values_as_array(self):
        return np.round(np.array([self.epi0, self.cos_pi0, self.eg1, self.eg2, self.theta1, self.theta2,
                                  self.phi1, self.phi2, np.degrees(self.open_angle), self.c1, self.c2,
                                  self.is_truth]), 2)

    def values_as_dict(self):
        return {"epi0": self.epi0, "cos_pi0": self.cos_pi0, "eg1": self.eg1, "eg2": self.eg2,
                "theta1": self.theta1, "theta2": self.theta2, "phi1": self.phi1, "phi2": self.phi2,
                "c1": self.c1, "c2": self.c2, "open_angle": np.degrees(self.open_angle)}

    def comparison_as_dict(self, fit_result):

        epi0_bias = self.epi0 / fit_result.epi0 - 1.
        cos_pi0_diff = fit_result.cos_pi0 - self.cos_pi0
        eg1_bias = self.eg1 / fit_result.eg1 - 1.
        eg2_bias = self.eg2 / fit_result.eg2 - 1.

        in_dir1 = np.array([[1., np.radians(fit_result.theta1), np.radians(fit_result.phi1)]])
        in_dir2 = np.array([[1., np.radians(fit_result.theta2), np.radians(fit_result.phi2)]])

        self_dir1 = np.array([[1., np.radians(self.theta1), np.radians(self.phi1)]])
        self_dir2 = np.array([[1., np.radians(self.theta2), np.radians(self.phi2)]])

        cos_dir1 = spherical_dot(self_dir1, in_dir1)[0]
        cos_dir2 = spherical_dot(self_dir2, in_dir2)[0]

        return {"epi0_bias": epi0_bias, "cos_pi0_diff": cos_pi0_diff, "eg1_bias": eg1_bias, "eg2_bias": eg2_bias,
                "cos_theta1": cos_dir1, "cos_theta2": cos_dir2}

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
        labels = ["Eπ0: reco/true-1", "cosθ_π0: true-reco", "Eγ1: reco/true-1", "Eγ2: reco/true-1",
                  "Dir1: cosθ", "Dir2: cosθ"]
        epi0_bias = self.epi0 / fit_result.epi0 - 1.
        cos_pi0_diff = fit_result.cos_pi0 - self.cos_pi0
        eg1_bias = self.eg1 / fit_result.eg1 - 1.
        eg2_bias = self.eg2 / fit_result.eg2 - 1.
        c1_diff = fit_result.c1 - self.c1
        c2_diff = fit_result.c2 - self.c2

        in_dir1 = np.array([[1., np.radians(fit_result.theta1), np.radians(fit_result.phi1)]])
        in_dir2 = np.array([[1., np.radians(fit_result.theta2), np.radians(fit_result.phi2)]])

        self_dir1 = np.array([[1., np.radians(self.theta1), np.radians(self.phi1)]])
        self_dir2 = np.array([[1., np.radians(self.theta2), np.radians(self.phi2)]])

        cos_dir1 = spherical_dot(self_dir1, in_dir1)[0]
        cos_dir2 = spherical_dot(self_dir2, in_dir2)[0]

        t = PrettyTable(labels)
        t.add_row(np.round(np.array([epi0_bias, cos_pi0_diff, eg1_bias, eg2_bias, cos_dir1, cos_dir2]), 3))
        print(t)
