import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

import pi0fit.fitter_utilities as futil


class Pi0Transformations:

    def __init__(self, config):
        self.config = config["pi0_transformations"]

        self.branch_list = self.config["branches"]
        self.debug = self.config["debug"]
        self.show_shower_plots = self.config["show_shower_plots"]
        self.use_fixed_endpoint = self.config["use_fixed_endpoint"]
        self.shower_direction_bins = self.config["shower_direction_bins"]

        self.xstart, self.ystart, self.zstart = self.config["true_pi0_start"]

    def transform_pi0_event(self, event_record):

        if not self.use_fixed_endpoint:
            self.xstart, self.ystart, self.zstart = self.get_beam_endpoint(event_record=event_record)

        # Transform points to the plane of the pi0 decay
        spherical_pts, p1_vec, p2_vec = self.reco_transform_to_pi0_plane(event_record=event_record)

        return spherical_pts, p1_vec, p2_vec

    def reco_transform_to_pi0_plane(self, event_record, evt):
        """
        Reco Version
        Uses reco information only.
        """
        # γ1 axis
        reco_theta, reco_phi = self.get_shower_direction(event_record=event_record)
        reco_gamma_dir = futil.spherical_to_cartesian(points=np.array([1., reco_theta, reco_phi]))  # [r,θ,ϕ]

        # π+ axis
        # Actually this axis can be arbitrary. In principle, we could just calculate the
        # angle of the points wrt to the one gamma shower direction.
        pi_spherical_dir = np.array([1., np.radians(18.), np.radians(10.)])
        reco_pip_dir = futil.spherical_to_cartesian(points=pi_spherical_dir)  # [r,θ,ϕ]

        if self.debug:
            print("reco_theta", reco_theta, "reco_phi", reco_phi)
            print("reco_gamma_dir", reco_gamma_dir)
            print("reco_pip_dir", reco_pip_dir)

        pts = np.vstack((ak.to_numpy(event_record["reco_all_spacePts_X"]) - self.xstart[evt],
                         ak.to_numpy(event_record["reco_all_spacePts_Y"]) - self.ystart[evt],
                         ak.to_numpy(event_record["reco_all_spacePts_Z"]) - self.zstart[evt])).T

        charge = ak.to_numpy(event_record["reco_all_spacePts_Integral"])
        charge = charge.reshape(len(charge), 1)

        gamma_dir = reco_gamma_dir / np.linalg.norm(reco_gamma_dir, axis=0)
        pip_dir = reco_pip_dir / np.linalg.norm(reco_pip_dir, axis=0)

        if self.debug:
            print("p1", gamma_dir)
            print("p2", pip_dir)

        test_plane = np.cross(gamma_dir, pip_dir)
        test_plane /= np.linalg.norm(test_plane)

        orth_axis = np.cross(gamma_dir, test_plane)
        orth_axis /= np.linalg.norm(orth_axis)

        # If we use the shower direction as the z-axis
        # in principle we can rotate in phi to find the pi0 decay plane
        # i.e. maximize the likelihood as a function of phi

        basis_mtx = np.array([orth_axis, test_plane, gamma_dir])

        if (gamma_dir @ orth_axis + gamma_dir @ test_plane + orth_axis @ test_plane) > 1.e-3:
            print("Failed Orthogonality Check")

        if self.debug:
            print("Basis/Pts Shape", basis_mtx.shape, "/", pts.shape)

        new_pts = basis_mtx @ pts.T
        spherical_pts = futil.single_to_spherical(v=new_pts)
        spherical_pts = np.hstack((spherical_pts.T, charge))

        try:
            p1r, p2r = (np.vstack((ak.to_numpy(event_record["pi0_gamma_startpx_initial"]),
                                   ak.to_numpy(event_record["pi0_gamma_startpy_initial"]),
                                   ak.to_numpy(event_record["pi0_gamma_startpz_initial"]))).T)
            p1 = p1r / np.linalg.norm(p1r, axis=0)
            p2 = p2r / np.linalg.norm(p2r, axis=0)
            p1_tran = basis_mtx @ p1.T
            p2_tran = basis_mtx @ p2.T
            flipped = np.argsort([futil.single_to_spherical(v=p1_tran)[1],
                                  futil.single_to_spherical(v=p2_tran)[1]])[0] == 1
        except:
            print("Failed to get truth gamma direction")
            flipped = False

        return spherical_pts, None, None, flipped #p1_vec, p2_vec

    def transform_point_to_spherical(self, event_record, rotate_polar_axis=False):

        tmp_us_x = event_record["reco_all_spacePts_X"]
        tmp_us_y = event_record["reco_all_spacePts_Y"]
        tmp_us_z = event_record["reco_all_spacePts_Z"]

        # Shift origin to beam interaction vertex
        # start point set at top function

        if rotate_polar_axis:
            tmp_x = tmp_us_x - self.xstart  # with SCE
            tmp_z = -(tmp_us_y - self.ystart)
            tmp_y = tmp_us_z - self.zstart
        else:
            tmp_x = tmp_us_x - self.xstart  # with SCE
            tmp_y = tmp_us_y - self.ystart
            tmp_z = tmp_us_z - self.zstart

        # rho
        xy = tmp_x*tmp_x + tmp_y*tmp_y

        # R
        event_record["reco_daughter_PFP_shower_spacePts_R"] = np.sqrt(xy + tmp_z*tmp_z)
        # Theta
        event_record["reco_daughter_PFP_shower_spacePts_Theta"] = np.arctan2(np.sqrt(xy), tmp_z)
        # event_record["reco_daughter_PFP_shower_spacePts_ThetaXZ"] = np.arctan2(tmp_x, tmp_z)
        # event_record["reco_daughter_PFP_shower_spacePts_ThetaYZ"] = np.arctan2(tmp_y, tmp_z)
        # Phi
        event_record["reco_daughter_PFP_shower_spacePts_Phi"] = np.arctan2(tmp_y, tmp_x)

    def get_shower_direction(self, event_record, gamma1=None, gamma2=None):

        theta = ak.to_numpy(event_record["reco_daughter_PFP_shower_spacePts_Theta"])
        phi = ak.to_numpy(event_record["reco_daughter_PFP_shower_spacePts_Phi"])

        if self.show_shower_plots:
            plt.figure(figsize=(12, 4))
            c, bx, by, _ = plt.hist2d(np.degrees(theta), np.degrees(phi),
                                      bins=[self.shower_direction_bins, self.shower_direction_bins],
                                      range=[[0, 90], [-180, 180]])
            if gamma1 is not None and gamma2 is not None:
                plt.plot(np.degrees(gamma1[1]), np.degrees(gamma1[2]), marker='*', markersize=12, color='orange')
                plt.plot(np.degrees(gamma2[1]), np.degrees(gamma2[2]), marker='*', markersize=12, color='red')
            plt.xlabel("$\\theta$ [deg]")
            plt.ylabel("$\phi$ [deg]")
            plt.colorbar()
            plt.show()
        else:
            c, bx, by = np.histogram2d(np.degrees(theta), np.degrees(phi),
                                       bins=[self.shower_direction_bins, self.shower_direction_bins],
                                       range=[[0, 90], [-180, 180]])

        bxc = (bx[1:] + bx[0:-1]) / 2.
        byc = (by[1:] + by[0:-1]) / 2.
        x_idx, y_idx = np.unravel_index(c.argmax(), c.shape)

        # Add an arbitrary angle to the shower. If we don't then the
        # center of the shower is the axis and the points symmetric about
        # this axis will have the same theta. This causes a bias when fitting the
        # shower direction.
        reco_theta = bxc[x_idx] + 10.

        return np.radians(reco_theta), np.radians(byc[y_idx])  # θ,ϕ

    def truth_transform_to_pi0_plane(self, event_record, evt):
        """
        Truth Version
         Uses true decay gamma directions
        """

        try:
            p1r, p2r = (np.vstack((ak.to_numpy(event_record["pi0_gamma_startpx_initial"]),
                                   ak.to_numpy(event_record["pi0_gamma_startpy_initial"]),
                                   ak.to_numpy(event_record["pi0_gamma_startpz_initial"]))).T)
            pts = np.vstack((ak.to_numpy(event_record["reco_all_spacePts_X"]) - self.xstart[evt],
                             ak.to_numpy(event_record["reco_all_spacePts_Y"]) - self.ystart[evt],
                             ak.to_numpy(event_record["reco_all_spacePts_Z"]) - self.zstart[evt])).T
        except:
            print("Failed to get event, skipping")
            print("Gamma E:", event_record["pi0_gamma_e_initial"])
            return None, None, None

        p1 = p1r / np.linalg.norm(p1r, axis=0)
        p2 = p2r / np.linalg.norm(p2r, axis=0)

        pi0_plane = np.cross(p1, p2)
        pi0_plane /= np.linalg.norm(pi0_plane)

        p2_orth = np.cross(p1, pi0_plane)
        p2_orth /= np.linalg.norm(p2_orth)

        basis_mtx = np.array([p1, p2_orth, pi0_plane])

        if (p1 @ p2_orth + p1 @ pi0_plane + p2_orth @ pi0_plane) > 1.e-3:
            print("Failed Orthogonality Check")

        new_pts = basis_mtx @ pts.T
        p1_vec = basis_mtx @ p1
        p2_vec = basis_mtx @ p2

        p1_vec /= np.linalg.norm(p1_vec)
        p2_vec /= np.linalg.norm(p2_vec)

        spherical_pts = futil.single_to_spherical(v=new_pts)
        return_pts = np.array([spherical_pts[0, :], spherical_pts[2, :], spherical_pts[1, :]])  # phi and theta swapped

        flipped = np.argsort([futil.single_to_spherical(v=p1_vec)[1],
                              futil.single_to_spherical(v=p2_vec)[1]])[0] == 1

        return return_pts, p1_vec, p2_vec, flipped

    def set_beam_endpoint(self, all_event_record):

        if self.use_fixed_endpoint:
            self.xstart = ak.Array([self.xstart] * len(all_event_record))
            self.ystart = ak.Array([self.ystart] * len(all_event_record))
            self.zstart = ak.Array([self.zstart] * len(all_event_record))
        else:
            self.xstart = all_event_record["reco_beam_endX"]
            self.ystart = all_event_record["reco_beam_endY"]
            self.zstart = all_event_record["reco_beam_endZ"]
