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

        pts = np.vstack((ak.to_numpy(event_record["reco_all_SpHit_X"]) - self.xstart[evt],
                         ak.to_numpy(event_record["reco_all_SpHit_Y"]) - self.ystart[evt],
                         ak.to_numpy(event_record["reco_all_SpHit_Z"]) - self.zstart[evt])).T

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

        return spherical_pts, None, None #p1_vec, p2_vec

    def transform_point_to_spherical(self, event_record):

        tmp_x = event_record["reco_all_SpHit_X"]
        tmp_y = event_record["reco_all_SpHit_Y"]
        tmp_z = event_record["reco_all_SpHit_Z"]

        # Shift origin to beam interaction vertex
        # start point set at top function
        tmp_x = tmp_x - self.xstart  # with SCE
        tmp_y = tmp_y - self.ystart
        tmp_z = tmp_z - self.zstart

        # rho
        xy = tmp_x ** 2 + tmp_y ** 2

        # R
        event_record["reco_daughter_PFP_shower_spacePts_R"] = np.sqrt(xy + tmp_z**2)
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

        return np.radians(bxc[x_idx]), np.radians(byc[y_idx])  # θ,ϕ

    def truth_transform_to_pi0_plane(self, event_record, evt):
        """
        Truth Version
         Uses true decay gamma directions
        """

        try:
            p2r, p1r = (np.vstack((ak.to_numpy(event_record["pi0_gamma_px_initial"]),
                                   ak.to_numpy(event_record["pi0_gamma_py_initial"]),
                                   ak.to_numpy(event_record["pi0_gamma_pz_initial"]))).T)
            pts = np.vstack((ak.to_numpy(event_record["reco_all_SpHit_X"]) - self.xstart[evt],
                             ak.to_numpy(event_record["reco_all_SpHit_Y"]) - self.ystart[evt],
                             ak.to_numpy(event_record["reco_all_SpHit_Z"]) - self.zstart[evt])).T
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

        return return_pts, p1_vec, p2_vec

    def set_beam_endpoint(self, all_event_record):

        if self.use_fixed_endpoint:
            self.xstart = ak.Array([self.xstart] * len(all_event_record))
            self.ystart = ak.Array([self.ystart] * len(all_event_record))
            self.zstart = ak.Array([self.zstart] * len(all_event_record))
        else:
            self.xstart = all_event_record["reco_beam_endX"]
            self.ystart = all_event_record["reco_beam_endY"]
            self.zstart = all_event_record["reco_beam_endZ"]
