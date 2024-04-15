import numpy as np
import awkward as ak

from pi0fit.pi0_transform import Pi0Transformations
import pi0fit.fitter_utilities as futil
from pi0fit.fitter_utilities import FitResults
from pi0fit.pi0_likelihood_minimizer import Pi0MinimizerBase


class Pi0Fitter:

    def __init__(self, config):

        self.config = config["pi0_fitter"]
        self.minimizer = self.config["minimizer"]
        self.debug = self.config["debug"]

        self.pi0_transform = Pi0Transformations(config=config)
        self.truth_comparison = self.config["truth_comparison"]
        self.use_true_transform = self.config["use_true_point_transform"]
        self.lower_range, self.upper_range = self.config["fit_range"]
        self.fit_all_events = self.config["fit_all_events"]
        self.return_pi0_only = self.config["return_pi0_only"]
        self.rotate_polar_axis = self.config["rotate_polar_axis"]
        self.transform_points = self.config["transform_points"]
        self.use_true_vertex = self.config["use_true_vertex"]

        self.minimizer_obj = futil.get_class(selected_class=self.minimizer,
                                             base_class=Pi0MinimizerBase,
                                             config=config)

    def fit_pi0(self, all_event_record, pi0_points=None):

        if self.transform_points:
            # Set the event pi0 start point
            self.pi0_transform.set_beam_endpoint(all_event_record=all_event_record)

            # Convert 3D hits to spherical coordinates
            self.pi0_transform.transform_point_to_spherical(event_record=all_event_record,
                                                            rotate_polar_axis=self.rotate_polar_axis)

        if self.fit_all_events: self.upper_range = len(all_event_record)
        fit_event = False

        for evt in range(self.lower_range, self.upper_range):
            print("######## Event:", evt)
            event_record = all_event_record[evt]

            # Get 3D points for event
            if pi0_points is None:
                # pi0_points = self.get_event_points(event_record=event_record)
                pi0_points = self.get_event_points(event_record=event_record, event=evt, return_spherical=True,
                                                   cosmics=False, get_gammas=False)

            # Get event truth information
            truth_values = None
            if self.truth_comparison:
                truth_values = self.get_event_truth_values(event_record=event_record)

            # Minimize fit
            self.minimizer_obj.minimize(pi0_points=pi0_points, truth_values=truth_values)

            #if self.return_pi0_only:
            #    return self.minimizer_obj.epi0, self.minimizer_obj.cos_pi0
            print("Fit Result for event", evt)
            print(self.minimizer_obj.values_as_dict())
            fit_event = True
        if fit_event:
            return self.minimizer_obj.values_as_dict(), self.minimizer_obj.comparison_as_dict(fit_result=truth_values)
        else:
            return None, None

    def get_event_points(self, event_record, event, return_spherical, cosmics=False, get_gammas=True):

        pts_shift,_ = self.get_vertex(event_record=event_record, event=event)

        pts = np.vstack((ak.to_numpy(event_record["reco_all_spacePts_X", event]) - pts_shift[0],
                         ak.to_numpy(event_record["reco_all_spacePts_Y", event]) - pts_shift[1],
                         ak.to_numpy(event_record["reco_all_spacePts_Z", event]) - pts_shift[2])).T

        if cosmics:
            cosmic_pts = np.vstack((ak.to_numpy(event_record["cosmic_pfp_spacePts_X", event]) - pts_shift[0],
                                    ak.to_numpy(event_record["cosmic_pfp_spacePts_Y", event]) - pts_shift[1],
                                    ak.to_numpy(event_record["cosmic_pfp_spacePts_Z", event]) - pts_shift[2])).T

        if get_gammas:
            ssdir = np.vstack((ak.to_numpy(event_record["true_beam_Pi0_decay_startPx", event]),
                               ak.to_numpy(event_record["true_beam_Pi0_decay_startPy", event]),
                               ak.to_numpy(event_record["true_beam_Pi0_decay_startPz", event]))).T

            sdir1 = ssdir[0] / (np.sqrt(ssdir[0] @ ssdir[0]))
            sdir2 = ssdir[1] / (np.sqrt(ssdir[1] @ ssdir[1]))

            spherical_sdir1 = futil.single_to_spherical(v=sdir1, rotate_polar_axis=False).T
            spherical_sdir2 = futil.single_to_spherical(v=sdir2, rotate_polar_axis=False).T

        event_charge = ak.to_numpy(event_record["reco_all_spacePts_Integral", event])
        spherical_pts_tmp = futil.single_to_spherical(v=pts.T).T
        spherical_pts = np.vstack((spherical_pts_tmp[:, 0], np.degrees(spherical_pts_tmp[:, 1]),
                                   np.degrees(spherical_pts_tmp[:, 2]), event_charge)).T

        if cosmics:
            if return_spherical:
                spherical_cosmic_pts_tmp = futil.single_to_spherical(v=cosmic_pts.T).T
                spherical_cosmic_pts = np.vstack((spherical_cosmic_pts_tmp[:, 0],
                                                  np.degrees(spherical_cosmic_pts_tmp[:, 1]),
                                                  np.degrees(spherical_cosmic_pts_tmp[:, 2]))).T
                if get_gammas:
                    return spherical_pts, spherical_sdir1, spherical_sdir2, spherical_cosmic_pts
                else:
                    return spherical_pts, spherical_cosmic_pts
            else:
                if get_gammas:
                    return pts, sdir1, sdir2, cosmic_pts
                else:
                    return pts, cosmic_pts
        else:
            if return_spherical:
                if get_gammas:
                    return spherical_pts, spherical_sdir1, spherical_sdir2
                else:
                    return spherical_pts
            else:
                if get_gammas:
                    return pts, sdir1, sdir2
                else:
                    return pts

    def get_vertex(self, event_record, event):

        xt = event_record["true_beam_endX_SCE", event]
        yt = event_record["true_beam_endY_SCE", event]
        zt = event_record["true_beam_endZ_SCE", event]

        xr = event_record["reco_beam_endX", event]
        yr = event_record["reco_beam_endY", event]
        zr = event_record["reco_beam_endZ", event]

        delta_r = np.sqrt((xt - xr) ** 2 + (yt - yr) ** 2 + (zt - zr) ** 2)
        print("Δx/Δy/Δz/Δr", np.round(xt - xr, 3), "/",
                             np.round(yt - yr, 3), "/",
                             np.round(zt - zr, 3), "/", delta_r)

        vertex = [xt, yt, zt] if self.use_true_vertex else [xr, yr, zr]
        return vertex, delta_r

    def get_event_truth_values(self, event_record, out_file=None):

        e1, e2 = event_record["true_beam_Pi0_decay_startP"] * 1.e3
        c1, c2 = 0., 0. #event_record["true_decay_gamma_conv_dist"]
        oa = event_record["true_beam_Pi0_decay_OA"]

        ssdir = np.vstack((ak.to_numpy(event_record["true_beam_Pi0_decay_startPx"]),
                           ak.to_numpy(event_record["true_beam_Pi0_decay_startPy"]),
                           ak.to_numpy(event_record["true_beam_Pi0_decay_startPz"]))).T

        sdir0 = ssdir[0] / (np.sqrt(ssdir[0] @ ssdir[0]))
        sdir1 = ssdir[1] / (np.sqrt(ssdir[1] @ ssdir[1]))
        spherical_sdir0 = futil.single_to_spherical(v=sdir0, rotate_polar_axis=self.rotate_polar_axis).T
        spherical_sdir1 = futil.single_to_spherical(v=sdir1, rotate_polar_axis=self.rotate_polar_axis).T

        true_theta1, true_phi1 = np.degrees(spherical_sdir0[1:])
        true_theta2, true_phi2 = np.degrees(spherical_sdir1[1:])

        ppi0 = np.sqrt(e1**2 + e2**2 + 2. * e1 * e2 * np.cos(oa))
        cos_pi0 = (e1 + e2 * np.cos(-oa)) / ppi0

        if out_file is None:
            print("True Eπ0:", np.sum(event_record["true_beam_Pi0_decay_startP"]) * 1.e3)
            print("True Gamma Eγ:", event_record["true_beam_Pi0_decay_startP"] * 1.e3)
            print("True OA:", np.degrees(oa))
            print("True cos_pi0:", cos_pi0)
            print("True Gamma Energy 1/2", round(e1, 2), "/", round(e2, 2))
            print("True Gamma Theta 1/2", round(true_theta1, 2), "/", round(true_theta2, 2))
            print("True Gamma Phi 1/2", round(true_phi1, 2), "/", round(true_phi2, 2))
            print("C1/C2:", c1, "/", c2)
        else:
            print("True Eπ0:", np.sum(event_record["true_beam_Pi0_decay_startP"]) * 1.e3, file=out_file)
            print("True Gamma Eγ:", event_record["true_beam_Pi0_decay_startP"] * 1.e3, file=out_file)
            print("True OA:", np.degrees(oa), file=out_file)
            print("True cos_pi0:", cos_pi0, file=out_file)
            print("True Gamma Energy 1/2", round(e1, 2), "/", round(e2, 2), file=out_file)
            print("True Gamma Theta 1/2", round(true_theta1, 2), "/", round(true_theta2, 2), file=out_file)
            print("True Gamma Phi 1/2", round(true_phi1, 2), "/", round(true_phi2, 2), file=out_file)
            print("C1/C2:", c1, "/", c2, file=out_file)

        truth_values = FitResults(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False)
        truth_values.set_event_values_shower(eg1=e1, eg2=e2, theta1=true_theta1, theta2=true_theta2,
                                             phi1=true_phi1, phi2=true_phi2, c1=c1, c2=c2, is_truth=True)

        return truth_values
