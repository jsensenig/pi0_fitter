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

            # if len(event_record["pi0_gamma_starte_initial"]) > 2:
            #     continue

            # Get 3D points for event
            if pi0_points is None:
                pi0_points = self.get_event_points(event_record=event_record)

            # Get event truth information
            truth_values = None
            if self.truth_comparison:
                truth_values = self.get_event_truth_values(event_record=event_record)

            if truth_values.eg1 < 45 or truth_values.eg1 > 1200 or truth_values.eg2 < 45 or truth_values.eg2 > 1200:
                continue
            print("MADE IT!!")

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

    @staticmethod
    def get_event_points(event_record):

        r = ak.to_numpy(event_record["reco_daughter_PFP_shower_spacePts_R"])
        theta = ak.to_numpy(event_record["reco_daughter_PFP_shower_spacePts_Theta"])
        phi = ak.to_numpy(event_record["reco_daughter_PFP_shower_spacePts_Phi"])
        charge = ak.to_numpy(event_record["reco_all_spacePts_Integral"])
        em_score = ak.to_numpy(event_record["reco_all_spacePts_EMScore"])

        return np.vstack((r, np.degrees(theta), np.degrees(phi), charge, em_score)).T

    def get_event_truth_values(self, event_record, out_file=None):

        e1, e2 = event_record["pi0_gamma_starte_initial"]
        c1, c2 = 0., 0. #event_record["true_decay_gamma_conv_dist"]
        oa = event_record["pi0_gamma_open_angle_initial"]

        ssdir = np.vstack((ak.to_numpy(event_record["pi0_gamma_startpx_initial"]),
                           ak.to_numpy(event_record["pi0_gamma_startpy_initial"]),
                           ak.to_numpy(event_record["pi0_gamma_startpz_initial"]))).T

        sdir0 = ssdir[0] / (np.sqrt(ssdir[0] @ ssdir[0]))
        sdir1 = ssdir[1] / (np.sqrt(ssdir[1] @ ssdir[1]))
        spherical_sdir0 = futil.single_to_spherical(v=sdir0, rotate_polar_axis=self.rotate_polar_axis).T
        spherical_sdir1 = futil.single_to_spherical(v=sdir1, rotate_polar_axis=self.rotate_polar_axis).T

        true_theta1, true_phi1 = np.degrees(spherical_sdir0[1:])
        true_theta2, true_phi2 = np.degrees(spherical_sdir1[1:])

        ppi0 = np.sqrt(e1**2 + e2**2 + 2. * e1 * e2 * np.cos(oa))
        cos_pi0 = (e1 + e2 * np.cos(-oa)) / ppi0

        if out_file is None:
            print("True Eπ0:", np.sum(event_record["pi0_gamma_starte_initial"]))
            print("True Gamma Eγ:", event_record["pi0_gamma_starte_initial"])
            print("True OA:", np.degrees(oa))
            print("True cos_pi0:", cos_pi0)
            print("True Gamma Energy 1/2", round(e1, 2), "/", round(e2, 2))
            print("True Gamma Theta 1/2", round(true_theta1, 2), "/", round(true_theta2, 2))
            print("True Gamma Phi 1/2", round(true_phi1, 2), "/", round(true_phi2, 2))
            print("C1/C2:", c1, "/", c2)
        else:
            print("True Eπ0:", np.sum(event_record["pi0_gamma_starte_initial"]), file=out_file)
            print("True Gamma Eγ:", event_record["pi0_gamma_starte_initial"], file=out_file)
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
