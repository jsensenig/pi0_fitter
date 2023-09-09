import numpy as np

from pi0fit.pi0_transform import Pi0Transformations
from pi0fit.pi0_model import Pi0Model
import pi0fit.fitter_utilities as futil
from pi0fit.fitter_utilities import FitResults
from pi0fit.pi0_likelihood_minimizer import Pi0MinimizerBase


class Pi0Fitter:

    def __init__(self, config):

        self.config = config["pi0_fitter"]
        self.minimizer = self.config["minimizer"]
        self.debug = self.config["debug"]

        self.pi0_transform = Pi0Transformations(config=config)
        self.pi0_model = Pi0Model(config=config)
        self.truth_comparison = self.config["truth_comparison"]
        self.use_true_transform = self.config["use_true_point_transform"]

        self.return_pi0_only = self.config["return_pi0_only"]

        self.minimizer_classes = {}
        self.minimizer_obj = self.get_minimizer(config=config)

    def get_minimizer(self, config):
        """
        Find requested minimizer and return configured object.
        :return:
        """
        minimizer_classes = {cls.__name__: cls for cls in Pi0MinimizerBase.__subclasses__()}

        if self.minimizer not in minimizer_classes.keys():
            print("Unknown Minimizer", self.minimizer, "must be one of", list(minimizer_classes.keys()))
            raise KeyError

        return minimizer_classes[self.minimizer](config)

    def fit_pi0(self, all_event_record):

        # Set the event pi0 start point
        self.pi0_transform.set_beam_endpoint(all_event_record=all_event_record)

        # Convert 3D hits to spherical coordinates
        self.pi0_transform.transform_point_to_spherical(event_record=all_event_record)

        #for evt, event_record in enumerate(all_event_record):
        for evt in range(5, 10):

            event_record = all_event_record[evt]

            # Get 3D points for event
            pi0_points = self.get_event_points(event_record=event_record, evt=evt,
                                               use_true_transform=self.use_true_transform)

            # Get event truth information
            truth_values = None
            if self.truth_comparison:
                truth_values = self.get_event_truth_values(event_record=event_record, evt=evt)

            # Minimize fit
            self.minimizer_obj.minimize(pi0_points=pi0_points, truth_values=truth_values)

            #if self.return_pi0_only:
            #    return self.minimizer_obj.epi0, self.minimizer_obj.cos_pi0
            print("Fit Result for event", evt)
            print(self.minimizer_obj.values_as_dict())
        return self.minimizer_obj.values_as_dict()

    def get_event_points(self, event_record, evt, use_true_transform=False):

        # Rotate 3D hits into the plane of the pi0 decay gammas
        if use_true_transform:
            reco_pts, _, _ = self.pi0_transform.truth_transform_to_pi0_plane(event_record=event_record, evt=evt)
        else:
            reco_pts, _, _ = self.pi0_transform.reco_transform_to_pi0_plane(event_record=event_record, evt=evt)

        return reco_pts

    def get_event_truth_values(self, event_record, evt, out_file=None):

        truth_rtp_pts, p1, p2 = self.pi0_transform.truth_transform_to_pi0_plane(event_record=event_record, evt=evt)
        truth_rtp = np.vstack((truth_rtp_pts[0, :], truth_rtp_pts[1, :], np.zeros_like(truth_rtp_pts[0, :]))).T

        e1, e2 = event_record["pi0_gamma_e_initial"]
        oa = event_record["pi0_gamma_open_angle_initial"]

        ppi0 = np.sqrt(e1**2 + e2**2 + 2. * e1 * e2 * np.cos(oa))
        cos_pi0 = (e1 + e2 * np.cos(-oa)) / ppi0

        if out_file is None:
            print("True Eπ0:", np.sum(event_record["pi0_gamma_e_initial"]))
            print("True Gamma Eγ:", event_record["pi0_gamma_e_initial"])
            print("True OA:", np.degrees(oa))
            print("True cos_pi0:", cos_pi0)
        else:
            print("True Eπ0:", np.sum(event_record["pi0_gamma_e_initial"]), file=out_file)
            print("True Gamma Eγ:", event_record["pi0_gamma_e_initial"], file=out_file)
            print("True OA:", np.degrees(oa), file=out_file)
            print("True cos_pi0:", cos_pi0, file=out_file)

        ############
        # Conversion distance

        true_theta1, true_theta2 = futil.single_to_spherical(v=p1)[2], futil.single_to_spherical(v=p2)[2]

        theta1_up, theta1_down = (true_theta1 + np.radians(5)), (true_theta1 - np.radians(5))
        theta2_up, theta2_down = (true_theta2 + np.radians(5)), (true_theta2 - np.radians(5))

        shower1_conv_mask = (truth_rtp[:, 1] < theta1_up) & (truth_rtp[:, 1] > theta1_down)
        shower2_conv_mask = (truth_rtp[:, 1] < theta2_up) & (truth_rtp[:, 1] > theta2_down)

        true_cdist1 = -100 if np.count_nonzero(shower1_conv_mask) < 1 else np.min(truth_rtp[:, 0][shower1_conv_mask])
        true_cdist2 = -100 if np.count_nonzero(shower2_conv_mask) < 1 else np.min(truth_rtp[:, 0][shower2_conv_mask])

        if self.debug:
            print("CDist:", true_cdist1, "/", true_cdist2, file=out_file)

        truth_values = FitResults(0,0,0,0,0,0,0,0,0,False)
        truth_values.set_event_values(epi0=(e1+e2), cos_pi0=cos_pi0, eg1=e1, c1=true_cdist1, c2=true_cdist2, is_truth=True)

        return truth_values
