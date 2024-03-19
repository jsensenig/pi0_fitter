from sklearn.neighbors import KDTree
import numpy as np
import awkward as ak
import pi0fit.fitter_utilities as futil


class CleanEvent:

    def __init__(self, config):
        self.config = config["clean_event"]

        self.use_true_vertex = self.config["use_true_vertex"]
        self.has_cosmics = self.config["has_cosmics"]
        self.beam_cos_theta_cut = self.config["beam_cos_theta_cut"]
        self.mean_beam_theta = self.config["mean_beam_theta"]
        self.mean_beam_phi = self.config["mean_beam_phi"]
        self.charge_point_cut = self.config["charge_point_cut"]
        self.rcut_low = self.config["rcut_low"]
        self.rcut_high = self.config["rcut_high"]
        self.daughter_rstart_cut = self.config["daughter_rstart_cut"]
        self.proton_chi2_cut = self.config["proton_chi2_cut"]
        self.daughter_nhit_cut = self.config["daughter_nhit_cut"]

        self.fiducial_xmin = self.config["fiducial_xmin"]
        self.fiducial_xmax = self.config["fiducial_xmax"]
        self.fiducial_ymin = self.config["fiducial_ymin"]
        self.fiducial_ymax = self.config["fiducial_ymax"]
        self.fiducial_zmin = self.config["fiducial_zmin"]
        self.fiducial_zmax = self.config["fiducial_zmax"]

    def clean_event(self, spherical_pts, cartesian_pts, charge_pts, cosmic_pts, xyz_vertex):
        """
        Call proton removeal after this
        :param spherical_pts:
        :param cartesian_pts:
        :param charge_pts:
        :param cosmic_pts:
        :param xyz_vertex:
        :return:
        """
        pts_fiducial_mask, cosmic_fiducial_mask = self.fiducial_cut(cartesian_pts=cartesian_pts, cosmic_pts=cosmic_pts,
                                                                    xyz_vertex=xyz_vertex)
        radius_cut = spherical_pts[:, 0] < self.rcut_high

        if self.has_cosmics:
            cosmic_removed_mask = self.remove_cosmics_from_pfp(cartesian_pts=cartesian_pts[pts_fiducial_mask & radius_cut],
                                                               cosmic_pts=cosmic_pts[cosmic_fiducial_mask])
            spherical_pts = spherical_pts[pts_fiducial_mask & radius_cut][cosmic_removed_mask]
            charge_pts = charge_pts[pts_fiducial_mask & radius_cut][cosmic_removed_mask]

        accum_mask = spherical_pts[:, 0] > self.rcut_low
        accum_mask &= self.beam_cut(spherical_pts=spherical_pts)
        accum_mask &= charge_pts < self.charge_point_cut

        return spherical_pts[accum_mask], charge_pts[accum_mask]

    def fiducial_cut(self, cartesian_pts, cosmic_pts, xyz_vertex):

        xdown, xup = (-300 - xyz_vertex[0]), (50 - xyz_vertex[0])
        ydown, yup = (0 - 300), 100
        zdown, zup = (0 - xyz_vertex[2] if xyz_vertex[2] < 100 else -100.), (150 + xyz_vertex[2])

        fiducial_cut_mask = ((cartesian_pts[:, 0] > xdown) & (cartesian_pts[:, 0] < xup) &
                             (cartesian_pts[:, 2] > zdown) & (cartesian_pts[:, 2] < zup))

        cosmic_fiducial_cut_mask = None
        if self.has_cosmics:
            cosmic_fiducial_cut_mask = ((cosmic_pts[:, 0] > xdown) & (cosmic_pts[:, 0] < xup) &
                                        (cosmic_pts[:, 2] > zdown) & (cosmic_pts[:, 2] < zup))

        return fiducial_cut_mask, cosmic_fiducial_cut_mask

    def beam_cut(self, spherical_pts):

        the_one = np.ones(len(spherical_pts[:, 1]))
        pts = np.vstack((the_one, np.radians(spherical_pts[:, 1]), np.radians(spherical_pts[:, 2]))).T
        beam_center = np.vstack((the_one,
                                 the_one * np.radians(self.mean_beam_theta),
                                 the_one * np.radians(self.mean_beam_phi))).T

        beam_point_cos = futil.spherical_dot(beam_center, pts)

        return ~(beam_point_cos > self.beam_cos_theta_cut)

    def remove_cosmics_from_pfp(self, cartesian_pts, cosmic_pts):

        # Only keep cosmic points which have >3 neighbors
        tree = KDTree(cosmic_pts)
        tmp_masked_pts = tree.query_radius(cosmic_pts, r=2.0)
        tmp_cosmic_remove_mask = (np.asarray([a.size > 3 for a in tmp_masked_pts]))

        # Now remove any event point within 2cm of a cosmic point
        tree = KDTree(cosmic_pts[tmp_cosmic_remove_mask])
        masked_pts = tree.query_radius(cartesian_pts, r=1.0)

        cosmic_removed_mask = ~(np.asarray([a.size > 0 for a in masked_pts]))

        return cosmic_removed_mask

    def is_outside_fiducial_box(self, xpoint, ypoint, zpoint):
        """
        ProtoDUNE Performance Paper: https://cds.cern.ch/record/2724554/files/Abi_2020_J._Inst._15_P12004.pdf
        Active volume (x2): 6.086 (h) × 3.597 (w) × 7.045 (l) m^3
        """
        return (xpoint <= self.fiducial_xmin) | (xpoint >= self.fiducial_xmax) | \
               (ypoint <= self.fiducial_ymin) | (ypoint >= self.fiducial_ymax) | \
               (zpoint <= self.fiducial_zmin) | (zpoint >= self.fiducial_zmax)

    def cosmic_selection(self, event_record, evt, hit_cut=200):

        hit_mask = event_record["cosmic_pfp_nSpPts", evt] > hit_cut

        startx = event_record["cosmic_pfp_start_X", evt][hit_mask]
        starty = event_record["cosmic_pfp_start_Y", evt][hit_mask]
        startz = event_record["cosmic_pfp_start_Z", evt][hit_mask]
        endx = event_record["cosmic_pfp_end_X", evt][hit_mask]
        endy = event_record["cosmic_pfp_end_Y", evt][hit_mask]
        endz = event_record["cosmic_pfp_end_Z", evt][hit_mask]

        is_outside_list = []
        for sx, sy, sz, ex, ey, ez in zip(startx, starty, startz, endx, endy, endz):
            is_outside_list.append(self.is_outside_fiducial_box(xpoint=sx, ypoint=sy, zpoint=sz) or
                                   self.is_outside_fiducial_box(xpoint=ex, ypoint=ey, zpoint=ez))

        cosmic_id_mask = ak.to_numpy(np.zeros_like(event_record["cosmic_pfp_spacePts_ID", evt], dtype=bool))
        print("Cosmic Pre", len(cosmic_id_mask))

        for cid, beam, prim, cosmic, hcut, out in zip(event_record["cosmic_pfp_ID", evt],
                                                      event_record["cosmic_pfp_IsBeam", evt],
                                                      event_record["cosmic_pfp_IsPrimary", evt],
                                                      event_record["cosmic_pfp_IsClearCosmic", evt], hit_mask,
                                                      is_outside_list):
            if not beam and ((hcut and out and prim) or (hcut and cosmic)):
                mask = event_record["cosmic_pfp_spacePts_ID", evt] == cid
                cosmic_id_mask[mask] = True

        print("Cosmic Post (remaining):", np.count_nonzero(~cosmic_id_mask))

        return cosmic_id_mask

    def proton_cut(self, event_record, spherical_pts, event, xyz_shift):

        dx = event_record["reco_daughter_allTrack_startX", event] - xyz_shift[0]
        dy = event_record["reco_daughter_allTrack_startY", event] - xyz_shift[1]
        dz = event_record["reco_daughter_allTrack_startZ", event] - xyz_shift[2]
        dradius = np.sqrt(dx * dx + dy * dy + dz * dz)

        daughter_pdg = event_record["reco_daughter_PFP_true_byHits_PDG", event]
        daughter_theta = np.degrees(event_record["reco_daughter_allTrack_Theta", event])
        daughter_phi = np.degrees(event_record["reco_daughter_allTrack_Phi", event])
        daughter_nhit = event_record["reco_daughter_PFP_nHits", event]
        daughter_proton_chi2 = event_record["reco_daughter_allTrack_Chi2_proton", event] / \
                               event_record["reco_daughter_allTrack_Chi2_ndof", event]

        proton_remove_mask = np.ones(shape=(len(spherical_pts)), dtype=bool)
        the_one = np.ones(len(spherical_pts[:, 1]))
        pts = np.vstack((the_one, np.radians(spherical_pts[:, 1]), np.radians(spherical_pts[:, 2]))).T

        # Find all shower-like particles
        n = 0
        shower_dict = {}
        for theta, phi, nhits, chi2_proton, dr in zip(daughter_theta, daughter_phi, daughter_nhit,
                                                      daughter_proton_chi2, dradius):
            valid_nhit = nhits > self.daughter_nhit_cut
            valid_distance = (dr > self.daughter_rstart_cut) and (dr < 90.)
            very_valid_distance = (dr > 15) and (dr < 90.)
            valid_chi2 = chi2_proton > self.proton_chi2_cut + 15
            if (valid_nhit and valid_distance and valid_chi2) or (valid_distance and chi2_proton > 200) or \
                (valid_nhit and very_valid_distance):
                shower_dict[n] = np.array([[1., np.radians(theta), np.radians(phi)]])
            n += 1
        print("nShower-like Particles:", len(shower_dict), "Selected:", shower_dict.keys())

        n = 0
        removed_protons = []
        for theta, phi, nhits, chi2_proton, dr, dpdg in zip(daughter_theta, daughter_phi, daughter_nhit,
                                                            daughter_proton_chi2, dradius, daughter_pdg):
            too_close_to_shower = False
            print(n, ")  theta/phi/nhit/Pchi2/dr", int(theta), "/", int(phi), "/", nhits, "/",
                  np.round(chi2_proton, 2), "/", np.round(dr, 2), " [", dpdg, "]")
            valid_nhit = (nhits > self.daughter_nhit_cut) and (nhits < 900)
            valid_distance = dr < 1.
            valid_chi2 = (chi2_proton > -100) and (chi2_proton != 1) and (chi2_proton < self.proton_chi2_cut)
            loose_chi2 = (chi2_proton > -100) and (chi2_proton != 1) and (chi2_proton < 200.)
            charged_daughter = (chi2_proton < 50.) and (chi2_proton != 1.0)
            if (valid_nhit and valid_chi2) or charged_daughter or (loose_chi2 and valid_distance and valid_nhit):
                proton_dir = np.array([[1., np.radians(theta), np.radians(phi)]])
                for shower in shower_dict:
                    too_close_to_shower = futil.spherical_dot(proton_dir, shower_dict[shower]) > 0.9
                    if too_close_to_shower: break
                if too_close_to_shower:
                    n += 1
                    continue
                proton_pts = np.vstack((the_one, the_one * np.radians(theta), the_one * np.radians(phi))).T
                proton_point_cos = futil.spherical_dot(proton_pts, pts)
                proton_remove_mask &= ~(proton_point_cos > 0.85)
                removed_protons.append(n)
            n += 1
        print("Removed Protons:", removed_protons)

        return proton_remove_mask
