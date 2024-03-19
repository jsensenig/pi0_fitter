from pi0fit.pi0_fit import Pi0Fitter
import awkward as ak
import json
import uproot
import numpy as np


def add_true_conv_dist(evt_record):
    conv_mask = evt_record["pi0_gamma_electron_process_initial"] == "conv"
    min_dist_x = ak.min(
        abs(evt_record["pi0_startx_initial"] - evt_record["pi0_gamma_electron_startx_initial"][conv_mask]), axis=2)
    min_dist_y = ak.min(
        abs(evt_record["pi0_starty_initial"] - evt_record["pi0_gamma_electron_starty_initial"][conv_mask]), axis=2)
    min_dist_z = ak.min(
        abs(evt_record["pi0_startz_initial"] - evt_record["pi0_gamma_electron_startz_initial"][conv_mask]), axis=2)

    return np.sqrt(min_dist_x*min_dist_x + min_dist_y*min_dist_y + min_dist_z*min_dist_z)


if __name__ == '__main__':

    with open("config.json", 'r') as f:
        config = json.load(f)

    print(json.dumps(config))
    pf = Pi0Fitter(config=config)

    # load events
    #file_name = "/Users/jsen/tmp/tmp_pi0_shower/sample/single_pi0/single_pi0_uniform_0_2gev_n4550.root:trkUtil/points"
    #file_name = "/home/hep/work/dune/pion_cex_ana/pi0_fitting/data/single_pi0_uniform_0_2gev_n4550.root:trkUtil/points"

    file_name = "/Users/jsen/tmp/tmp_pi0_shower/sample/single_pi0/single_pi0_updated_tree_uniform_0_2gev_n4550.root:trkUtil/points"

    # branches = ["reco_all_SpHit_X", "reco_all_SpHit_Y", "reco_all_SpHit_Z", "reco_all_SpHit_sadc",
    #             "pi0_gamma_px_initial", "pi0_gamma_py_initial", "pi0_gamma_pz_initial", "pi0_gamma_e_initial",
    #             "pi0_gamma_open_angle_initial"]
    #
    # total_events = uproot.concatenate(files={file_name}, expressions=branches)
    #
    # tmp_zeros = ak.zeros_like(total_events["reco_all_SpHit_X"])
    # total_events["reco_daughter_PFP_shower_spacePts_R"] = tmp_zeros
    # total_events["reco_daughter_PFP_shower_spacePts_Theta"] = tmp_zeros
    # total_events["reco_daughter_PFP_shower_spacePts_Phi"] = tmp_zeros
    #
    # print("nEvts:", len(total_events))

    # Space Points
    branches = ["reco_all_spacePts_X", "reco_all_spacePts_Y", "reco_all_spacePts_Z", "reco_all_spacePts_Integral"]

    # Pi0
    branches += ["pi0_pdg_initial", "pi0_trackid_initial", "pi0_startpx_initial", "pi0_startpy_initial",
                 "pi0_startpz_initial",
                 "pi0_startx_initial", "pi0_starty_initial", "pi0_startz_initial", "pi0_starte_initial",
                 "pi0_gamma_open_angle_initial"]

    # Pi0 Decay Gammas
    branches += ["pi0_gamma_startx_initial", "pi0_gamma_starty_initial", "pi0_gamma_startz_initial",
                 "pi0_gamma_endx_initial", "pi0_gamma_endy_initial",
                 "pi0_gamma_endz_initial", "pi0_gamma_startpx_initial", "pi0_gamma_startpy_initial",
                 "pi0_gamma_startpz_initial", "pi0_gamma_starte_initial",
                 "pi0_gamma_pdg_initial", "pi0_gamma_motherid_initial", "pi0_gamma_trackid_initial",
                 "pi0_gamma_process_initial", "pi0_gamma_endprocess_initial"]

    # Pi0 Decay Gamma Pair e-e+ or Compton electron
    branches += ["pi0_gamma_electron_startx_initial", "pi0_gamma_electron_starty_initial",
                 "pi0_gamma_electron_startz_initial",
                 "pi0_gamma_electron_startpx_initial", "pi0_gamma_electron_startpy_initial",
                 "pi0_gamma_electron_startpz_initial",
                 "pi0_gamma_electron_starte_initial", "pi0_gamma_electron_pdg_initial",
                 "pi0_gamma_electron_motherid_initial",
                 "pi0_gamma_electron_trackid_initial", "pi0_gamma_electron_process_initial"]

    #raw_event_record = uproot.concatenate(files={file_name}, expressions=branches)
    all_event_record = uproot.concatenate(files={file_name}, expressions=branches)

    all_event_record["true_decay_gamma_conv_dist"] = add_true_conv_dist(evt_record=all_event_record)
    # Only consider decays with 2 gammas
    #two_gamma_decay_mask = np.count_nonzero(raw_event_record["pi0_gamma_process_initial"] == "Decay", axis=1) == 2
    #all_event_record = raw_event_record[two_gamma_decay_mask]

    tmp_zeros = ak.zeros_like(all_event_record["reco_all_spacePts_X"])
    all_event_record["reco_daughter_PFP_shower_spacePts_R"] = tmp_zeros
    all_event_record["reco_daughter_PFP_shower_spacePts_Theta"] = tmp_zeros
    all_event_record["reco_daughter_PFP_shower_spacePts_Phi"] = tmp_zeros

    print("nEvts:", len(all_event_record))
    pf.fit_pi0(all_event_record=all_event_record)
