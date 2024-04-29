import pickle
import sys

from pi0fit.pi0_fit import Pi0Fitter
from pi0fit.fitter_utilities import spherical_dot
import awkward as ak
import json
import uproot
import numpy as np


def branches(has_cosmics):
    branch = ["event", "run", "subrun", "reco_beam_startX", "reco_beam_startY", "reco_beam_startZ",
              "reco_beam_endX", "reco_beam_endY", "reco_beam_endZ", "reco_beam_calo_endX", "reco_beam_calo_endY",
              "reco_beam_calo_endZ", "reco_beam_calo_endDirX", "reco_beam_calo_endDirY", "reco_beam_calo_endDirZ"]

    branch += ["true_beam_PDG", "true_beam_endProcess", "true_beam_startP", "true_daughter_nPi0",
               "true_daughter_nPiMinus", "true_daughter_nProton", "true_daughter_nPiPlus",
               "true_beam_daughter_PDG", "true_beam_endX_SCE", "true_beam_endY_SCE", "true_beam_endZ_SCE",
               "true_beam_daughter_startP", "true_beam_daughter_startPx", "true_beam_daughter_startPy",
               "true_beam_daughter_startPz"]

    branch += ["reco_daughter_allShower_energy", "reco_daughter_allShower_dirX", "reco_daughter_allShower_dirY",
                "reco_daughter_allShower_dirZ", "reco_daughter_PFP_emScore", "true_beam_startP", "true_beam_endP"]

    branch += ["reco_daughter_allTrack_startX", "reco_daughter_allTrack_startY", "reco_daughter_allTrack_startZ"]
    branch += ["reco_daughter_allTrack_endX", "reco_daughter_allTrack_endY", "reco_daughter_allTrack_endZ"]
    branch += ["reco_daughter_allTrack_Theta", "reco_daughter_allTrack_Phi", "reco_daughter_allTrack_alt_len",
                       "reco_daughter_allTrack_calibrated_dEdX_SCE"]
    branch += ["reco_daughter_allShower_dirX", "reco_daughter_allShower_dirY", "reco_daughter_allShower_dirZ",
                       "reco_daughter_PFP_nHits"]
    branch += ["reco_daughter_allTrack_Chi2_proton", "reco_daughter_allTrack_Chi2_ndof",
                       "reco_daughter_PFP_true_byHits_PDG"]
    branch += ["reco_daughter_allTrack_Chi2_pion", "reco_daughter_allTrack_Chi2_ndof_pion"]
    branch += ["true_beam_Pi0_decay_startP", "true_beam_Pi0_decay_startPx", "true_beam_Pi0_decay_startPy",
                       "true_beam_Pi0_decay_startPz"]

    branch += ["reco_all_spacePts_X", "reco_all_spacePts_Y", "reco_all_spacePts_Z", "reco_all_spacePts_Integral"]

    ## Cosmic Stuff
    if has_cosmics:
        branch += ["cosmic_pfp_spacePts_X", "cosmic_pfp_spacePts_Y", "cosmic_pfp_spacePts_Z", "cosmic_pfp_spacePts_ID"]

        branch += ["cosmic_pfp_start_X", "cosmic_pfp_start_Y", "cosmic_pfp_start_Z", "cosmic_pfp_end_X",
                   "cosmic_pfp_end_Y", "cosmic_pfp_end_Z", "cosmic_pfp_IsPrimary", "cosmic_pfp_IsClearCosmic",
                   "cosmic_pfp_IsBeam", "cosmic_pfp_ID", "cosmic_pfp_nSpPts"]
    return branch


def calculate_open_angle(evt_record):

    oa_list = []
    for event in range(len(evt_record)):
        try:
            ss = np.vstack((ak.to_numpy(evt_record["true_beam_Pi0_decay_startPx", event]),
                            ak.to_numpy(evt_record["true_beam_Pi0_decay_startPy", event]),
                            ak.to_numpy(evt_record["true_beam_Pi0_decay_startPz", event]))).T
            oa = np.degrees(np.arccos(spherical_dot(x1=np.array([ss[0]]), x2=np.array([ss[1]]), spherical=False)))[0]
            oa_list.append(oa)
        except:
            oa_list.append(-1)

    return np.asarray(oa_list)


if __name__ == '__main__':

    run_name = sys.argv[1]

    with open("config.json", 'r') as f:
        config = json.load(f)

    print(json.dumps(config))
    pf = Pi0Fitter(config=config)

    # load events
    file_name = "/Users/jsen/work/Protodune/analysis/event_data/prod4a/new_set/1gev_files/pduneana_12.root:pduneana/beamana"

    print("Loading events!")
    all_event_record = uproot.concatenate(files={file_name}, expressions=branches(has_cosmics=True))
    print("Loaded", len(all_event_record), "events!")

    all_event_record["true_beam_Pi0_decay_OA"] = calculate_open_angle(evt_record=all_event_record)
    all_event_record["true_cex"] = ((all_event_record["true_beam_PDG"] == 211) &
                                    (all_event_record["true_beam_endProcess"] == "pi+Inelastic") &
                                    (all_event_record["true_daughter_nPi0"] == 1) &
                                    (all_event_record["true_daughter_nPiPlus"] == 0) &
                                    (all_event_record["true_daughter_nPiMinus"] == 0) &
                                    (all_event_record["true_daughter_nProton"] > 0))

    all_event_record["true_single_pi0"] = ((all_event_record["true_beam_PDG"] == 211) &
                                           (all_event_record["true_beam_endProcess"] == "pi+Inelastic") &
                                           (all_event_record["true_daughter_nPi0"] == 1))

    print("CeX/Total =", np.count_nonzero(all_event_record["true_cex"]), "/", len(all_event_record))

    print("nEvts:", len(all_event_record))
    num_events, fit_results_list, truth_list = pf.fit_pi0(all_event_record=all_event_record[all_event_record["true_cex"]])

    results_dict = {"number_events": num_events, "fit_results": fit_results_list, "truth_list": truth_list}
    results_file = run_name + '.pickle'
    print("Writing results to file:", results_file)
    with open(results_file, 'wb') as f:
        pickle.dump(results_dict, f)
