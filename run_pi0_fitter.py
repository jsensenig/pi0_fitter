from pi0fit.pi0_fit import Pi0Fitter
import awkward as ak
import json
import uproot


if __name__ == '__main__':

    with open("config.json", 'r') as f:
        config = json.load(f)

    print(json.dumps(config))
    pf = Pi0Fitter(config=config)

    # load events
    file_name = "/Users/jsen/tmp/tmp_pi0_shower/sample/single_pi0/single_pi0_uniform_0_2gev_n4550.root:trkUtil/points"
    #file_name = "/home/hep/work/dune/pion_cex_ana/pi0_fitting/data/single_pi0_uniform_0_2gev_n4550.root:trkUtil/points"

    branches = ["reco_all_SpHit_X", "reco_all_SpHit_Y", "reco_all_SpHit_Z",
                "pi0_gamma_px_initial", "pi0_gamma_py_initial", "pi0_gamma_pz_initial", "pi0_gamma_e_initial",
                "pi0_gamma_open_angle_initial"]

    total_events = uproot.concatenate(files={file_name}, expressions=branches)

    tmp_zeros = ak.zeros_like(total_events["reco_all_SpHit_X"])
    total_events["reco_daughter_PFP_shower_spacePts_R"] = tmp_zeros
    total_events["reco_daughter_PFP_shower_spacePts_Theta"] = tmp_zeros
    total_events["reco_daughter_PFP_shower_spacePts_Phi"] = tmp_zeros

    print("nEvts:", len(total_events))
    pf.fit_pi0(all_event_record=total_events)
