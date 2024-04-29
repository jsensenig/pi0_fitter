import pickle
import sys
import os
import concurrent.futures
import time
import numpy as np

from pi0fit.pi0_fit import Pi0Fitter
import json
import uproot


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


def check_thread_count(threads):
    if threads > os.cpu_count():
        print("Requested", threads, "threads but only", os.cpu_count(), "available!")
        print("Setting number of threads to", os.cpu_count())
        return os.cpu_count()
    return threads


def fitter_wrapper(configuration, event_record):
    fitter_instance = Pi0Fitter(config=configuration)
    time.sleep(0.1)
    return fitter_instance.fit_pi0(all_event_record=event_record, pi0_points=None, loop_events=True)


def thread_creator(flist, config, results_file, num_workers):

    branch = branches(has_cosmics=True)
    threads = check_thread_count(threads=num_workers)

    # Context manager handles joining of the threads
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        # Use iterations of the tree read operation to batch the data for each thread
        for i, array in enumerate(uproot.iterate(files=flist, expressions=branch, report=True, step_size='10000 MB',
                                                 num_workers=threads)):
            print("---------- Starting thread", i, "----------")
            futures.append(executor.submit(fitter_wrapper, config, array[0]))
            print(array[1])  # The report part of the array tuple from the tree iterator
            time.sleep(0.2)

        save_results(thread_results=concurrent.futures.as_completed(futures), results_file=results_file)


def save_results(thread_results, results_file):
    # Wait for the threads to complete
    results_list = [future.result() for future in thread_results]

    thread_num_event_list = [thread[0] for thread in results_list]
    thread_fit_results_list = [thread[1] for thread in results_list]
    thread_truth_list = [thread[2] for thread in results_list]

    print("Result Number Events:", thread_num_event_list)

    eff_list = []
    fit_list = []
    comp_list = []
    truth_list = []
    for total_nevt, fit, truth in zip(thread_num_event_list, thread_fit_results_list, thread_truth_list):
        for fvals, tvals in zip(fit, truth):
            fit_list.append(fvals[1])
            comp_list.append(fvals[2])
            truth_list.append(tvals[1])
        eff_list.append(len(fit) / total_nevt)

    results_dict = {"efficiency": eff_list, "fit_results": fit_list, "truth_list": truth_list, "comparison": comp_list}

    print("Writing results to file:", results_file)
    with open(results_file, 'wb') as f:
        pickle.dump(results_dict, f)


if __name__ == '__main__':

    run_name = sys.argv[1]
    results_file = run_name + '.pickle'

    with open("config.json", 'r') as f:
        config = json.load(f)

    print(json.dumps(config))

    #file_list = ["/Users/jsen/work/Protodune/analysis/event_data/prod4a/new_set/1gev_files/pduneana_12.root:pduneana/beamana",
    #             "/Users/jsen/work/Protodune/analysis/event_data/prod4a/new_set/1gev_files/pduneana_13.root:pduneana/beamana"
    #             ]

    file_list = ["/home/jon/work/protodune/analysis/pi0_reco/data/1gev_ana_files/subset/pduneana_10.root:pduneana/beamana"]

   # file_list = ["/home/jon/work/protodune/analysis/pi0_reco/data/1gev_ana_files/subset/pduneana_10.root:pduneana/beamana",
   #              "/home/jon/work/protodune/analysis/pi0_reco/data/1gev_ana_files/subset/pduneana_11.root:pduneana/beamana",
   #              "/home/jon/work/protodune/analysis/pi0_reco/data/1gev_ana_files/subset/pduneana_12.root:pduneana/beamana",
   #              "/home/jon/work/protodune/analysis/pi0_reco/data/1gev_ana_files/subset/pduneana_13.root:pduneana/beamana",
   #              "/home/jon/work/protodune/analysis/pi0_reco/data/1gev_ana_files/subset/pduneana_14.root:pduneana/beamana",
   #              "/home/jon/work/protodune/analysis/pi0_reco/data/1gev_ana_files/subset/pduneana_15.root:pduneana/beamana",
   #              "/home/jon/work/protodune/analysis/pi0_reco/data/1gev_ana_files/subset/pduneana_16.root:pduneana/beamana",
   #              "/home/jon/work/protodune/analysis/pi0_reco/data/1gev_ana_files/subset/pduneana_17.root:pduneana/beamana"
   #              ]


    try:
        # Dispatch threads
        thread_creator(flist=file_list, config=config, results_file=results_file, num_workers=2)
    except KeyboardInterrupt:
        os._exit(1)

