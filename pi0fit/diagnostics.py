import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import pi0fit.fitter_utilities as futil
import pi0fit.pi0_fit as pi0fit
from pi0fit.pi0_model import BinnedPi0Model
from pi0fit.clean_event import CleanEvent


class Diagnostics:
    def __init__(self, config):
        self.config = config
        self.pi0_fit = pi0fit.Pi0Fitter(config=config)
        self.pi0_model = BinnedPi0Model(config=config)
        self.clean_event = CleanEvent(config=config)
        self._cut_list = self.clean_event.cut_list

    @property
    def cut_list(self):
        return self._cut_list

    def plot_cut_steps(self, event_record, event, cut_name, cosmic_nhit, show_gammas, show_daughters, pre_hist_cut, plot_axes='yz'):

        xyz_vertex, dr = self.pi0_fit.get_vertex(event_record=event_record, event=event)
        print('Vertex xyz:', xyz_vertex)

        cartesian_pts, cosmic_pts = self.pi0_fit.get_event_points(event_record=event_record, event=event,
                                                                  return_spherical=False, cosmics=True, get_gammas=False)
        spherical_pts = self.pi0_fit.get_event_points(event_record=event_record, event=event,
                                                      return_spherical=True, cosmics=False, get_gammas=False)

        valid_cosmic_mask = self.clean_event.cosmic_selection(event_record=event_record, evt=event, hit_cut=cosmic_nhit)

        ###############################
        self.clean_event.cut_name = cut_name
        tmp_cleaned_pts = self.clean_event.clean_event(spherical_pts=spherical_pts, cartesian_pts=cartesian_pts,
                                     cosmic_pts=cosmic_pts[valid_cosmic_mask], xyz_vertex=xyz_vertex)
        cleaned_spherical_pts = \
            self.clean_event.cut_points if cut_name != "proton_cut" and cut_name != "hist_cut" else tmp_cleaned_pts

        if cut_name == "proton_cut":
            no_proton_mask = self.clean_event.proton_cut(event_record=event_record, spherical_pts=cleaned_spherical_pts,
                                                         event=event, xyz_shift=xyz_vertex)
            cleaned_spherical_pts = cleaned_spherical_pts[no_proton_mask]

        if len(cleaned_spherical_pts) < 1:
            print("No points survived cuts!")
            return

        precut = cut_name != "hist_cut"
        charge_hist, dir_hist = self.pi0_model.construct_event_hists(pi0_pts=cleaned_spherical_pts, return_precut=pre_hist_cut)
        total_energy = self.pi0_model.calo_to_energy(charge=np.sum(charge_hist))
        print("Total Energy:", total_energy)

        _, (ax1) = plt.subplots(1, 1, figsize=(9, 7))
        self.plot_histogram(event_record=event_record, event=event, ax=ax1, hist=dir_hist,
                            dir_bins=self.pi0_model.direction_dict['bins'], show_gammas=show_gammas,
                            show_daughters=show_daughters, plot_axes=plot_axes)

        return total_energy

    def show_pi0_event(self, event_record, event, show_daughters=False, show_gammas=True, plot_axes='yz', return_precut=False):

        xyz_vertex,_ = self.pi0_fit.get_vertex(event_record=event_record, event=event)

        cartesian_pts, cosmic_pts = self.pi0_fit.get_event_points(event_record=event_record, event=event,
                                                                  return_spherical=False, cosmics=True, get_gammas=False)

        preclean_spherical_pts = self.pi0_fit.get_event_points(event_record=event_record, event=event,
                                                               return_spherical=True, cosmics=False, get_gammas=False)

        valid_cosmic_mask = self.clean_event.cosmic_selection(event_record=event_record, evt=event, hit_cut=200)
        cleaned_spherical_pts = self.clean_event.clean_event(spherical_pts=preclean_spherical_pts,
                                                                             cartesian_pts=cartesian_pts,
                                                                             cosmic_pts=cosmic_pts[valid_cosmic_mask],
                                                                             xyz_vertex=xyz_vertex)

        if cleaned_spherical_pts is None:
            print("No points survived cuts!")
            return

        no_proton_mask = self.clean_event.proton_cut(event_record=event_record, spherical_pts=cleaned_spherical_pts,
                                                     event=event, xyz_shift=xyz_vertex)

        if np.count_nonzero(no_proton_mask) < 1:
            print("No points survived cuts!")
            return

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Pre-cut histogram
        _, dir_hist = self.pi0_model.construct_event_hists(pi0_pts=preclean_spherical_pts, return_precut=True)
        self.plot_histogram(event_record=event_record, event=event, ax=ax1, hist=dir_hist,
                            dir_bins=self.pi0_model.direction_dict['bins'], show_gammas=show_gammas,
                            show_daughters=show_daughters, plot_axes=plot_axes)

        # Post-cut histogram
        _, dir_hist = self.pi0_model.construct_event_hists(pi0_pts=cleaned_spherical_pts[no_proton_mask], return_precut=return_precut)
        self.plot_histogram(event_record=event_record, event=event, ax=ax2, hist=dir_hist,
                            dir_bins=self.pi0_model.direction_dict['bins'], show_gammas=show_gammas,
                            show_daughters=show_daughters, plot_axes=plot_axes)

        plt.show()

    def plot_histogram(self, event_record, event, ax, hist, dir_bins, show_gammas, show_daughters, plot_axes):

        if plot_axes == 'xy':
            sum_axis = 2
            xlim_down, xlim_up = 0, 200
            ylim_down, ylim_up = 0, 180
            bx, by = np.meshgrid(dir_bins[0], dir_bins[1], indexing='ij')
        elif plot_axes == 'xz':
            sum_axis = 1
            xlim_down, xlim_up = 0, 200
            ylim_down, ylim_up = -180, 180
            bx, by = np.meshgrid(dir_bins[0], dir_bins[2], indexing='ij')
        elif plot_axes == 'yz':
            sum_axis = 0
            xlim_down, xlim_up = 0, 180
            ylim_down, ylim_up = -180, 180
            bx, by = np.meshgrid(dir_bins[1], dir_bins[2], indexing='ij')
        else:
            print('Unknown axis!', plot_axes)

        sdir1, sdir2 = None, None
        if show_gammas:
            _, sdir1, sdir2 = self.pi0_fit.get_event_points(event_record=event_record, event=event, return_spherical=True,
                                                            cosmics=False, get_gammas=show_gammas)

        daughters, momentum = None, None
        if show_daughters:
            daughters, momentum = self.get_pesky_daughters(event_record=event_record[event])

        f = ax.scatter(bx, by, c=np.sum(hist, axis=sum_axis), marker='s', s=50, cmap=plt.cm.jet, norm=LogNorm(vmax=5e3, vmin=5))
        if show_gammas:
            ax.plot(np.degrees(sdir1[1]), np.degrees(sdir1[2]), marker='*', markersize=12, color='magenta')
            ax.plot(np.degrees(sdir2[1]), np.degrees(sdir2[2]), marker='*', markersize=12, color='magenta')
        if show_daughters:
            for d, p in zip(daughters, momentum):
                ax.plot(d[1], d[2], marker='*', markersize=12, color='orange')
                ax.text(d[1], d[2], str(int(p)), color="red", fontsize=10)
        ax.set_xlim(xlim_down, xlim_up)
        ax.set_ylim(ylim_down, ylim_up)
        plt.colorbar(f)

    @staticmethod
    def get_pesky_daughters(event_record):

        true_daughter_pdg = event_record["true_beam_daughter_PDG"]
        valid_daughter = ((true_daughter_pdg < 2000) | (true_daughter_pdg == 2212)) & (true_daughter_pdg != 111) & \
                         (event_record["true_beam_daughter_startP"] > 0.01)

        daughter_dir = np.vstack(
            (ak.to_numpy(event_record["true_beam_daughter_startPx"][valid_daughter]),
             ak.to_numpy(event_record["true_beam_daughter_startPy"][valid_daughter]),
             ak.to_numpy(event_record["true_beam_daughter_startPz"][valid_daughter]))).T

        daughter_list = []
        for daughter in daughter_dir:
            norm_dir = daughter / (np.sqrt(daughter @ daughter))
            spherical_dir = futil.single_to_spherical(v=norm_dir, rotate_polar_axis=False).T
            daughter_list.append(np.degrees(spherical_dir))

        return daughter_list, event_record["true_beam_daughter_startP"][valid_daughter] * 1000.

    @staticmethod
    def truth_reco_comparison(dir_list, elist, sdir0, sdir1, print_stuff):

        elist = np.asarray(elist)
        dir_list = np.asarray(dir_list)
        if len(dir_list) > 2:  # get 2 largest energy showers
            reco_e1, reco_e2 = np.asarray(elist[elist.argsort()][-2:])
            reco_xyz0, reco_xyz1 = np.asarray(dir_list[elist.argsort()][-2:])
        elif len(dir_list) == 2:
            reco_xyz0, reco_xyz1 = np.asarray(dir_list)
            reco_e1, reco_e2 = np.asarray(elist)
        else:
            return [-5, -5]

        reco_sdir0 = reco_xyz0 / (np.sqrt(reco_xyz0 @ reco_xyz0))
        reco_sdir1 = reco_xyz1 / (np.sqrt(reco_xyz1 @ reco_xyz1))

        spherical_sdir0 = futil.single_to_spherical(v=reco_sdir0, rotate_polar_axis=False).T
        spherical_sdir1 = futil.single_to_spherical(v=reco_sdir1, rotate_polar_axis=False).T

        reco_a1, reco_a2 = np.degrees(spherical_sdir0[1]), np.degrees(spherical_sdir1[1])
        reco_p1, reco_p2 = np.degrees(spherical_sdir0[2]), np.degrees(spherical_sdir1[2])

        true_a1, true_a2 = np.degrees(sdir0[1]), np.degrees(sdir1[1])
        true_p1, true_p2 = np.degrees(sdir0[2]), np.degrees(sdir1[2])

        angle_check1 = futil.spherical_dot(np.array([[1, np.radians(reco_a1), np.radians(reco_p1)]]),
                                           np.array([[1, np.radians(true_a1), np.radians(true_p1)]]))
        angle_check2 = futil.spherical_dot(np.array([[1, np.radians(reco_a2), np.radians(reco_p2)]]),
                                           np.array([[1, np.radians(true_a1), np.radians(true_p1)]]))

        angle_check1_s2 = futil.spherical_dot(np.array([[1, np.radians(reco_a1), np.radians(reco_p1)]]),
                                              np.array([[1, np.radians(true_a2), np.radians(true_p2)]]))
        angle_check2_s2 = futil.spherical_dot(np.array([[1, np.radians(reco_a2), np.radians(reco_p2)]]),
                                              np.array([[1, np.radians(true_a2), np.radians(true_p2)]]))
        # print(angle_check1,"/",angle_check2)
        flipped = (angle_check2 > angle_check1) & (angle_check2_s2 < angle_check1_s2)

        if flipped:
            reco_e2, reco_e1 = reco_e1, reco_e2
            reco_a2, reco_a1 = reco_a1, reco_a2
            reco_p2, reco_p1 = reco_p1, reco_p2

        # print("Flipped?", flipped)

        open_angle = np.arccos(futil.spherical_dot(np.array([[1, np.radians(reco_a1), np.radians(reco_p1)]]),
                                                   np.array([[1, np.radians(reco_a2), np.radians(reco_p2)]])))
        if print_stuff: print("OA", np.degrees(open_angle))

        cos_true_reco1 = futil.spherical_dot(np.array([[1, np.radians(reco_a1), np.radians(reco_p1)]]),
                                             np.array([[1, np.radians(true_a1), np.radians(true_p1)]]))[0]
        cos_true_reco2 = futil.spherical_dot(np.array([[1, np.radians(reco_a2), np.radians(reco_p2)]]),
                                             np.array([[1, np.radians(true_a2), np.radians(true_p2)]]))[0]
        if print_stuff:
            print("Cosθ", np.round(cos_true_reco1, 2), "/", np.round(cos_true_reco2, 2), " -- ", np.round(reco_e1, 2), "/",
                  np.round(reco_e2, 2))

        alt_cos_true_reco1 = futil.spherical_dot(np.array([[1, np.radians(reco_a2), np.radians(reco_p2)]]),
                                                 np.array([[1, np.radians(true_a1), np.radians(true_p1)]]))[0]
        alt_cos_true_reco2 = futil.spherical_dot(np.array([[1, np.radians(reco_a1), np.radians(reco_p1)]]),
                                                 np.array([[1, np.radians(true_a2), np.radians(true_p2)]]))[0]
        if print_stuff: print("AltCosθ", np.round(alt_cos_true_reco1, 2), "/", np.round(alt_cos_true_reco2, 2))

        return [cos_true_reco1, cos_true_reco2]

    def compare_to_pandora(self, event_record, event_list, nhit_cut=20, make_plots=False, print_stuff=False):
        num_gamma_list = []
        energy_list = []
        true_energy_list = []
        direction_res_list = []
        for event in event_list:
            # print("--> Evt:", event, " | ", event_record["true_beam_Pi0_decay_startP", event] * 1.e3, "=",
            #       np.sum(event_record["true_beam_Pi0_decay_startP", event] * 1.e3))

            _, sdir0, sdir1 = self.pi0_fit.get_event_points(event_record=event_record, event=event,
                                                                  return_spherical=True, cosmics=False)
            sdx = event_record["reco_daughter_allShower_dirX", event]
            sdy = event_record["reco_daughter_allShower_dirY", event]
            sdz = event_record["reco_daughter_allShower_dirZ", event]

            daughter_pdg = event_record["reco_daughter_PFP_true_byHits_PDG", event]
            daughter_nhit = event_record["reco_daughter_PFP_nHits", event]
            daughter_energy = event_record["reco_daughter_allShower_energy", event]
            true_energy_list.append(np.sum(event_record["true_beam_Pi0_decay_startP", event] * 1.e3))

            num_gamma, total_energy = 0, 0
            dir_list = []
            evt_energy_list = []
            for pdg, nhits, energy, x, y, z in zip(daughter_pdg, daughter_nhit, daughter_energy, sdx, sdy, sdz):
                if pdg == 22 and nhits > nhit_cut:
                    num_gamma += 1
                    total_energy += energy if energy > 0 else 0.
                    dir_list.append([x, y, z])
                    evt_energy_list.append(energy)

            if print_stuff: print("Total Energy:", total_energy)
            dir_res = self.truth_reco_comparison(dir_list=dir_list, elist=evt_energy_list, sdir0=sdir0, 
                                                 sdir1=sdir1, print_stuff=print_stuff)

            direction_res_list.append(dir_res)
            num_gamma_list.append(num_gamma)
            energy_list.append(total_energy)

        num_gamma_list = np.asarray(num_gamma_list)
        energy_list = np.asarray(energy_list)
        true_energy_list = np.asarray(true_energy_list)

        if make_plots:
            self.make_comparison_plots(energy_list=energy_list, true_energy_list=true_energy_list,
                                       num_gamma_list=num_gamma_list, direction_res_list=direction_res_list,
                                       nevents=len(event_list))
        else:
            return energy_list, true_energy_list, num_gamma_list, direction_res_list

    @staticmethod
    def make_comparison_plots(energy_list, true_energy_list, num_gamma_list, direction_res_list, nevents):
        comp_dir_list = np.asarray(direction_res_list)
        correct_showers = (comp_dir_list[:, 0] > 0.9) & (comp_dir_list[:, 1] > 0.9)
        print("Correct Shower / Total", np.count_nonzero(correct_showers), "/", nevents)

        c, _, _ = plt.hist(num_gamma_list, range=[0, 6], bins=6)
        print("Total:", np.sum(c))
        plt.show()

        c, _, _ = plt.hist(((energy_list/true_energy_list)-1.)[correct_showers], range=(-0.975, 1.025),
                           alpha=0.8, bins=40, density=False, label='Pandora')
        print("Calo:", np.sum(c), "/", nevents)
        plt.xlabel("$Reco/True - 1$", fontsize=12)
        plt.xticks(np.arange(-1, 1.2, 0.2))
        plt.legend()
        plt.show()

        print("Total:", np.sum(c))
