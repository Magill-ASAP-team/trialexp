"""
Compute extremum channel for each unit in a SortingAnalyzer.
"""

from __future__ import annotations

import pandas as pd
from spikeinterface.core.sortinganalyzer import register_result_extension, AnalyzerExtension
from spikeinterface.core import SortingAnalyzer
import numpy as np

class ComputeFiringProperties(AnalyzerExtension):
    """
    Compute the extremum channel for each unit.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The SortingAnalyzer object
    peak_sign : {"neg", "pos"}, default: "neg"
        Whether to use the positive ("pos") or negative ("neg") peaks to estimate extremum channels.

    Returns
    -------
    extremum_channels : pd.Series
        Series with unit_id as index and extremum channel_id as values

    Examples
    --------
    >>> analyzer.compute("extremum_channel")
    >>> extremum_channels = analyzer.get_extension("extremum_channel").get_data()
    """

    extension_name = "firing_properties"
    depend_on = ["autocorrelograms"]
    need_recording = False
    use_nodepipeline = False
    need_job_kwargs = False
    need_backward_compatibility_on_load = False

    def _set_params(self, long_isi_limit=2, post_spike_suppression_win=[600,900], autocorrelogram_bin_size=1):
        # long_isi_limit is in s
        # post_spike_suppression_win is in ms
        # autocorrelogram_bin_size is in ms
        params = dict(long_isi_limit=long_isi_limit, 
                      post_spike_suppression_win=post_spike_suppression_win, 
                      autocorrelogram_bin_size=autocorrelogram_bin_size)

        return params

    def _select_extension_data(self, unit_ids):
        new_extremum_channels = self.data["extremum_channels"].loc[unit_ids]
        return dict(extremum_channels=new_extremum_channels)

    def _merge_extension_data(
        self, merge_unit_groups, new_unit_ids, new_sorting_analyzer, keep_mask=None, verbose=False, **job_kwargs
    ):
        pass #TODO
        # peak_sign = self.params["peak_sign"]
        # old_extremum_channels = self.data["extremum_channels"]

        # all_unit_ids = new_sorting_analyzer.unit_ids
        # not_new_ids = all_unit_ids[~pd.Series(all_unit_ids).isin(new_unit_ids).values]

        # # Compute extremum channels for new units
        # new_extremum_channels = get_template_extremum_channel(
        #     new_sorting_analyzer, peak_sign=peak_sign, outputs="id"
        # )

        # # Combine old and new
        # extremum_channels = pd.Series(index=all_unit_ids, dtype=old_extremum_channels.dtype)
        # extremum_channels.loc[not_new_ids] = old_extremum_channels.loc[not_new_ids]
        # extremum_channels.loc[new_unit_ids] = pd.Series(new_extremum_channels)[new_unit_ids]

        # new_data = dict(extremum_channels=extremum_channels)
        # return new_data

    def _run(self, verbose=False):
        # peak_sign = self.params["peak_sign"]
        unit_ids = self.sorting_analyzer.unit_ids
        df_firing_properties = pd.DataFrame(index=unit_ids)
        
        
        long_isi_limit = self.params['long_isi_limit']
        long_isi_portion = self.compute_long_isi_portion(self.sorting_analyzer, long_isi_limit=long_isi_limit)
        
        post_spike_suppression_win = self.params['post_spike_suppression_win']
        post_spike_suppression = self.compute_post_spike_suppression(self.sorting_analyzer, post_spike_suppression_win=post_spike_suppression_win)

        # # Compute extremum channel for each unit
        # extremum_channel_dict = get_template_extremum_channel(
        #     self.sorting_analyzer, peak_sign=peak_sign, outputs="index"
        # )

        # # Convert to pandas Series
        # extremum_channels = pd.Series(extremum_channel_dict, name="extremum_channel_id")
        # extremum_channels.index.name = "unit_id"
        
        # self.data

        df_firing_properties['long_isi_portion'] = pd.Series(long_isi_portion)
        df_firing_properties['post_spike_suppression_ms'] = pd.Series(post_spike_suppression)
        
        self.data['firing_properties'] = df_firing_properties

    def _get_data(self):
        return self.data["firing_properties"]
    
    def compute_long_isi_portion(self, sorting_analyzer:SortingAnalyzer, long_isi_limit=2):
        sorting = sorting_analyzer.sorting
 
        num_segs = sorting.get_num_segments()
        unit_ids = sorting.unit_ids
        total_duration = sorting_analyzer.get_total_duration()
        fs = sorting_analyzer.sampling_frequency
        
        long_isi_portion = {}
        
        for unit_id in unit_ids:
            isi_total_duration = 0
            for segment_index in range(num_segs):
                st = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)/fs 
                isi = np.diff(st) # in second
                isi = isi[isi>long_isi_limit]
                isi_total_duration += np.sum(isi)
                
            long_isi_portion[unit_id] = isi_total_duration/total_duration
        
        return long_isi_portion
    
    def compute_post_spike_suppression(self, sorting_analyzer:SortingAnalyzer, post_spike_suppression_win=[600,900]):
        # post_spike_suppression_win is in ms
            # Get autocorrelogram extension
            autocorr_ext = sorting_analyzer.get_extension("autocorrelograms")
            if autocorr_ext is None:
                raise ValueError("Autocorrelograms extension must be computed first")
            
            autocorrelograms, bin_size = autocorr_ext.get_data()
            bin_size = self.params['autocorrelogram_bin_size']  # in ms
            unit_ids = sorting_analyzer.unit_ids
            
            post_spike_suppression = {}
            
            for uidx in range(len(unit_ids)):
                autocorr = autocorrelograms[uidx]
                
                # Get the baseline firing rate from the specified window (e.g., 600-900 ms)
                win_start_bin = int(post_spike_suppression_win[0] / bin_size)
                win_end_bin = int(post_spike_suppression_win[1] / bin_size)
                
                # Get bins after center (positive lag)
                center_bin = len(autocorr) // 2
                baseline_bins = autocorr[center_bin + win_start_bin:center_bin + win_end_bin]
                avg_baseline_rate = np.mean(baseline_bins)
                
                # Count bins from center until firing rate >= baseline
                suppression_bins = 0
                for i in range(1, center_bin):  # Start from 1 to skip the center bin
                    if autocorr[center_bin + i] >= avg_baseline_rate:
                        break
                    suppression_bins += 1
                
                # Convert to ms
                post_spike_suppression[unit_ids[uidx]] = suppression_bins * bin_size
            
            return post_spike_suppression


register_result_extension(ComputeFiringProperties)
compute_extremum_channel = ComputeFiringProperties.function_factory()
