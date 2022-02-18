# 0_export-to-nwb
This notebook is used to convert our data sets into NWB files. Each converted NWB file contains: raw extracellular recordings - spike-sorted units - imaging series - segmentation ROIs (of the target spines and adjacent dendritic shaft). You can download from Dandi Archive (https://gui.dandiarchive.org/#/dandiset/000223). 

# 1_match-spine-trace-with-spike-trains
In this notebook, we show how to match the corresponding presynaptic cell for a given dendritic spine. More detailed explanation can be found in Results of "Inferring monosynaptic connections from paired dendritic spine Ca2+ imaging and large-scale recording of extracellular spiking" (https://doi.org/10.1101/2022.02.16.480643).

# 2_simulate_surrogate_data_for_validation
In this notebook, we show how we would accept or reject the spike train with the best match to a given spine response as a monosynaptic connection.

# 3_method_validation_and_explore_optimal_thresholds
In this notebook, we use again the surrogate methods to validate our method and show how we would find optimal thresholds for the acceptance test.
