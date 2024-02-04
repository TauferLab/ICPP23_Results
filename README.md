# Overview
Data and scripts for reproducing figures in paper:

N. Tan, B. Nicolae, J. Luettgau, J. Marquez, K. Teranishi, N. Morales, S. Bhowmick, M. Taufer, and F. Cappello. Scalable Checkpointing of Applications with Sparsely Updated Data. In Proceedings of the 52nd International Conference on Parallel Processing (ICPP), 2023.

This paper received three ACM badges: result replicated, artifact available, and artifact evaluated-functional.

# Results_Deduplication_Module

This repository is used to collect measurement as well as scripting to explore and plot the results.

> **NOTE:** The **main** branch is protected, store results to a seperate branch to avoid needing to download all the results for measurment campaigns on other systems.


    # Clone and be prompted for new branch (e.g., per system or measurement campaign)
    git clone git@github.com:TauferLab/Results_Deduplication_Module.git
    cd Results_Deduplication_Module 
    ./init.sh
    

In job scripts/post-processing scripts consider using:

    # source to obtain RESULTS_PATH environment variable, extend PATH/LD_LIBRARY_PATH/...
    source <path-to-repo>/activate

