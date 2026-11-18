# Packages without an R 3.5 build on conda-forge, from CRAN as of 2019-04-01
# (R 3.6.0 came out 2019-04-26, so this snapshot still targets R 3.5).
# Run from the paper folder: micromamba run -p env/r351 Rscript env/cran_packages.R
Sys.setenv(R_MAKEVARS_USER = normalizePath("env/Makevars"))
snapshot <- "https://packagemanager.posit.co/cran/2019-04-01"
pkgs <- c("rgenoud", "mediation", "LCFdata", "LMERConvenienceFunctions", "DataCombine",
          "ltm")  # ltm: library(ltm) at line 549 of the authors' script
need <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(need)) install.packages(need, repos = snapshot, Ncpus = 2)
missing <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(missing)) stop("not installed: ", paste(missing, collapse = ", "))
cat("CRAN packages OK\n")
