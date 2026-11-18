# Packages that produce the paper's numbers, at the version current on 2016-01-20 (ei 1.3's CRAN release;
# the authors' code is dated 2015-12-09 .. 2016-01-09). Posit snapshots start 2017-10-10, so we install
# exact tarballs from the CRAN Archive, in dependency order. Versions picked with: latest archived release
# dated <= 2016-01-20 (one exception: ucminf, below). Run from the paper folder: micromamba run -p env/r351 Rscript env/cran_packages.R
Sys.setenv(R_MAKEVARS_USER = normalizePath("env/Makevars"))
arch <- "https://cran.r-project.org/src/contrib/Archive"
pins <- c(xtable = "1.8-0",      # 2015-11-02  Table*.R
          wru = "0.0-1",         # 2015-12-09  Predict.R race.pred(); 0.0-2 came 2016-03-11
          ROCR = "1.0-7",        # 2015-03-26  Figure1.R
          mvtnorm = "1.0-4",     # 2016-01-19
          msm = "1.6",           # 2015-11-18
          mnormt = "1.5-3",      # 2015-05-25
          cubature = "1.1-2",    # 2013-02-25
          ucminf = "1.1-4",      # 2016-08-18, NOT 1.1-3 (2012, current in Jan 2016). See below.
          sandwich = "2.3-4",    # 2015-09-24  (gmm import)
          gmm = "1.5-2",         # 2015-03-17  (tmvtnorm import)
          tmvtnorm = "1.4-10",   # 2015-08-28
          eiPack = "0.1-7",      # 2012-01-13
          ei = "1.3")            # 2016-01-20  KingEI.R; first CRAN release (authors used a pre-CRAN build)
# ucminf: 1.1-3's C code lets Fortran overwrite its input vectors in place, including .stepmax, which aliases
# the literal 1 in ucminf()'s own code. On R 3.5 that corrupts the step bound, and ei() (KingEI.R) fails:
# district 1 White with "lower>upper integration limits", or, installed without byte-compiling, the 5th fit
# with "sigma must be a symmetric matrix" (both observed on R 3.5.1). 1.1-4 (Aug 2016)
# only adds duplicate() of those in/out arguments in interface.c (plus numDeriv namespace calls on a path
# ei does not use: hessian = 3); the optimizer is the same. The authors' R 3.2 runs did not hit the bug.
ip <- installed.packages()[, "Version"]
for (p in names(pins)) {
  if (!is.na(ip[p]) && ip[p] == pins[[p]]) next
  url <- sprintf("%s/%s/%s_%s.tar.gz", arch, p, p, pins[[p]])
  tf <- file.path(tempdir(), basename(url)); download.file(url, tf, quiet = TRUE)
  install.packages(tf, repos = NULL, type = "source")
}
ip <- installed.packages()[, "Version"]
bad <- names(pins)[is.na(ip[names(pins)]) | ip[names(pins)] != pins]
if (length(bad)) stop("not at pinned version: ", paste(bad, collapse = ", "))
cat("CRAN pins OK\n")
