# Native Windows R 3.5.1 for the two mediate() sections only (they need ~9.3 GB; a 2 GB XDC cannot run them).
# Matches env/r-packages.csv (the XDC's conda env) as closely as R 3.5 Windows binaries allow.
# Usage: Rscript install_win_r351.R LIBDIR
lib <- commandArgs(TRUE)[1]; .libPaths(lib)
ppm <- function(d) sprintf("https://packagemanager.posit.co/cran/%s", d)
want <- read.csv("r-packages.csv", stringsAsFactors = FALSE)
base <- rownames(installed.packages(priority = "base"))
pk <- setdiff(want$Package, base)
# Exact XDC versions that the 2020-05-01 snapshot does not have as R 3.5 binaries:
exact <- list("2019-09-01" = c("BH", "mediation", "stringi", "latticeExtra", "pbkrtest"),
              "2020-04-01" = "foreign",
              "2020-02-03" = "Hmisc")                       # 4.3-0: no R 3.5 binary of 4.4-0 exists
for (d in names(exact)) install.packages(exact[[d]], lib, repos = ppm(d), type = "win.binary", dependencies = FALSE)
car <- file.path(tempdir(), "car_3.0-7.tar.gz")
download.file("https://cran.r-project.org/src/contrib/Archive/car/car_3.0-7.tar.gz", car, mode = "wb")
install.packages(car, lib, repos = NULL, type = "source")  # pure R, no compiler needed
rest <- setdiff(pk, rownames(installed.packages(lib)))
if (length(rest)) install.packages(rest, lib, repos = ppm("2020-05-01"), type = "win.binary", dependencies = FALSE)
ip <- installed.packages(lib)[, "Version"]
cmp <- data.frame(Package = pk, xdc = want$Version[match(pk, want$Package)], win = unname(ip[pk]), stringsAsFactors = FALSE)
print(cmp[is.na(cmp$win) | cmp$win != cmp$xdc, ], row.names = FALSE)
write.csv(cmp, "r-packages-win.csv", row.names = FALSE)
