# Run ONE section (a line range) of the authors' script, unchanged, in a fresh R process.
# Usage: Rscript run_section.R SCRIPT FROM TO PREV_CHECKPOINT|none OUT_CHECKPOINT
# Between sections the whole workspace (data, models, .Random.seed) is saved and reloaded, so memory is freed
# but the computation is the same as one long run. Peak RSS of this process goes to sections_mem.tsv.
# Runner state lives in local(), NOT the global env: load() of a checkpoint would otherwise overwrite it.
local({
args <- commandArgs(TRUE)
file <- args[1]; from <- as.integer(args[2]); to <- as.integer(args[3]); prev <- args[4]; out <- args[5]
exprs <- parse(file, keep.source = TRUE)
starts <- vapply(attr(exprs, "srcref"), function(s) as.integer(s[1]), 1L)
if (prev != "none") {
  load(prev, envir = globalenv())
  # A saved workspace does not keep attached packages: re-run every library() call the script made before FROM.
  for (i in which(starts < from)) {
    e <- exprs[[i]]
    if (is.call(e) && identical(e[[1]], as.name("library"))) eval(e, globalenv())
  }
  sink("log.txt", append = TRUE, type = "output")   # continue the script's own sink(file = "log.txt")
}
for (i in which(starts >= from & starts <= to)) {
  r <- withVisible(eval(exprs[[i]], globalenv()))
  if (r$visible) print(r$value)                       # what Rscript prints for a top-level expression
}
while (sink.number() > 0) sink()
save.image(out)
hwm <- if (file.exists("/proc/self/status")) grep("^VmHWM", readLines("/proc/self/status"), value = TRUE) else
  sprintf("WinPeakMB: %s", memory.size(max = TRUE))   # native Windows R
cat(sprintf("%s\t%d-%d\t%s\n", out, from, to, gsub("[[:space:]]+", " ", hwm)), file = "sections_mem.tsv", append = TRUE)
})
