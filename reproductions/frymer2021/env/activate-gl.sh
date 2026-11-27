# Copied into env/r351/etc/conda/activate.d/ by setup: lets rgl (pulled in by LMERConvenienceFunctions)
# find the libGL that ships inside the env, and use rgl's null device (no display on the XDC).
export LD_LIBRARY_PATH="$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export RGL_USE_NULL=TRUE
