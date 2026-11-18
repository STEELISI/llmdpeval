# Copied into env/py37/etc/conda/activate.d/ by setup.sh: the pip wheel faiss==1.5.3 links against system
# libopenblas.so.0 and libomp.so, which the XDC does not have; the env ships them (conda-forge libopenblas,
# llvm-openmp), so point the loader at the env's lib/.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
