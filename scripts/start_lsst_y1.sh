export LD_LIBRARY_PATH="${ROOTDIR:?}/projects/lsst_y1/interface":${LD_LIBRARY_PATH}

export PYTHONPATH="${ROOTDIR:?}/projects/lsst_y1/interface":${PYTHONPATH}

if [ -n "${COSMOLIKE_DEBUG_MODE}" ]; then
    export SPDLOG_LEVEL=debug
else
    export SPDLOG_LEVEL=info
fi
