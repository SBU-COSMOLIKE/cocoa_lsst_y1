# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
if [ -z "${IGNORE_COSMOLIKE_LSSTY1_CODE}" ]; then

  if [ -z "${ROOTDIR}" ]; then
    source start_cocoa.sh || { pfail 'ROOTDIR'; return 1; }
  fi

  # Parenthesis = run in a subshell
  ( source "${ROOTDIR:?}/installation_scripts/flags_check.sh" ) || return 1;

  PROJECT="${ROOTDIR:?}/projects"
  FOLDER="${XXX_NAME:-"xxx"}"
  PACKDIR="${PROJECT:?}/${FOLDER:?}"

  export LD_LIBRARY_PATH="${FOLDER:?}/interface":${LD_LIBRARY_PATH}

  export PYTHONPATH="${FOLDER:?}/interface":${PYTHONPATH}

  if [ -n "${COSMOLIKE_DEBUG_MODE}" ]; then
      export SPDLOG_LEVEL=debug
  else
      export SPDLOG_LEVEL=info
  fi

  unset -v PROJECT FOLDER PACKDIR

fi
