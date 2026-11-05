#!/bin/bash
# Sourced by the ablation launchers. render_config <template.yml> fills @OUT_ROOT@ / @SHARED_ROOT@
# and sets RENDERED=$OUT_ROOT/ablations/configs/<template.yml>.
render_config() {
  local template="${PROJECT_ROOT}/ablations/configs/$1"
  [[ -f "${template}" ]] || { echo "[ERROR] template not found: ${template} (run: python ablations/make_configs.py)" >&2; exit 2; }
  mkdir -p "${OUT_ROOT}/ablations/configs"
  RENDERED="${OUT_ROOT}/ablations/configs/$1"
  sed -e "s#@OUT_ROOT@#${OUT_ROOT}#g" -e "s#@SHARED_ROOT@#${SHARED_ROOT}#g" "${template}" > "${RENDERED}"
  if grep -n "@OUT_ROOT@\|@SHARED_ROOT@" "${RENDERED}"; then echo "[ERROR] unrendered placeholder" >&2; exit 2; fi
  echo "[INFO] config: ${RENDERED}"
}
