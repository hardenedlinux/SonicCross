TESTS = load-module.scm test-core-ir.scm test-generated-functions.scm test-irregex.scm test-native-function.scm test-per-operator-headers.scm test-schema-parser.scm test-yaml.scm

AM_TESTS_ENVIRONMENT = \
  GUILE_LOAD_PATH="$(abs_top_srcdir)/modules:$(abs_top_srcdir)/scripts:$$GUILE_LOAD_PATH"; \
  export GUILE_LOAD_PATH;

EXTRA_DIST = $(TESTS) fixtures/native-functions.yaml \
  differential/fixtures/core-v0.yaml \
  differential/python-oracle.py differential/scheme-dump.scm \
  differential/compare.py \
  differential/run-differential.sh \
  differential/manifest-oracle.py differential/scheme-manifest.scm \
  differential/run-manifest-differential.sh \
  differential/per-operator-headers-oracle.py \
  differential/scheme-per-operator-headers.scm \
  differential/compare-per-operator-headers.py \
  differential/run-per-operator-headers-differential.sh \
  differential/register-dispatch-key-per-operator-oracle.py \
  differential/scheme-register-dispatch-key-per-operator.scm \
  differential/compare-register-dispatch-key-per-operator.py \
  differential/run-register-dispatch-key-per-operator-differential.sh
