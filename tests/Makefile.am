TESTS = load-module.scm test-core-ir.scm test-generated-functions.scm test-irregex.scm test-native-function.scm test-schema-parser.scm test-yaml.scm

AM_TESTS_ENVIRONMENT = \
  GUILE_LOAD_PATH="$(abs_top_srcdir)/modules:$(abs_top_srcdir)/scripts:$$GUILE_LOAD_PATH"; \
  export GUILE_LOAD_PATH;

EXTRA_DIST = $(TESTS) fixtures/native-functions.yaml \
  differential/fixtures/core-v0.yaml \
  differential/python-oracle.py differential/scheme-dump.scm \
  differential/compare.py \
  differential/run-differential.sh \
  differential/manifest-oracle.py differential/scheme-manifest.scm \
  differential/run-manifest-differential.sh
