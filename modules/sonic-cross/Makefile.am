SUBDIRS = yaml

moddir = $(GUILE_SITE)/sonic-cross
godir = $(GUILE_SITE_CCACHE)/sonic-cross

dist_mod_DATA = api.scm backend-select.scm bootstrap.scm code-template.scm composite-view-copy-kernels.scm core-ir.scm core-sources.scm dispatcher.scm emitter.scm file-manager.scm functionalization.scm generate.scm generated-functions.scm irregex.scm native-function.scm operators.scm orchestration.scm register-dispatch-key.scm regex.scm schema-parser.scm sha1.scm tensor-body.scm ufunc.scm yaml.scm

GOBJECTS = $(dist_mod_DATA:.scm=.go)
nodist_go_DATA = $(GOBJECTS)

CLEANFILES = $(GOBJECTS)

# Compile .scm -> .go at build time (GUILE_AUTO_COMPILE=0: never lazy
# auto-compile at load time).  The source is derived from the target name
# (not $<) because the explicit dependency rules below add extra prerequisites.
%.go: %.scm
	@echo "  GUILEC   $@"
	@GUILE_AUTO_COMPILE=0 \
	  GUILE_LOAD_COMPILED_PATH="$(abs_top_builddir)/modules" \
	  $(GUILD) compile \
	    -L "$(abs_top_srcdir)" -L "$(abs_top_srcdir)/modules" \
	    -o "$@" "$(patsubst %.go,%.scm,$@)"

# Direct module dependencies (the frozen import graph).  Each X.go depends on
# the .scm of every (sonic-cross ...) module X #:use-module's, so changing a
# dependency rebuilds its importers — and only them.  Guile resolves function
# bindings at load time and SonicCross defines no macros, so direct (rather
# than transitive) dependencies are sufficient.
api.go: core-ir.scm native-function.scm schema-parser.scm yaml.scm
backend-select.go: api.scm code-template.scm core-ir.scm dispatcher.scm generated-functions.scm native-function.scm
composite-view-copy-kernels.go: api.scm code-template.scm core-ir.scm generated-functions.scm native-function.scm
core-sources.go: code-template.scm core-ir.scm generated-functions.scm native-function.scm
dispatcher.go: core-ir.scm native-function.scm
emitter.go: api.scm code-template.scm core-ir.scm dispatcher.scm generated-functions.scm native-function.scm
file-manager.go: sha1.scm
functionalization.go: api.scm code-template.scm core-ir.scm file-manager.scm generated-functions.scm native-function.scm orchestration.scm
generated-functions.go: core-ir.scm native-function.scm
generate.go: backend-select.scm composite-view-copy-kernels.scm core-ir.scm core-sources.scm emitter.scm functionalization.scm generated-functions.scm native-function.scm operators.scm orchestration.scm register-dispatch-key.scm tensor-body.scm ufunc.scm yaml.scm
native-function.go: core-ir.scm schema-parser.scm yaml.scm
operators.go: code-template.scm core-ir.scm dispatcher.scm emitter.scm file-manager.scm generated-functions.scm native-function.scm
orchestration.go: core-ir.scm file-manager.scm generated-functions.scm native-function.scm
regex.go: irregex.scm
register-dispatch-key.go: api.scm code-template.scm core-ir.scm file-manager.scm native-function.scm orchestration.scm
schema-parser.go: core-ir.scm regex.scm
tensor-body.go: api.scm code-template.scm core-ir.scm generated-functions.scm native-function.scm
ufunc.go: api.scm code-template.scm core-ir.scm emitter.scm generated-functions.scm native-function.scm yaml.scm
# yaml.scm imports (sonic-cross yaml ffi), which lives in the yaml/ subdir.
yaml.go: yaml/ffi.scm
