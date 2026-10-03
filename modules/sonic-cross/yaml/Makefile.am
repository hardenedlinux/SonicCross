moddir = $(GUILE_SITE)/sonic-cross/yaml
godir = $(GUILE_SITE_CCACHE)/sonic-cross/yaml

dist_mod_DATA = ffi.scm

GOBJECTS = $(dist_mod_DATA:.scm=.go)
nodist_go_DATA = $(GOBJECTS)

CLEANFILES = $(GOBJECTS)

# ffi.scm imports only Guile core modules, so it has no SonicCross dependency
# lines (see modules/sonic-cross/Makefile.am for the dependency-tracking rule).
%.go: %.scm
	@echo "  GUILEC   $@"
	@GUILE_AUTO_COMPILE=0 \
	  GUILE_LOAD_COMPILED_PATH="$(abs_top_builddir)/modules" \
	  $(GUILD) compile \
	    -L "$(abs_top_srcdir)" -L "$(abs_top_srcdir)/modules" \
	    -o "$@" "$(patsubst %.go,%.scm,$@)"
