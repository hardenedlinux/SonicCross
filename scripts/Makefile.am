scriptdir = $(GUILE_SITE)/scripts
godir = $(GUILE_SITE_CCACHE)/scripts

dist_script_DATA = torchgen.scm

GOBJECTS = $(dist_script_DATA:.scm=.go)
nodist_go_DATA = $(GOBJECTS)

CLEANFILES = $(GOBJECTS)

%.go: %.scm
	@echo "  GUILEC   $@"
	@GUILE_AUTO_COMPILE=0 \
	  GUILE_LOAD_COMPILED_PATH="$(abs_top_builddir)/modules" \
	  $(GUILD) compile \
	    -L "$(abs_top_srcdir)" -L "$(abs_top_srcdir)/modules" \
	    -o "$@" "$(patsubst %.go,%.scm,$@)"

# torchgen.scm imports (sonic-cross generate), which lives in the modules tree.
torchgen.go: $(abs_top_srcdir)/modules/sonic-cross/generate.scm
