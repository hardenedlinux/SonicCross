AC_INIT([SonicCross], [0.1.0], [nobody@example.invalid])
AC_CONFIG_SRCDIR([modules/sonic-cross/bootstrap.scm])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([foreign])

AC_PROG_INSTALL
GUILE_PROGS([3.0])
GUILE_SITE_DIR

AC_CONFIG_FILES([
  Makefile
  modules/Makefile
  modules/sonic-cross/Makefile
  scripts/Makefile
  tests/Makefile
  doc/Makefile
  pre-inst-env
])
AC_CONFIG_COMMANDS([pre-inst-env-mode], [chmod +x pre-inst-env])
AC_OUTPUT
