#!/usr/bin/guile --no-auto-compile
!#

(use-modules (srfi srfi-64)
             (sonic-cross bootstrap))

(test-begin "sonic-cross infrastructure")
(test-equal 'sonic-cross-bootstrap
  (sonic-cross-bootstrap-message))
(test-end)
