#!/usr/bin/guile --no-auto-compile
!#
;; Emit SonicCross's rendered Functions.h and Functions.cpp (G5) from the frozen
;; Semantic IR into OUTDIR.  These are the exact C++ texts, so they can be
;; byte-compared against the oracle and fed to the Tier 1 semantic extractor.

(use-modules (srfi srfi-1)
             (sonic-cross yaml)
             (sonic-cross core-ir)
             (sonic-cross native-function)
             (sonic-cross generated-functions)
             (sonic-cross emitter))

(unless (= (length (command-line)) 3)
  (error "usage: scheme-functions.scm NATIVE_FUNCTIONS_YAML OUTDIR"))
(let* ((native-yaml (cadr (command-line)))
       (outdir (caddr (command-line)))
       (root (yaml-load native-yaml))
       (entries (if (yaml-sequence? root) (yaml-sequence-items root)
                    (error 'fixture-root-must-be-sequence)))
       (original
        (map (lambda (mapping)
               (call-with-values
                   (lambda () (native-function-from-yaml mapping))
                 (lambda (function dispatch) function)))
             entries))
       (functions
        (call-with-values
            (lambda () (add-generated-native-functions! original '()))
          (lambda (functions indices) functions))))
  (call-with-output-file (string-append outdir "/Functions.h")
    (lambda (port) (display (render-functions-h functions) port)))
  (call-with-output-file (string-append outdir "/Functions.cpp")
    (lambda (port) (display (render-functions-cpp) port))))
