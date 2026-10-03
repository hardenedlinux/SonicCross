#!/usr/bin/guile --no-auto-compile
!#
;; Emit SonicCross's rendered RegisterSchema.cpp (G1) from the frozen Semantic IR.
;; This is the exact C++ text, so it can be byte-compared against the oracle and
;; fed to the Tier 1 semantic extractor.

(use-modules (srfi srfi-1)
             (sonic-cross yaml)
             (sonic-cross core-ir)
             (sonic-cross native-function)
             (sonic-cross generated-functions)
             (sonic-cross emitter))

(unless (= (length (command-line)) 2)
  (error "usage: scheme-register-schema.scm NATIVE_FUNCTIONS_YAML"))
(let* ((root (yaml-load (cadr (command-line))))
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
  (display (render-register-schema-cpp functions)))
