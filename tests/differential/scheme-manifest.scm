#!/usr/bin/guile --no-auto-compile
!#
;; Emit the SonicCross Tier 0 artifact manifest in the same format as
;; manifest-oracle.py:
;;
;;     <label>\t<filename>\t<ops>
;;
;; where <ops> is "-" for unsharded artifacts or a space-separated, sorted
;; multiset of root names for sharded artifacts.

(use-modules (srfi srfi-1)
             (srfi srfi-13)
             (sonic-cross yaml)
             (sonic-cross core-ir)
             (sonic-cross native-function)
             (sonic-cross generated-functions)
             (sonic-cross orchestration))

(unless (= (length (command-line)) 2)
  (error "usage: scheme-manifest.scm NATIVE_FUNCTIONS_YAML"))
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
  (for-each
   (lambda (entry)
     (format #t "~a\t~a\t~a~%" (car entry) (cadr entry) (caddr entry)))
   (generate-artifact-manifest functions)))
