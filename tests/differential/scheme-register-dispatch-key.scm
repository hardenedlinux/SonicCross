#!/usr/bin/guile --no-auto-compile
!#
;; Emit SonicCross's rendered Register{DispatchKey}.cpp artifacts (G2) from the
;; frozen Semantic IR.  These are the exact C++ texts, byte-comparable against
;; the oracle and feedable to the Tier 1 semantic extractor.

(use-modules (srfi srfi-1)
             (srfi srfi-13)
             (sonic-cross yaml)
             (sonic-cross core-ir)
             (sonic-cross native-function)
             (sonic-cross generated-functions)
             (sonic-cross orchestration)
             (sonic-cross register-dispatch-key))

(define (index-entry indices key)
  (find (lambda (entry) (string=? (car entry) key)) indices))

(define (index-add indices key operator metadata)
  (let ((entry (index-entry indices key)))
    (if entry
        (cons (cons key
                    (cons (cons operator metadata)
                          (filter (lambda (item)
                                    (not (string=? (car item) operator)))
                                  (cdr entry))))
              (filter (lambda (item) (not (string=? (car item) key))) indices))
        (cons (cons key (list (cons operator metadata))) indices))))

(define (merge-dispatch indices dispatch)
  ;; dispatch :: ((key (op . backend-metadata) ...) ...)
  (fold (lambda (key-entry acc)
          (let ((key (car key-entry)))
            (fold (lambda (op-meta inner)
                    (index-add inner key (car op-meta) (cdr op-meta)))
                  acc
                  (cdr key-entry))))
        indices
        dispatch))

(unless (= (length (command-line)) 3)
  (error "usage: scheme-register-dispatch-key.scm NATIVE_FUNCTIONS_YAML OUTPUT_DIR"))
(let* ((root (yaml-load (cadr (command-line))))
       (out-dir (caddr (command-line)))
       (entries (if (yaml-sequence? root) (yaml-sequence-items root)
                    (error 'fixture-root-must-be-sequence)))
       (parsed
        (map (lambda (mapping)
               (call-with-values
                   (lambda () (native-function-from-yaml mapping))
                 (lambda (function dispatch) (cons function dispatch))))
             entries))
       (original (map car parsed))
       (original-dispatch (map cdr parsed))
       (seeded (fold merge-dispatch '() original-dispatch)))
  (call-with-values
      (lambda () (add-generated-native-functions! original seeded))
    (lambda (functions indices)
      (let* ((grouped (get-grouped-native-functions functions))
             (files (render-register-dispatch-key-files grouped indices)))
        (for-each
         (lambda (entry)
           (let* ((filename (car entry))
                  (content (cdr entry))
                  (path (string-append out-dir "/" filename)))
             (unless (file-exists? out-dir) (mkdir out-dir))
             (call-with-output-file path
               (lambda (port) (display content port)))
             (format #t "~a\t~a~%" filename (string-length content))))
         files)))))
