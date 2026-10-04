#!/usr/bin/guile --no-auto-compile
!#
;; Emit SonicCross's per-operator header closure from the frozen Semantic IR.
;; Every ops/{name}[_.*].h header, the five include-only aggregate shims, and
;; the fourteen {DispatchKey}Functions.h / _inl.h headers are written into the
;; output directory under their basenames, byte-comparable against the oracle.

(use-modules (srfi srfi-1)
             (sonic-cross yaml)
             (sonic-cross core-ir)
             (sonic-cross native-function)
             (sonic-cross generated-functions)
             (sonic-cross orchestration)
             (sonic-cross per-operator-headers))

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
  (error "usage: scheme-per-operator-headers.scm NATIVE_FUNCTIONS_YAML OUTDIR"))
(let* ((root (yaml-load (cadr (command-line))))
       (outdir (caddr (command-line)))
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
      (let ((files (render-per-operator-headers
                    functions (get-grouped-native-functions functions) indices)))
        (for-each
         (lambda (entry)
           (let ((filename (cadr entry))
                 (content (caddr entry)))
             (unless (file-exists? outdir) (mkdir outdir))
             (call-with-output-file (string-append outdir "/" filename)
               (lambda (port) (display content port)))))
         files)))))
