#!/usr/bin/guile --no-auto-compile
!#
;; Emit SonicCross's CompositeViewCopyKernels.cpp (G11 structured composite
;; kernels + G12 view_copy kernels) from the frozen Semantic IR.  The single
;; translation unit is written into the output directory, byte-comparable
;; against the oracle.

(use-modules (srfi srfi-1)
             (sonic-cross yaml)
             (sonic-cross core-ir)
             (sonic-cross native-function)
             (sonic-cross generated-functions)
             (sonic-cross orchestration)
             (sonic-cross composite-view-copy-kernels))

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
  (fold (lambda (key-entry acc)
          (let ((key (car key-entry)))
            (fold (lambda (op-meta inner)
                    (index-add inner key (car op-meta) (cdr op-meta)))
                  acc
                  (cdr key-entry))))
        indices
        dispatch))

(define (backend-index-for indices key)
  (list key (or (assoc-ref indices key) '()) #f))

(unless (= (length (command-line)) 3)
  (error "usage: scheme-composite-view-copy-kernels.scm NATIVE_FUNCTIONS_YAML OUTDIR"))
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
      (let* ((grouped (get-grouped-native-functions functions))
             (structured (get-structured-native-functions grouped))
             (view-groups (native-functions-view-groups functions))
             (backend-index
              (backend-index-for indices "CompositeExplicitAutogradNonFunctional"))
             (content
              (render-composite-view-copy-kernels-cpp
               structured view-groups backend-index)))
        (call-with-output-file
            (string-append outdir "/CompositeViewCopyKernels.cpp")
          (lambda (port) (display content port)))))))
