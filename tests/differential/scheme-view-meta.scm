#!/usr/bin/guile --no-auto-compile
!#
;; Emit SonicCross's FunctionalInverses.h + ViewMetaClasses.h/.cpp (G14/G15)
;; from the frozen Semantic IR, byte-comparable against the oracle.

(use-modules (srfi srfi-1)
             (sonic-cross yaml)
             (sonic-cross core-ir)
             (sonic-cross native-function)
             (sonic-cross generated-functions)
             (sonic-cross orchestration)
             (sonic-cross functionalization))

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

(define (write-file path content)
  (call-with-output-file path (lambda (port) (display content port))))

(unless (= (length (command-line)) 3)
  (error "usage: scheme-view-meta.scm NATIVE_FUNCTIONS_YAML OUTDIR"))
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
      (let ((view-groups (native-functions-view-groups functions)))
        (write-file (string-append outdir "/FunctionalInverses.h")
                    (render-functional-inverses view-groups))
        (write-file (string-append outdir "/ViewMetaClasses.h")
                    (render-view-meta-classes-h view-groups))
        (write-file (string-append outdir "/ViewMetaClasses.cpp")
                    (render-view-meta-classes-cpp view-groups))))))
