#!/usr/bin/guile --no-auto-compile
!#
;; Emit SonicCross's structured ufunc kernels (G10) from the frozen Semantic IR:
;; UfuncCPU_add.cpp, UfuncCPUKernel_add.cpp, UfuncCUDA_add.cu.  Each is written
;; into the output directory, byte-comparable against the oracle.

(use-modules (srfi srfi-1)
             (sonic-cross yaml)
             (sonic-cross core-ir)
             (sonic-cross native-function)
             (sonic-cross generated-functions)
             (sonic-cross orchestration)
             (sonic-cross ufunc))

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

(define (backend-index-for indices key)
  (list key (or (assoc-ref indices key) '()) #f))

(unless (= (length (command-line)) 3)
  (error "usage: scheme-ufunc.scm NATIVE_FUNCTIONS_YAML OUTDIR"))
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
             (groups (filter native-functions-group? grouped))
             (ufunc-groups
              (filter (lambda (g)
                        (native-function-ufunc-inner-loop
                         (native-functions-group-out g)))
                      groups))
             (name (lambda (g)
                     (base-operator-spelling
                      (native-function-func
                       (native-functions-group-functional g)))))
             (cpu-bi (backend-index-for indices "CPU"))
             (cuda-bi (backend-index-for indices "CUDA")))
        (for-each
         (lambda (g)
           (call-with-output-file (string-append outdir "/UfuncCPU_" (name g) ".cpp")
             (lambda (port) (display (render-ufunc-cpu-cpp g cpu-bi) port)))
           (call-with-output-file (string-append outdir "/UfuncCPUKernel_" (name g) ".cpp")
             (lambda (port) (display (render-ufunc-cpu-kernel-cpp g) port)))
           (call-with-output-file (string-append outdir "/UfuncCUDA_" (name g) ".cu")
             (lambda (port) (display (render-ufunc-cuda-cu g cuda-bi) port))))
         ufunc-groups)))))
