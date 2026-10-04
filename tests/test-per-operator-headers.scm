#!/usr/bin/guile --no-auto-compile
!#

;;  -*-  indent-tabs-mode:nil; coding: utf-8 -*-
;;  Copyright (C) 2026
;;      "Mu Lei" known as "NalaGinrut" <roy@hardenedlinux.org>
;;  SonicCross is free software: you can redistribute it and/or modify
;;  it under the terms of the GNU General Public License published
;;  by the Free Software Foundation, either version 3 of the License,
;;  or (at your option) any later version.

;;  SonicCross is distributed in the hope that it will be useful,
;;  but WITHOUT ANY WARRANTY; without even the implied warranty of
;;  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
;;  GNU General Public License for more details.

;;  You should have received a copy of the GNU General Public License
;;  along with this program. If not, see <http://www.gnu.org/licenses/>.

(use-modules (srfi srfi-64)
             (srfi srfi-1)
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
  (fold (lambda (key-entry acc)
          (let ((key (car key-entry)))
            (fold (lambda (op-meta inner)
                    (index-add inner key (car op-meta) (cdr op-meta)))
                  acc
                  (cdr key-entry))))
        indices
        dispatch))

(define (load-corpus-text text)
  ;; Returns (values functions indices) from a YAML sequence document.
  (let* ((root (yaml-load-string text))
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
    (add-generated-native-functions! original seeded)))

(define (files-for label result)
  ;; All filenames under the given label, in generation order.
  (map cadr (filter (lambda (entry) (eq? (car entry) label)) result)))

(define (content-for filename result)
  (let ((entry (find (lambda (e) (string=? (cadr e) filename)) result)))
    (and entry (caddr entry))))

(define corpus
  "- func: add.Tensor(Tensor self, Tensor other) -> Tensor
  variants: function, method
  dispatch:
    CPU: add_cpu
- func: zeros(SymInt[] size, *, ScalarType dtype=None) -> Tensor
  variants: function
  dispatch:
    CPU: zeros_cpu
")

(test-begin "sonic-cross per-operator headers")

(call-with-values
    (lambda () (load-corpus-text corpus))
  (lambda (functions indices)
    (let* ((grouped (get-grouped-native-functions functions))
           (result (render-per-operator-headers functions grouped indices)))

      ;; Every entry is a (label filename content) triple of strings.
      (test-assert (every (lambda (e)
                            (and (= 3 (length e))
                                 (symbol? (car e))
                                 (string? (cadr e))
                                 (string? (caddr e))))
                          result))

      ;; ops files for each root: {name}_ops.h, {name}.h, {name}_native.h.
      (let ((ops (files-for 'ops result)))
        (test-assert (member "add_ops.h" ops))
        (test-assert (member "add.h" ops))
        (test-assert (member "add_native.h" ops))
        (test-assert (member "zeros_ops.h" ops))
        (test-assert (member "zeros.h" ops))
        (test-assert (member "zeros_native.h" ops))
        ;; non-structured roots must not produce a _meta.h.
        (test-assert (not (member "add_meta.h" ops)))
        (test-assert (not (member "zeros_meta.h" ops))))

      ;; The five aggregate shims are cpu-labeled and include-only.
      (let ((cpu (files-for 'cpu result)))
        (for-each
         (lambda (shim) (test-assert (member shim cpu)))
         '("Functions.h" "Operators.h" "NativeMetaFunctions.h"
           "NativeFunctions.h" "MethodOperators.h")))

      ;; Shims are include-only: they #include the per-op headers and carry no
      ;; inline declarations (an empty declarations block substitutes to '').
      (let ((functions-h (content-for "Functions.h" result)))
        (test-assert (string-contains functions-h "#include <ATen/ops/add.h>"))
        (test-assert (string-contains functions-h "#include <ATen/ops/zeros.h>")))

      ;; MethodOperators.h only lists roots with a method variant.
      (let ((method-h (content-for "MethodOperators.h" result)))
        (test-assert (string-contains method-h "#include <ATen/ops/add_ops.h>"))
        (test-assert (not (string-contains method-h
                                          "#include <ATen/ops/zeros_ops.h>"))))

      ;; Every ops header starts with the canonical pragma banner.
      (test-assert
       (string-prefix? "#pragma once"
                       (content-for "add_ops.h" result)))

      ;; The ops header declares the operator inside at::_ops.
      (test-assert
       (string-contains (content-for "add_ops.h" result) "namespace _ops"))

      ;; The function header pulls in the ops header and emits a definition.
      (let ((add-h (content-for "add.h" result)))
        (test-assert (string-contains add-h "#include <ATen/ops/add_ops.h>"))
        (test-assert (string-contains add-h "add(")))

      ;; {key}Functions* headers are present for every functions key.
      (for-each
       (lambda (key)
         (let ((h (string-append key "Functions.h"))
               (inl (string-append key "Functions_inl.h")))
           (test-assert (member h (files-for 'cpu result)))
           (test-assert (member inl (files-for 'cpu result)))))
       '("CPU" "CompositeImplicitAutograd"
         "CompositeImplicitAutogradNestedTensor"
         "CompositeExplicitAutograd"
         "CompositeExplicitAutogradNonFunctional" "Meta")))))

;; Structured kernels produce {name}_meta.h (functional + structured out pair).
(call-with-values
    (lambda ()
      (load-corpus-text
       "- func: sgn(Tensor self) -> Tensor
  variants: function, method
  structured_delegate: sgn.out
  dispatch:
    CPU: sgn_cpu
- func: sgn.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU: sgn_out
"))
  (lambda (functions indices)
    (let* ((grouped (get-grouped-native-functions functions))
           (result (render-per-operator-headers functions grouped indices)))
      (test-assert (member "sgn_meta.h" (files-for 'ops result)))
      (test-assert
       (string-contains (content-for "sgn_meta.h" result) "namespace meta")))))

(let ((runner (test-runner-current)))
  (test-end)
  (exit (if (zero? (test-runner-fail-count runner)) 0 1)))
