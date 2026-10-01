#!/usr/bin/guile --no-auto-compile
!#

;;  -*-  indent-tabs-mode:nil; coding: utf-8 -*-
;;  Copyright (C) 2026
;;      "Mu Lei" known as "NalaGinrut" <roy@hardenedlinux.org>
;;  SonicCross is free software: you can redistribute it and/or modify
;;  it under the terms of the GNU General Public License published
;;  by the Free Software Foundation, either version 3 of the License,
;;  or (at your option) any later version.

(use-modules (srfi srfi-64)
             (srfi srfi-1)
             (sonic-cross yaml)
             (sonic-cross core-ir)
             (sonic-cross native-function)
             (sonic-cross schema-parser)
             (sonic-cross generated-functions))

(define (nf text)
  (call-with-values
      (lambda () (native-function-from-yaml (yaml-load-string text)))
    (lambda (function dispatch) function)))

(define (generate functions)
  (call-with-values
      (lambda () (add-generated-native-functions! functions '()))
    (lambda (result indices) (cons result indices))))

(define (schema-of function)
  (function-schema->string (native-function-func function)))

(define (yaml-for schema . fields)
  (string-append "func: " schema "\n"
                 (apply string-append fields)))

(test-begin "sonic-cross generated functions")

;; The four prerequisite predicates.
(let* ((plain (nf (yaml-for "sym(SymInt x) -> Tensor" "dispatch:\n  CPU: sym_cpu\n")))
       (optional (nf (yaml-for "sym(SymInt? x) -> Tensor" "dispatch:\n  CPU: sym_cpu\n")))
       (list-type (nf (yaml-for "sym(SymInt[] x) -> Tensor" "dispatch:\n  CPU: sym_cpu\n"))))
  (test-assert (function-schema-has-symint?
                (native-function-func plain)))
  (test-assert (function-schema-has-symint?
                (native-function-func optional)))
  (test-assert (function-schema-has-symint?
                (native-function-func list-type))))

(for-each
 (lambda (item)
   (test-equal (cdr item)
     (type-is-tensor-like? (parse-type (car item)))))
 '(("Tensor" . #t) ("Tensor?" . #t) ("Tensor[]" . #t)
   ("Tensor?[]" . #t) ("Tensor[]?" . #t) ("SymInt" . #f)
   ("Scalar" . #f) ("bool" . #f) ("Generator" . #f)))

(let ((view (nf (yaml-for "view(Tensor self) -> Tensor(a)"
                          "dispatch:\n  CPU: view_cpu\n")))
      (wild (nf (yaml-for "wild(Tensor(a -> *) self) -> Tensor"
                          "dispatch:\n  CPU: wild_cpu\n")))
      (tagged (nf (yaml-for "foo(Tensor self) -> Tensor"
                            "tags: inplace_view\ndispatch:\n  CPU: foo_cpu\n")))
      (resize (nf (yaml-for "resize_(Tensor(a!) self) -> Tensor(a!)"
                            "tags: inplace_view\ndispatch:\n  CPU: resize_cpu\n"))))
  (test-assert (native-function-is-view-op? view))
  (test-assert (native-function-is-view-op? wild))
  (test-assert (native-function-is-view-op? tagged))
  (test-assert (not (native-function-is-view-op? resize))))

;; Functional with autogen out produces out only, with the generated metadata.
(let* ((source (nf (yaml-for "add.Tensor(Tensor self) -> Tensor"
                             "autogen: add.out\ndispatch:\n  CPU: add_cpu\n")))
       (generated (car (generate (list source))))
       (out (cadr generated)))
  (test-equal 2 (length generated))
  (test-equal "add.Tensor(Tensor self) -> Tensor" (schema-of (car generated)))
  (test-equal "add.Tensor_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"
              (schema-of out))
  (test-assert (member "generated" (native-function-tags out)))
  (test-assert (member "out" (native-function-tags out)))
  (let* ((indices (cdr (generate (list source))))
         (entry (assoc "CompositeExplicitAutograd" indices))
         (metadata (cdr (car (cdr entry)))))
    (test-equal "add_Tensor_out" (backend-metadata-kernel metadata))))

;; Inplace and mutable sources generate out before functional.
(let* ((inplace (nf (yaml-for "add_(Tensor(a!) self) -> Tensor(a!)"
                              "autogen: add.out\ndispatch:\n  CPU: add_cpu\n")))
       (result (car (generate (list inplace)))))
  (test-equal 3 (length result))
  (test-equal "add_(Tensor(a!) self) -> Tensor(a!)" (schema-of (car result)))
  (test-equal "add.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"
              (schema-of (cadr result)))
  (test-assert (member "generated" (native-function-tags (caddr result)))))

(let* ((mutable (nf (yaml-for "add(Tensor self, Tensor(a!) other) -> Tensor"
                             "autogen: add.out\ndispatch:\n  CPU: add_cpu\n")))
       (result (car (generate (list mutable)))))
  (test-equal 3 (length result))
  (test-equal "add.out(Tensor self, Tensor(a!) other, *, Tensor(b!) out) -> Tensor(b!)"
              (schema-of (cadr result)))
  (test-assert
   (base-operator-name-functional-overload?
    (operator-name-base
     (function-schema-name
      (native-function-func (caddr result))))))
  (test-equal "add_functional"
              (operator-name->string
               (function-schema-name
                (native-function-func (caddr result))))))

;; Existing out pairing suppresses generation; composite implicit non-core skips.
(let* ((functional (nf (yaml-for "add.Tensor(Tensor self) -> Tensor"
                                 "dispatch:\n  CPU: add_cpu\n")))
       (out (nf (yaml-for "add.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"
                          "dispatch:\n  CPU: add_out\n"))))
  (test-equal 2 (length (car (generate (list functional out))))))

(let ((composite (nf (yaml-for "add.Tensor(Tensor self) -> Tensor"
                               "dispatch:\n  CompositeImplicitAutograd: add\n"))))
  (test-equal 1 (length (car (generate (list composite))))))

(let ((core (nf (yaml-for "add.Tensor(Tensor self) -> Tensor"
                          "tags: core\nautogen: add.out\ndispatch:\n  CompositeImplicitAutograd: add\n"))))
  (test-equal 2 (length (car (generate (list core))))))

(let ((runner (test-runner-current)))
  (test-end)
  (exit (if (zero? (test-runner-fail-count runner)) 0 1)))
