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
             (sonic-cross core-ir)
             (sonic-cross schema-parser))

(define (round-trip schema)
  (let ((parsed (parse-function-schema schema)))
    (test-equal schema (function-schema->string parsed))
    parsed))

(test-begin "sonic-cross schema parser")

;; Annotation parsing and round trips.
(for-each
 (lambda (text)
   (let ((annotation (parse-annotation text)))
     (test-equal text (annotation->string annotation))))
 '("a" "a!" "a|b" "a -> b" "a! -> b" "a -> b|c" "a -> *"))
(test-error (parse-annotation "a|b!"))
(test-error (parse-annotation "a|b -> c|d"))

;; Type parsing and recursive structure.
(for-each
 (lambda (text)
   (test-equal text (type->string (parse-type text))))
 '("Tensor" "Tensor?" "Tensor[]" "Tensor?[]" "Tensor[]?"
   "bool[3]" "SymInt[]" "Generator?"
   "__torch__.torch.classes.foo.Bar"))
(test-assert (has-symint? (parse-type "SymInt[]")))
(test-assert (not (equal? (parse-type "int") (parse-type "SymInt"))))

;; Argument and return parsing, including raw defaults and annotations.
(for-each
 (lambda (text)
   (test-equal text (argument->string (parse-argument text))))
 '("Tensor self" "Tensor(a!) self" "Scalar alpha=1"
   "Tensor? foo=None" "Tensor[] foo=None" "str mode=\"foo\""))
(for-each
 (lambda (text)
   (test-equal text (return->string (parse-return text))))
 '("Tensor" "Tensor(a!)" "Tensor values" "Tensor(a!) values"))

;; Complete schemas and seven-region grouping.
(round-trip "add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")
(let ((schema (round-trip
               "add.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")))
  (test-assert (function-schema-is-out-fn? schema))
  (test-equal schema-kind-out (function-schema-kind schema)))
(let ((schema (round-trip "resize_(Tensor(a!) self, SymInt[] size) -> Tensor(a!)")))
  (test-equal schema-kind-inplace (function-schema-kind schema))
  (test-assert (arguments-self-arg (function-schema-arguments schema))))
(let ((schema
       (round-trip
        "empty(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False) -> Tensor")))
  (let ((options (arguments-tensor-options (function-schema-arguments schema))))
    (test-assert (tensor-options-arguments? options))
    (test-equal "dtype" (argument-name (tensor-options-arguments-dtype options)))
    (test-equal '() (arguments-post-tensor-options-kwarg-only
                     (function-schema-arguments schema))))
  (test-equal schema-kind-functional (function-schema-kind schema)))
(let ((schema
       (round-trip
        "zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, MemoryFormat? memory_format=None) -> Tensor")))
  (test-equal "memory_format"
    (argument-name
     (car (arguments-post-tensor-options-kwarg-only
           (function-schema-arguments schema))))))

(let ((schema (round-trip "where.self(Tensor condition, Tensor self, Tensor other) -> Tensor")))
  (test-equal "condition"
    (argument-name (car (arguments-pre-self-positional
                         (function-schema-arguments schema)))))
  (test-equal "self"
    (argument-name
     (self-argument-argument
      (arguments-self-arg (function-schema-arguments schema))))))

;; Mutable classification is annotation-driven, not name-driven.
(let ((schema (round-trip "mutate(Tensor self, Tensor(a!) other) -> Tensor")))
  (test-equal schema-kind-mutable (function-schema-kind schema))
  (test-assert (not (function-schema-is-out-fn? schema))))

;; Invalid forms required by the frozen parser contract.
(test-error (parse-argument "x=a=b"))
(test-error (parse-argument "Tensor(a|b!) self"))
(test-error (parse-type "NotAType"))
(test-error (parse-argument "Tensor(a)?? self"))
(test-error (parse-function-schema "not a schema"))
(test-error (parse-function-schema "foo(Tensor x, *, *, Tensor y) -> Tensor"))
(test-error (parse-function-schema "foo(Tensor x, *, Tensor(a!) out, Tensor y) -> Tensor"))
(test-error
 (parse-function-schema
  "foo(Tensor x, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False) -> Tensor"))

(test-end "sonic-cross schema parser")
