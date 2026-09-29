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
             (sonic-cross core-ir))

(define tensor (make-tensor-type))
(define integer (make-int-type))
(define symint (make-sym-int-type))

(define (arg name type default annotation write?)
  (make-argument name type default annotation write?))

(define (returns . values)
  values)

(test-begin "sonic-cross core ir v0")

;; OperatorName and BaseOperatorName.
(let ((plain (parse-operator-name "add"))
      (overloaded (parse-operator-name "add.Tensor"))
      (out (parse-operator-name "add.out"))
      (overloaded-out (parse-operator-name "add.Tensor_out"))
      (inplace (parse-operator-name "add_")))
  (test-equal "add" (base-operator-name-base (operator-name-base plain)))
  (test-equal "" (operator-name-overload-name plain))
  (test-equal "Tensor" (operator-name-overload-name overloaded))
  (test-equal "out" (operator-name-overload-name out))
  (test-equal "Tensor_out" (operator-name-overload-name overloaded-out))
  (test-assert (not (operator-name-inplace? plain)))
  (test-assert (operator-name-inplace? inplace))
  (test-equal "add" (base-operator-name-base
                      (operator-name-base inplace)))
  (test-equal "add.Tensor" (operator-name->string overloaded)))

(test-equal "add" (base-operator-name-base
                    (operator-name-base (parse-operator-name "__add__"))))
(test-assert (operator-name-dunder-method?
              (parse-operator-name "__add__")))
(test-assert (operator-name-inplace?
              (parse-operator-name "__iadd__")))
(test-equal "add" (base-operator-name-base
                    (operator-name-base (parse-operator-name
                                         "aten::__iadd__"))))
(for-each
 (lambda (name)
   (test-assert (operator-name-inplace?
                 (parse-operator-name (string-append "__i" name "__")))))
 '("add" "sub" "mul" "div" "mod" "pow" "lshift" "rshift"
   "and" "xor" "or"))
(test-error (parse-operator-name "__ifoobar__"))

(let* ((base (make-base-operator-name "add" #f #f #t))
       (name (make-operator-name base "")))
  (test-assert (operator-name-functional-overload? name))
  (test-assert (not (operator-name-inplace? name))))

;; Type structure and predicates.
(test-assert (type-is-tensor-like? tensor))
(test-assert (not (type-is-tensor-like? integer)))
(test-assert (not (equal? integer symint)))
(test-assert (has-symint? symint))
(test-assert (has-symint? (make-optional-type (make-list-type symint))))
(test-assert (not (has-symint? (make-optional-type integer))))
(test-equal 'optional (type-kind (make-optional-type tensor)))
(test-equal 'list (type-kind (make-list-type tensor)))
(test-equal 'const (type-kind (make-const-type)))
(test-equal 'base-type (type-kind (make-base-type)))
(test-assert (equal? (make-const-type) (make-const-type)))
(test-equal (type-hash (make-base-type))
  (type-hash (make-base-type)))

;; SchemaKind and Arguments.
(define functional-arguments
  (make-arguments (list (arg "self" tensor #f 'alias #f))
                  #f '() '() #f '() '()))
(define inplace-arguments
  (make-arguments '() (arg "self" tensor #f 'write #t) '() '() #f '() '()))
(define mutable-arguments
  (make-arguments '() (arg "self" tensor #f #f #f) '()
                  (list (arg "other" tensor #f 'write #t))
                  #f '() '()))
(define out-arguments
  (make-arguments '() (arg "self" tensor #f #f #f) '() '() #f '()
                  (list (arg "out" tensor #f 'write #t))))

(define (schema arguments name)
  (make-function-schema (parse-operator-name name) arguments '()))

(test-equal schema-kind-functional
  (function-schema-kind (schema functional-arguments "add")))
(test-equal schema-kind-inplace
  (function-schema-kind (schema inplace-arguments "add_")))
(test-equal schema-kind-mutable
  (function-schema-kind (schema mutable-arguments "add")))
(test-equal schema-kind-out
  (function-schema-kind (schema out-arguments "add")))
(test-assert (function-schema-is-out-fn? (schema out-arguments "add")))
(test-assert (not (function-schema-is-out-fn?
                   (schema functional-arguments "add.out"))))
(test-assert (view-schema-kind? view-schema-kind-aliasing))
(test-assert (view-schema-kind? view-schema-kind-aliasing-inplace))
(test-assert (view-schema-kind? view-schema-kind-non-aliasing))
(test-equal 'aliasing_inplace view-schema-kind-aliasing-inplace)
(test-equal 'non_aliasing view-schema-kind-non-aliasing)

;; Argument and Return preserve defaults, annotations, and names.
(let ((a (arg "alpha" (make-scalar-type) "1" 'alias #t))
      (r (make-return "result" tensor 'alias)))
  (test-equal "alpha" (argument-name a))
  (test-equal "1" (argument-default a))
  (test-equal 'alias (argument-annotation a))
  (test-assert (argument-is-write? a))
  (test-equal "result" (return-name r))
  (test-equal 'alias (return-annotation r)))

;; Signature normalization: aliases, overloads, writes, out, and tensor
;; options are normalized structurally while return order is retained.
(define signature-arguments
  (make-arguments
   (list (arg "self" tensor #f 'self-alias #f))
   #f
   (list (arg "other" tensor #f 'other-alias #f))
   (list (arg "before" (make-scalar-type) "1" 'before-alias #f))
   (arg "options" (make-type 'memory-format '()) #f 'options-alias #f)
   (list (arg "after" (make-scalar-type) "2" 'after-alias #f))
   (list (arg "out" tensor #f 'out-alias #t))))
(define original-returns
  (list (make-return "first" tensor 'return-alias)
        (make-return "second" (make-list-type tensor) #f)))
(define original-schema
  (make-function-schema (parse-operator-name "add.Tensor_out")
                        signature-arguments original-returns))
(define normalized
  (function-schema-signature original-schema))
(define named-normalized
  (function-schema-signature original-schema #:keep-return-names #t))

(test-equal "add" (base-operator-name-base
                    (operator-name-base (function-schema-name normalized))))
(test-equal "" (operator-name-overload-name (function-schema-name normalized)))
(test-assert (not (operator-name-inplace?
                   (function-schema-name normalized))))
(test-assert (not (operator-name-functional-overload?
                   (function-schema-name normalized))))
(test-equal '() (arguments-out (function-schema-arguments normalized)))
(test-equal #f (arguments-tensor-options (function-schema-arguments normalized)))
(test-equal '() (arguments-post-tensor-options-kwarg-only
                 (function-schema-arguments normalized)))
(test-equal 2 (length (arguments-pre-tensor-options-kwarg-only
                       (function-schema-arguments normalized))))
(test-equal #f (return-name (car (function-schema-returns normalized))))
(test-equal #f (return-annotation (car (function-schema-returns normalized))))
(test-assert (not (argument-is-write?
                   (car (arguments-pre-tensor-options-kwarg-only
                         (function-schema-arguments normalized))))))
(test-equal "first" (return-name (car (function-schema-returns named-normalized))))
(test-equal "second" (return-name (cadr (function-schema-returns named-normalized))))
(test-equal tensor (return-type (car (function-schema-returns named-normalized))))

;; Mutable inputs become synthetic returns unless already represented.
(define mutable-inputs
  (make-arguments '() #f '()
                  (list (arg "buffer" tensor #f 'write #t))
                  #f '() '()))
(define mutable-schema
  (make-function-schema (parse-operator-name "mutate.Tensor")
                        mutable-inputs
                        (list (make-return "value" tensor #f))))
(define mutable-signature
  (function-schema-signature mutable-schema #:keep-return-names #t))
(test-equal 2 (length (function-schema-returns mutable-signature)))
(test-equal "buffer_out"
  (return-name (cadr (function-schema-returns mutable-signature))))
(test-equal #f
  (return-name
   (cadr (function-schema-returns
          (function-schema-signature mutable-schema)))))

;; Out normalization drops the explicit out argument. Mutable arguments retain
;; is_write in the signature and therefore do not collapse accidentally.
(define family-returns (list (make-return "self" tensor #f)))
(define family-functional
  (make-function-schema
   (parse-operator-name "family.Tensor")
   (make-arguments '() (arg "self" tensor #f #f #f) '()
                   (list (arg "other" tensor #f #f #f)) #f '() '())
   family-returns))
(define family-inplace
  (make-function-schema
   (parse-operator-name "family_.Tensor")
   (make-arguments '() (arg "self" tensor #f 'write #t) '()
                   (list (arg "other" tensor #f #f #f)) #f '() '())
   family-returns))
(define family-mutable
  (make-function-schema
   (parse-operator-name "family.mutable")
   (make-arguments '() (arg "self" tensor #f #f #f) '()
                   (list (arg "other" tensor #f 'write #t)) #f '() '())
   family-returns))
(define family-out
  (make-function-schema
   (parse-operator-name "family.out")
   (make-arguments '() (arg "self" tensor #f #f #f) '()
                   (list (arg "other" tensor #f #f #f)) #f '()
                   (list (arg "out" tensor #f 'write #t)))
   family-returns))
(define family-signature (function-schema-signature family-functional
                                                    #:keep-return-names #t))
(test-assert (equal? family-signature
                     (function-schema-signature family-out
                                                 #:keep-return-names #t)))
(define family-mutable-signature
  (function-schema-signature family-mutable #:keep-return-names #t))
(test-equal 2 (length (function-schema-returns family-mutable-signature)))
(test-equal "other_out"
  (return-name (cadr (function-schema-returns family-mutable-signature))))
(test-equal tensor
  (return-type (cadr (function-schema-returns family-mutable-signature))))
(test-equal #f
  (return-annotation (cadr (function-schema-returns family-mutable-signature))))
(test-assert (argument-is-write?
              (car (arguments-pre-tensor-options-kwarg-only
                    (function-schema-arguments
                     (function-schema-signature family-mutable))))))
(test-equal #f
  (return-name
   (cadr (function-schema-returns
          (function-schema-signature family-mutable)))))

;; Structural equality and deterministic hashing make signatures grouping keys.
(define equivalent-schema
  (make-function-schema (parse-operator-name "add.Other")
                        (make-arguments
                         (list (arg "self" tensor #f #f #f))
                         #f '() '() #f '() '())
                        (list (make-return "result" tensor #f))))
(define equivalent-schema-2
  (make-function-schema (parse-operator-name "add.Tensor")
                        (make-arguments
                         (list (arg "self" tensor #f 'different #f))
                         #f '() '() #f '() '())
                        (list (make-return "result" tensor 'different))))
(test-assert (equal? (function-schema-signature equivalent-schema)
                     (function-schema-signature equivalent-schema-2)))
(test-equal (function-schema-hash (function-schema-signature equivalent-schema))
  (function-schema-hash (function-schema-signature equivalent-schema-2)))
(test-assert (not (equal? (function-schema-signature equivalent-schema
                                                     #:keep-return-names #t)
                          (function-schema-signature equivalent-schema-2))))

(let ((runner (test-runner-current)))
  (test-end)
  (exit (if (zero? (test-runner-fail-count runner)) 0 1)))
