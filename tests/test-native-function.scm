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
             (sonic-cross native-function))

(define (parse-one text)
  (call-with-values
      (lambda () (native-function-from-yaml (yaml-load-string text)))
    (lambda (function dispatch) (cons function dispatch))))

(define (nf text) (car (parse-one text)))
(define (dispatch-of text) (cdr (parse-one text)))
(define (metadata-of text key)
  (cdar (cdr (assoc key (dispatch-of text)))))

(define basic-yaml
  "func: add.Tensor(Tensor self, Tensor other) -> Tensor\ndispatch:\n  CPU: add_cpu\n")

(test-begin "sonic-cross native function")

(let ((plain (namespace-helper-from-namespaced-entity "add.Tensor" 3))
      (aten (namespace-helper-from-namespaced-entity "aten::add.Tensor" 3))
      (kernel (namespace-helper-from-namespaced-entity "at::native::add_cpu" 3)))
  (test-equal "" (namespace-helper-namespace plain))
  (test-equal "add.Tensor" (namespace-helper-entity-name plain))
  (test-equal "aten" (namespace-helper-namespace aten))
  (test-equal "add.Tensor" (namespace-helper-entity-name aten))
  (test-equal "at::native" (namespace-helper-namespace kernel))
  (test-equal "at::native" (namespace-helper-get-cpp-namespace kernel "fallback"))
  (test-equal "fallback" (namespace-helper-get-cpp-namespace plain "fallback")))
(test-error (namespace-helper-from-namespaced-entity "a::b::c::d" 2))
(for-each (lambda (key) (test-equal key (dispatch-key-parse key))) dispatch-keys)
(test-error (dispatch-key-parse "NotAKey"))

(let* ((pair (parse-one basic-yaml))
       (function (car pair))
       (dispatch (cdr pair))
       (metadata (metadata-of basic-yaml "CPU")))
  (test-assert (native-function? function))
  (test-equal "aten" (native-function-namespace function))
  (test-equal '("function") (native-function-variants function))
  (test-equal "add_cpu" (backend-metadata-kernel metadata))
  (test-equal "at::native" (backend-metadata-cpp-namespace metadata))
  (test-assert (not (backend-metadata-structured metadata)))
  (test-assert (not (backend-metadata-supports-symint? metadata))))

(let ((function (nf "func: aten::add.Tensor(Tensor self) -> Tensor\ndispatch:\n  CPU: at::native::add_cpu_symint\n")))
  (test-equal "aten" (native-function-namespace function))
  (test-assert (backend-metadata-supports-symint?
                (metadata-of
                 "func: aten::add.Tensor(Tensor self) -> Tensor\ndispatch:\n  CPU: at::native::add_cpu_symint\n"
                 "CPU")))

(let ((function (nf "func: add.Tensor(Tensor self) -> Tensor\nvariants: function, method\ndispatch:\n  CPU, CUDA: add_cpu\n  CPU: add_cpu_later\n")))
  (test-equal '("function" "method") (native-function-variants function))
  (test-equal "add_cpu_later"
    (backend-metadata-kernel
     (metadata-of
      "func: add.Tensor(Tensor self) -> Tensor\nvariants: function, method\ndispatch:\n  CPU, CUDA: add_cpu\n  CPU: add_cpu_later\n"
      "CPU"))))
(test-error (nf "func: add.Tensor(Tensor self) -> Tensor\nvariants: nope\ndispatch:\n  CPU: add_cpu\n"))

(let ((function
       (nf "func: add.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)\nstructured: true\ndispatch:\n  CPU: add_out_cpu\n")))
  (test-assert (native-function-structured? function))
  (test-assert (backend-metadata-structured
                (metadata-of
                 "func: add.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)\nstructured: true\ndispatch:\n  CPU: add_out_cpu\n"
                 "CPU"))))
  (test-assert (native-function-is-abstract? function)))

(let ((function (nf "func: add.Tensor(Tensor self) -> Tensor\ndispatch:\n  CompositeImplicitAutograd: add_composite\n")))
  (test-assert (native-function-has-composite-implicit-autograd-kernel? function))
  (test-assert (not (native-function-is-abstract? function))))
(let ((function (nf "func: add.Tensor(Tensor self) -> Tensor\ndispatch:\n  CompositeImplicitAutogradNestedTensor: add_nested\n")))
  (test-assert (native-function-has-composite-implicit-autograd-nested-tensor-kernel? function)))
(let ((function (nf "func: add.Tensor(Tensor self) -> Tensor\ndispatch:\n  CompositeExplicitAutograd: add_explicit\n")))
  (test-assert (native-function-has-composite-explicit-autograd-kernel? function))
  (test-assert (native-function-is-abstract? function)))
(let ((function (nf "func: add.Tensor(Tensor self) -> Tensor\ndispatch:\n  CompositeExplicitAutogradNonFunctional: add_explicit\n")))
  (test-assert (native-function-has-composite-explicit-autograd-non-functional-kernel? function)))

(let ((dispatch (dispatch-of "func: add.Tensor(Tensor self) -> Tensor\n")))
  (test-assert (assoc "CompositeImplicitAutograd" dispatch)))
(let ((dispatch (dispatch-of "func: add() -> Tensor\n")))
  (test-assert (assoc "CompositeImplicitAutograd" dispatch)))
(test-error (nf "func: empty(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False) -> Tensor\n"))
(test-error (nf "func: empty_like(Tensor self) -> Tensor\n"))
(test-error (nf "func: new_empty() -> Tensor\n"))

(let ((function (nf "func: add.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)\ntags: nondeterministic_seeded\nautogen: add.Tensor, add.Other\ndispatch:\n  CPU: add_out\n")))
  (test-assert (member "nondeterministic_seeded" (native-function-tags function)))
  (test-assert (member "out" (native-function-tags function)))
  (test-assert (member "pt2_compliant_tag" (native-function-tags function)))
  (test-equal 2 (length (native-function-autogen function))))
(let ((function (nf "func: add_(Tensor(a!) self) -> Tensor(a!)\ndispatch:\n  CPU: add_\n")))
  (test-assert (member "inplace" (native-function-tags function))))
(let ((function (nf "func: rand(Tensor self) -> Tensor\ndispatch:\n  CPU: rand_cpu\n")))
  (test-assert (member "nondeterministic_seeded" (native-function-tags function))))
(test-error (nf "func: add.Tensor(Tensor self) -> Tensor\ntags: unknown\ndispatch:\n  CPU: add_cpu\n"))

(test-error (nf "func: add.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)\nvariants: method\ndispatch:\n  CPU: add_out\n"))
(test-error (nf "func: add.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)\nuse_const_ref_for_mutable_tensors: true\ndispatch:\n  CPU: add_out\n"))
(test-error (nf "func: add.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)\nstructured: true\ndevice_guard: false\ndispatch:\n  CPU: add_out\n"))
(test-error (nf "func: add.Tensor(Tensor self) -> Tensor\nstructured_delegate: other::op\ndispatch:\n  CPU: add_cpu\n"))
(test-error (nf "func: add.Tensor(Tensor self) -> Tensor\nstructured: true\nstructured_delegate: other\ndispatch:\n  CPU: add_cpu\n"))
(test-error (nf "func: add.Tensor(Tensor self) -> Tensor\nstructured_inherits: base::foo\ndispatch:\n  CPU: add_cpu\n"))
(test-error (nf "func: add.Tensor(Tensor self) -> Tensor\npython_module: foo\nvariants: method\ndispatch:\n  CPU: add_cpu\n"))
(test-error (nf "func: add.Tensor(Tensor self) -> Tensor\nprecomputed: []\ndispatch:\n  CPU: add_cpu\n"))
(test-error (nf "func: add.Tensor(Tensor self, Scalar alpha=1) -> Tensor\ncpp_no_default_args: self\ndispatch:\n  CPU: add_cpu\n"))
(test-error (nf "func: _foreach_add(Tensor self) -> Tensor\ndispatch:\n  CPU: add_cpu\n"))
(test-error (nf "func: add.Tensor(Tensor self) -> Tensor\nunknown_field: x\ndispatch:\n  CPU: add_cpu\n"))

(let* ((functional (nf "func: add.Tensor(Tensor self, Tensor other) -> Tensor\ndispatch:\n  CPU: add_cpu\n"))
       (out (nf "func: add.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)\ndispatch:\n  CPU: add_out\n"))
       (group (native-functions-group-from-dict
               (list (cons "functional" functional) (cons "out" out)))))
  (test-assert (native-functions-group? group))
  (test-equal 2 (length (native-functions-group-functions group)))
  (test-assert (not (native-functions-group-structured? group)))
  (test-assert (native-functions-group-signature group)))
(test-equal #f
  (native-functions-group-from-dict
   (list (cons "functional" (nf basic-yaml)))))
(test-equal #f
  (native-functions-group-from-dict
   (list (cons "functional" (nf basic-yaml))
         (cons "inplace" (nf "func: add_(Tensor(a!) self, Tensor other) -> Tensor(a!)\ndispatch:\n  CPU: add_cpu\n")))))
(test-error (native-functions-group-from-dict '()))

(let ((runner (test-runner-current)))
  (test-end)
  (exit (if (zero? (test-runner-fail-count runner)) 0 1)))
