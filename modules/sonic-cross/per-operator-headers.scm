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

;; Per-operator header generation (torchgen gen.py gen_per_operator_headers,
;; gated by --per-operator-headers).  Splits the aggregated Function/Operator/
;; Native/Meta headers into one header per operator root name under ATen/ops,
;; plus the include-only aggregate shims and per-dispatch-key dispatch headers.
;; The templates are embedded verbatim from the frozen commit and the rendered
;; text is the sole output.

(define-module (sonic-cross per-operator-headers)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-13)
  #:use-module (sonic-cross core-ir)
  #:use-module (sonic-cross native-function)
  #:use-module (sonic-cross code-template)
  #:use-module (sonic-cross orchestration)
  #:use-module (sonic-cross emitter)
  #:use-module (sonic-cross operators)
  #:use-module (sonic-cross register-dispatch-key)
  #:export (render-per-operator-headers))

(define %generator-path "torchgen/gen.py")

;; ---------------------------------------------------------------------------
;; templates (verbatim from aten/src/ATen/templates at the frozen commit)
;; ---------------------------------------------------------------------------

(define %per-op-operator-template
  (string-append
   "#pragma once\n"
   "\n"
   "// ${generated_comment}\n"
   "\n"
   "#include <string_view>\n"
   "#include <tuple>\n"
   "#include <vector>\n"
   "\n"
   "// Forward declarations of any types needed in the operator signatures.\n"
   "// We can't directly include these classes because it will cause circular include dependencies.\n"
   "// This file is included by TensorBody.h, which defines the Tensor class.\n"
   "#include <ATen/core/ATen_fwd.h>\n"
   "\n"
   "namespace at {\n"
   "namespace _ops {\n"
   "\n"
   "${declarations}\n"
   "\n"
   "}} // namespace at::_ops\n"))

(define %per-op-function-template
  (string-append
   "#pragma once\n"
   "\n"
   "// ${generated_comment}\n"
   "\n"
   "#include <ATen/Context.h>\n"
   "#include <ATen/DeviceGuard.h>\n"
   "#include <ATen/TensorUtils.h>\n"
   "#include <ATen/TracerMode.h>\n"
   "#include <ATen/core/Generator.h>\n"
   "#include <ATen/core/Reduction.h>\n"
   "#include <ATen/core/Tensor.h>\n"
   "#include <c10/core/Scalar.h>\n"
   "#include <c10/core/Storage.h>\n"
   "#include <c10/core/TensorOptions.h>\n"
   "#include <c10/util/Deprecated.h>\n"
   "#include <optional>\n"
   "#include <string_view>\n"
   "\n"
   "${static_dispatch_ops_headers}\n"
   "\n"
   "${operator_includes}\n"
   "\n"
   "namespace at {\n"
   "\n"
   "${function_definitions}\n"
   "\n"
   "}\n"))

(define %per-op-meta-template
  (string-append
   "#pragma once\n"
   "\n"
   "// ${generated_comment}\n"
   "\n"
   "#include <c10/core/Scalar.h>\n"
   "#include <c10/core/Storage.h>\n"
   "#include <c10/core/TensorOptions.h>\n"
   "#include <c10/util/Deprecated.h>\n"
   "#include <optional>\n"
   "#include <c10/core/QScheme.h>\n"
   "#include <ATen/core/Reduction.h>\n"
   "#include <ATen/TensorIterator.h>\n"
   "#include <ATen/TensorMeta.h>\n"
   "#include <tuple>\n"
   "#include <vector>\n"
   "\n"
   "namespace at {\n"
   "namespace meta {\n"
   "\n"
   "${meta_function_declarations}\n"
   "\n"
   "} // namespace native\n"
   "} // namespace at\n"))

(define %per-op-native-template
  (string-append
   "#pragma once\n"
   "\n"
   "// ${generated_comment}\n"
   "\n"
   "#include <c10/core/Scalar.h>\n"
   "#include <c10/core/Storage.h>\n"
   "#include <c10/core/TensorOptions.h>\n"
   "#include <c10/util/Deprecated.h>\n"
   "#include <optional>\n"
   "#include <c10/core/QScheme.h>\n"
   "#include <ATen/core/Reduction.h>\n"
   "#include <ATen/core/Tensor.h>\n"
   "#include <tuple>\n"
   "#include <vector>\n"
   "${extra_includes}\n"
   "\n"
   "${native_function_declarations}\n"))

(define %per-op-dispatch-template
  (string-append
   "#pragma once\n"
   "// ${generated_comment}\n"
   "\n"
   "// NB: The implementing C++ file is RegisterDispatchKey.cpp\n"
   "\n"
   "// The only #includes we need are for custom classes that have defaults in the C++ API\n"
   "#include <c10/core/MemoryFormat.h>\n"
   "#include <c10/core/Scalar.h>\n"
   "#include <ATen/core/Reduction.h>\n"
   "\n"
   "// Forward declarations of any types needed in the operator signatures.\n"
   "// We can't directly include these classes because it will cause circular include dependencies.\n"
   "// This file is included by TensorBody.h, which defines the Tensor class.\n"
   "#include <ATen/core/ATen_fwd.h>\n"
   "\n"
   "namespace at {\n"
   "\n"
   "namespace ${dispatch_namespace} {\n"
   "\n"
   "${dispatch_namespaced_declarations}\n"
   "\n"
   "} // namespace ${dispatch_namespace}\n"
   "} // namespace at\n"))

;; ---------------------------------------------------------------------------
;; grouping
;; ---------------------------------------------------------------------------

(define (group-items-by-root items root-fn)
  ;; returns ((root . items) ...) preserving first-appearance order of roots.
  (let ((order '()) (tbl (make-hash-table)))
    (for-each
     (lambda (item)
       (let* ((root (root-fn item))
              (existing (hash-ref tbl root #f)))
         (unless existing (set! order (append order (list root))))
         (hash-set! tbl root (append (or existing '()) (list item)))))
     items)
    (map (lambda (root) (cons root (hash-ref tbl root))) order)))

(define (functions-by-root-name functions)
  (group-items-by-root functions native-function-root-name))

(define (grouped-functions-by-root-name grouped)
  (group-items-by-root grouped item-root-name))

(define (structured-groups grouped-functions)
  (filter (lambda (g)
            (and (native-functions-group? g)
                 (native-functions-group-structured? g)))
          grouped-functions))

;; ---------------------------------------------------------------------------
;; per-operator header renderers (one file each)
;; ---------------------------------------------------------------------------

(define (render-per-op-ops-h functions)
  (code-template-substitute
   %per-op-operator-template
   (lambda (key)
     (cond
      ((string=? key "generated_comment")
       (string-append "@generated by " %generator-path " from Operator.h"))
      ((string=? key "declarations") (map operator-declaration functions))
      (else (error 'unknown-per-op-operator-key key))))))

(define (render-per-op-function-h name functions)
  (code-template-substitute
   %per-op-function-template
   (lambda (key)
     (cond
      ((string=? key "generated_comment")
       (string-append "@generated by " %generator-path " from Function.h"))
      ((string=? key "static_dispatch_ops_headers") '())
      ((string=? key "operator_includes")
       (string-append "#include <ATen/ops/" name "_ops.h>"))
      ((string=? key "function_definitions") (map compute-function functions))
      (else (error 'unknown-per-op-function-key key))))))

(define (render-per-op-meta-h structured-functions)
  (code-template-substitute
   %per-op-meta-template
   (lambda (key)
     (cond
      ((string=? key "generated_comment")
       (string-append "@generated by " %generator-path " from NativeMetaFunction.h"))
      ((string=? key "meta_function_declarations")
       (filter-map compute-meta-function-declaration structured-functions))
      (else (error 'unknown-per-op-meta-key key))))))

(define (render-per-op-native-h name is-structured grouped-functions indices)
  (code-template-substitute
   %per-op-native-template
   (lambda (key)
     (cond
      ((string=? key "generated_comment")
       (string-append "@generated by " %generator-path " from NativeFunction.h"))
      ((string=? key "extra_includes")
       (if is-structured
           (string-append "#include <ATen/ops/" name "_meta.h>")
           '()))
      ((string=? key "native_function_declarations")
       (get-native-function-declarations
        grouped-functions (ordered-backend-indices indices)))
      (else (error 'unknown-per-op-native-key key))))))

(define (render-per-op-dispatch-h dispatch-key declarations)
  (code-template-substitute
   %per-op-dispatch-template
   (lambda (key)
     (cond
      ((string=? key "generated_comment")
       (string-append "@generated by " %generator-path " from DispatchKeyFunction.h"))
      ((string=? key "dispatch_namespace") (string-downcase dispatch-key))
      ((string=? key "dispatch_namespaced_declarations") declarations)
      (else (error 'unknown-per-op-dispatch-key key))))))

;; ---------------------------------------------------------------------------
;; aggregate shims (include-only)
;; ---------------------------------------------------------------------------

(define (sorted-ops-includes fbrn suffix)
  ;; fbrn :: ((root . functions) ...) — include lines for every root, sorted.
  (map (lambda (name)
         (string-append "#include <ATen/ops/" name suffix ".h>"))
       (sort (map car fbrn) string<?)))

;; ---------------------------------------------------------------------------
;; top-level
;; ---------------------------------------------------------------------------

(define (render-per-operator-headers functions grouped indices)
  ;; returns ((label filename content) ...) in generation order, where label is
  ;; ops | cpu | cuda.  Mirrors gen_per_operator_headers.
  (let* ((fbrn (functions-by-root-name functions))
         (gbrn (grouped-functions-by-root-name grouped))
         (names (map car fbrn))
         (out '()))
    (define (emit! label filename content)
      (set! out (cons (list label filename content) out)))

    ;; per-operator ops/{name}[_.*].h
    (for-each
     (lambda (entry)
       (let* ((name (car entry))
              (fns (cdr entry))
              (grouped-fns (or (assoc-ref gbrn name) '()))
              (structured-fns (structured-groups grouped-fns))
              (is-structured (pair? structured-fns)))
         (emit! 'ops (string-append name "_ops.h")
                (render-per-op-ops-h fns))
         (emit! 'ops (string-append name ".h")
                (render-per-op-function-h name fns))
         (when is-structured
           (emit! 'ops (string-append name "_meta.h")
                  (render-per-op-meta-h structured-fns)))
         (emit! 'ops (string-append name "_native.h")
                (render-per-op-native-h name is-structured grouped-fns indices))))
     fbrn)

    ;; aggregate shims (cpu)
    (emit! 'cpu "Functions.h"
           (render-functions-h functions
                              #:includes (sorted-ops-includes fbrn "")
                              #:declarations '()))
    (emit! 'cpu "Operators.h"
           (render-operators-h functions
                               #:includes (sorted-ops-includes fbrn "_ops")
                               #:declarations '()))
    (emit! 'cpu "NativeMetaFunctions.h"
           (render-native-meta-functions-h
            (get-structured-native-functions grouped)
            #:includes (sorted-ops-includes fbrn "_meta")
            #:declarations '()))
    (emit! 'cpu "NativeFunctions.h"
           (render-native-functions-h grouped indices
                                      #:includes (sorted-ops-includes fbrn "_native")
                                      #:declarations '()))

    ;; per-dispatch-key dispatch headers + {key}Functions.h / _inl.h
    (for-each
     (lambda (dispatch-key)
       (let ((backend-index (make-backend-index dispatch-key indices))
             (dispatch-names '()))
         (for-each
          (lambda (entry)
            (let* ((name (car entry))
                   (grouped-fns (or (assoc-ref gbrn name) '()))
                   (declarations
                    (append-map
                     (lambda (item)
                       (gen-dispatch item dispatch-key backend-index
                                     'namespaced-declaration))
                     grouped-fns)))
              (unless (null? declarations)
                (set! dispatch-names (cons name dispatch-names))
                (emit! 'ops
                       (string-append name "_" (string-downcase dispatch-key)
                                      "_dispatch.h")
                       (render-per-op-dispatch-h dispatch-key declarations)))))
          fbrn)
         (let ((fm-label (if (is-cuda-dispatch-key? dispatch-key) 'cuda 'cpu)))
           (emit! fm-label (string-append dispatch-key "Functions.h")
                  (render-dispatch-key-functions-h dispatch-key))
           (emit! fm-label (string-append dispatch-key "Functions_inl.h")
                  (render-dispatch-key-functions-inl-h
                   grouped dispatch-key backend-index
                   #:includes
                   (map (lambda (name)
                          (string-append "#include <ATen/ops/" name "_"
                                         (string-downcase dispatch-key)
                                         "_dispatch.h>"))
                        (sort dispatch-names string<?))
                   #:declarations '())))))
     functions-keys)

    ;; MethodOperators.h (cpu) — include-only over method-variant names.
    ;; NB: the oracle sorts the *formatted include strings* (not the root
    ;; names), so a root name that is a prefix of another (e.g. `_to_sparse`
    ;; vs `_to_sparse_bsc`) sorts by the full `#include ..._ops.h>` string.
    (emit! 'cpu "MethodOperators.h"
           (render-method-operators-h
            functions
            #:includes
            (sort
             (map (lambda (name)
                    (string-append "#include <ATen/ops/" name "_ops.h>"))
                  (map car
                       (filter (lambda (entry)
                                 (any (lambda (fn)
                                        (member "method"
                                                (native-function-variants fn)))
                                      (cdr entry)))
                               fbrn)))
             string<?)
            #:declarations '()))

    (reverse out)))
