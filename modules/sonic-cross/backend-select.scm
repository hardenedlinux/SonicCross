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

;; G20: RegisterBackendSelect.cpp (torchgen gen.py ComputeBackendSelect).
;;
;;   BackendSelect kernels provide specialized computation of the dispatch key
;;   for signatures that cannot be auto-computed by templating: functions whose
;;   schema carries tensor-options (dtype/layout/device/pin_memory), excluding
;;   the `*_like` and `new_*` families.
;;
;;   The nop SelectiveBuilder used by the frozen differential oracle selects
;;   every native function, so `needs_backend_select` reduces to the name and
;;   tensor-options filters.
;;
;; Byte-identical against frozen torchgen 41ffbc4a994e058af9fe00ed5caba73fc1033359.

(define-module (sonic-cross backend-select)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-13)
  #:use-module (sonic-cross core-ir)
  #:use-module (sonic-cross native-function)
  #:use-module (sonic-cross dispatcher)
  #:use-module (sonic-cross generated-functions)
  #:use-module ((sonic-cross api) #:prefix api:)
  #:use-module (sonic-cross code-template)
  #:export (needs-backend-select?
            backend-select-definition
            backend-select-registration
            render-backend-select-cpp))

(define %generator-path "torchgen/gen.py")

(define (root-name f)
  ;; NativeFunction.root_name == func.name.name.base.
  (base-operator-name-base
   (operator-name-base (function-schema-name (native-function-func f)))))

;; ---------------------------------------------------------------------------
;; Template (verbatim from aten/src/ATen/templates at the frozen commit)
;; ---------------------------------------------------------------------------

(define %backend-select-template
  (string-append
   "// We register ops with a higher priority dispatch key (BackendSelect) than the usual backend-specific keys (e.g. CPU)\n"
   "// which makes calls to the factory functions dispatch to here.\n"
   "// We then 'manually' compute a lower-priority to re-dispatch to (e.g. CPU) to get to the eventually correct backend.\n"
   "// ${generated_comment}\n"
   "\n"
   "#define TORCH_ASSERT_ONLY_METHOD_OPERATORS\n"
   "#include <ATen/core/Tensor.h>\n"
   "#include <ATen/core/dispatch/DispatchKeyExtractor.h>\n"
   "#include <torch/library.h>\n"
   "\n"
   "#ifndef AT_PER_OPERATOR_HEADERS\n"
   "#include <ATen/Operators.h>\n"
   "#else\n"
   "\n"
   "${ops_headers}\n"
   "#endif\n"
   "\n"
   "namespace at {\n"
   "\n"
   "namespace {\n"
   "\n"
   "${backend_select_method_definitions}\n"
   "\n"
   "TORCH_LIBRARY_IMPL(aten, BackendSelect, m) {\n"
   "  ${backend_select_function_registrations};\n"
   "}\n"
   "\n"
   "} // namespace\n"
   "} // at\n"))

;; ---------------------------------------------------------------------------
;; needs_backend_select (torchgen gen.py)
;; ---------------------------------------------------------------------------

(define (needs-backend-select? f)
  ;; Mirrors torchgen gen.py needs_backend_select(): the nop selector selects
  ;; every native function, so only the name and tensor-options filters remain.
  (let* ((func (native-function-func f))
         (name (base-operator-spelling func)))
    (and (not (string-suffix? "_like" name))
         (not (string-prefix? "new_" name))
         (arguments-tensor-options (function-schema-arguments func)))))

;; ---------------------------------------------------------------------------
;; ComputeBackendSelect
;; ---------------------------------------------------------------------------

(define (native-tensor-arg-names func)
  ;; native_sig.arguments() = native.arguments(func, symint=True), the bindings
  ;; over non_out + out.  A binding counts when its .argument is a plain
  ;; Argument whose type is tensor-like; TensorOptionsArguments bindings
  ;; (dtype/layout/device/pin_memory) are never tensor-like, and the flattened
  ;; `arguments-all` (self-as-Argument + tensor-options-as-4-Arguments) yields
  ;; exactly the native order, so a tensor-like filter over it is faithful.
  (map argument-name
       (filter (lambda (a) (type-is-tensor-like? (argument-type a)))
               (arguments-all (function-schema-arguments func)))))

(define (backend-select-definition f)
  ;; Mirrors torchgen gen.py ComputeBackendSelect.__call__ (DEFINITION branch).
  ;; The multi-line compute_dk is emitted as a single string after the "  "
  ;; indent, so its 2nd and 3rd lines carry no leading whitespace -- faithfully
  ;; reproducing the frozen f-string splicing.
  (let* ((func (native-function-func f))
         (name (api:native-name func))
         (use-const-ref (native-function-use-const-ref-for-mutable-tensors? f))
         (use-ilistref (or (native-function-structured? f)
                           (native-function-structured-delegate f)))
         (defn (dispatcher-signature-defn func use-const-ref use-ilistref
                                          name #f))
         (dispatch-key "c10::computeDispatchKey(dtype, layout, device)")
         (tensor-args (native-tensor-arg-names func))
         (compute-dk
          (if (null? tensor-args)
              (string-append "DispatchKeySet _dk = c10::DispatchKeySet("
                             dispatch-key ");")
              (string-append
               "DispatchKeySet _dk_set = c10::DispatchKeySet(" dispatch-key
               ") | c10::detail::multi_dispatch_key_set("
               (string-join tensor-args ", ") ");\n"
               "DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);\n"
               "DispatchKeySet _dk = c10::impl::computeDispatchKeySet(_dk_set, _dk_mask);")))
         (redispatch-args
          (string-join (map argument-name
                            (arguments-all (function-schema-arguments func)))
                       ", "))
         (unambig (unambiguous-name func)))
    (string-append
     "// aten::" (function-schema->string func) "\n"
     "C10_ALWAYS_INLINE\n"
     defn " {\n"
     "  " compute-dk "\n"
     "  return at::_ops::" unambig "::redispatch(\n"
     "      _dk, " redispatch-args ");\n"
     "}\n")))

(define (backend-select-registration f)
  ;; Mirrors torchgen gen.py ComputeBackendSelect.__call__ (REGISTRATION
  ;; branch): `m.impl("aten::{str(OperatorName)}", TORCH_FN({native.name}));`.
  (let ((func (native-function-func f)))
    (string-append
     "m.impl(\"aten::" (operator-name->string (function-schema-name func))
     "\", TORCH_FN(" (api:native-name func) "));")))

;; ---------------------------------------------------------------------------
;; RegisterBackendSelect.cpp
;; ---------------------------------------------------------------------------

(define (render-backend-select-cpp functions)
  (let* ((relevant (filter needs-backend-select? functions))
         (ops-headers (map (lambda (f)
                             (string-append "#include <ATen/ops/"
                                            (root-name f) "_ops.h>"))
                           relevant))
         (definitions (map backend-select-definition relevant))
         (registrations (map backend-select-registration relevant)))
    (code-template-substitute
     %backend-select-template
     (lambda (key)
       (cond
        ((string=? key "generated_comment")
         (string-append "@generated by " %generator-path
                        " from RegisterBackendSelect.cpp"))
        ((string=? key "ops_headers") ops-headers)
        ((string=? key "backend_select_method_definitions") definitions)
        ((string=? key "backend_select_function_registrations") registrations)
        (else (error 'unknown-template-key key)))))))
