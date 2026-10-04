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

;; G16: Operators.h + Operators.cpp (torchgen gen.py ComputeOperators).
;;
;;   Operators.h   ComputeOperators(Target.DECLARATION) over
;;                 non_method_native_functions, with
;;                 Operators_includes = ["#include <ATen/MethodOperators.h>"].
;;   Operators.cpp ComputeOperators(Target.DEFINITION) over native_functions,
;;                 sharded x5 on string_stable_hash(root_name) % 5 (plus the
;;                 discarded "Everything" shard).
;;
;; Byte-identical against frozen torchgen 41ffbc4a994e058af9fe00ed5caba73fc1033359.

(define-module (sonic-cross operators)
  #:use-module (ice-9 optargs)
  #:use-module (srfi srfi-1)
  #:use-module (sonic-cross core-ir)
  #:use-module (sonic-cross native-function)
  #:use-module (sonic-cross dispatcher)
  #:use-module (sonic-cross generated-functions)
  #:use-module (sonic-cross emitter)
  #:use-module (sonic-cross code-template)
  #:use-module (sonic-cross file-manager)
  #:export (operator-declaration
            operator-definition
            render-operators-h
            render-operators-cpp-shards
            render-operators-cpp-files))

(define %generator-path "torchgen/gen.py")

(define (root-name f)
  ;; NativeFunction.root_name == func.name.name.base.
  (base-operator-name-base
   (operator-name-base (function-schema-name (native-function-func f)))))

;; ---------------------------------------------------------------------------
;; Templates (verbatim from aten/src/ATen/templates at the frozen commit)
;; ---------------------------------------------------------------------------

(define %operators-h-template
  "#pragma once\n\n// ${generated_comment}\n\n#ifdef TORCH_ASSERT_NO_OPERATORS\n#error This change adds a dependency on native_functions.yaml,             \\\n  meaning the file will need to be re-compiled every time an operator      \\\n  is changed or added. Consider if your change would be better placed in   \\\n  another file, or if a more specific header might achieve the same goal.  \\\n  See NOTE: [Tensor vs. TensorBase]\n#endif\n\n#if defined(AT_PER_OPERATOR_HEADERS) && defined(TORCH_ASSERT_ONLY_METHOD_OPERATORS)\n#error This change adds a dependency on all pytorch operators, meaning the     \\\n  file will need to be re-compiled every time an operator is changed or added. \\\n  Consider including a specific operator from <ATen/ops/{my_operator}_ops.h>   \\\n  and see NOTE [TORCH_ASSERT_ONLY_METHOD_OPERATORS].\n#endif\n\n#include <c10/core/SymInt.h>\n#include <c10/core/SymIntArrayRef.h>\n#include <c10/core/Scalar.h>\n#include <c10/core/TensorOptions.h>\n#include <c10/core/QScheme.h>\n#include <c10/util/OptionalArrayRef.h>\n#include <tuple>\n#include <vector>\n\n${Operators_includes}\n\n// Extension writers: do you write wrapper functions? Are you frustrated with\n// resolving overloads of operators? Are you frustrated with dealing with\n// pointer-to-methods and resolving overloads of pointer-to-methods?? Look no\n// further, this is the utility for you.\n//\n// Given an operator schema: aten::op.overload(...\n//\n// Use ATEN_FN2(op, overload) to get a *function* version of the operator\n// that is guaranteed to not be overloaded. This means that you can safely\n// decltype(&ATEN_FN2(op, overload)) it. NB: the 2 means this macro takes 2 args.\n//\n// Given an operator schema without an overload name: aten::op(...\n//\n// Use ATEN_FN(op) to get an unambiguous *function* version of the operator.\n//\n// There is some interesting behavior for out= operations.\n// ATEN_FN2(sin, out) gives a function that is *faithful* to the schema;\n// that is, the order of arguments is exactly what it looks like in the schema.\n\n#define ATEN_FN2(op_name, overload) at::_ops::op_name##_##overload::call\n#define ATEN_FN(op_name) at::_ops::op_name::call\n\n// Separately, ATEN_OP(op) and ATEN_OP2(op, overload) define a class containing compile-time\n// metadata about a given aten operator.\n// Notable data on the class includes:\n// - ATEN_OP2(add, Tensor)::name // returns the string name: \"add\"\n// - ATEN_OP2(add, Tensor)::overload_name // returns the string overload name: \"Tensor\"\n// - ATEN_OP2(add, Tensor)::schema // returns the C++ schema type: at::Tensor (const at::Tensor &, const at::Tensor &, const at::Scalar &)\n// - ATEN_OP2(add, Tensor)::schema_str // returns the string jit type: \"add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor\"\n\n#define ATEN_OP2(op_name, overload) at::_ops::op_name##_##overload\n#define ATEN_OP(op_name) at::_ops::op_name\n\n// WARNING: Please do not call any of the ops in the _ops namespace directly.\n// Use the ATEN_FN macros. We do not guarantee stability of the naming\n// scheme for the functions in at::_ops\n\n// See Note [The ATen Operators API] for details of the at::_ops namespace\n\nnamespace at {\nnamespace _ops {\n${Operators_declarations}\n} // namespace _ops\n} // namespace at\n")

(define %operators-cpp-template
  "#include <ATen/Tensor.h>\n#include <ATen/core/dispatch/Dispatcher.h>\n\n// ${generated_comment}\n// NOTE See [Sharded File] comment in VariableType\n\n#ifndef AT_PER_OPERATOR_HEADERS\n#include <ATen/Operators.h>\n#else\n${operator_headers}\n#endif\n\n${static_dispatch_extra_headers}\n\nnamespace at { namespace _ops {\n\n${definitions}\n\n}} // namespace at::_ops\n")

;; ---------------------------------------------------------------------------
;; ComputeOperators, DECLARATION branch
;; ---------------------------------------------------------------------------

(define (operator-declaration f)
  ;; Mirrors torchgen gen.py ComputeOperators.__call__ with
  ;; static_dispatch_backend_indices == [] (the default flag set).
  (let* ((func (native-function-func f))
         (name (unambiguous-name func))
         (use-const-ref (native-function-use-const-ref-for-mutable-tensors? f))
         (use-ilistref (or (native-function-structured? f)
                           (native-function-structured-delegate f))))
    (string-append
     "\nstruct TORCH_API " name " {\n"
     "  using schema = "
     (dispatcher-signature-type func use-const-ref use-ilistref) ";\n"
     "  using ptr_schema = schema*;\n"
     "  // See Note [static constexpr char* members for windows NVCC]\n"
     "  static constexpr const char* name = \"aten::"
     (base-operator-spelling func) "\";\n"
     "  static constexpr const char* overload_name = \""
     (operator-name-overload-name (function-schema-name func)) "\";\n"
     "  static constexpr const char* schema_str = "
     (cpp-string (function-schema->string func)) ";\n"
     "  static "
     (dispatcher-signature-defn func use-const-ref use-ilistref "call" #f) ";\n"
     "  static "
     (dispatcher-signature-defn func use-const-ref use-ilistref "redispatch" #t)
     ";\n"
     "};")))

;; ---------------------------------------------------------------------------
;; ComputeOperators, DEFINITION branch
;; ---------------------------------------------------------------------------

(define (operator-definition f)
  ;; Mirrors torchgen gen.py ComputeOperators.__call__ (DEFINITION branch).  The
  ;; "    \n" (four spaces then newline) before each body is an artifact of the
  ;; frozen f-string "{    {fn_body}}": fn_body itself opens with a newline.
  (let* ((func (native-function-func f))
         (name (unambiguous-name func))
         (use-const-ref (native-function-use-const-ref-for-mutable-tensors? f))
         (use-ilistref (or (native-function-structured? f)
                           (native-function-structured-delegate f)))
         (schema (function-schema->string func))
         (arg-names (map argument-name
                         (arguments-all (function-schema-arguments func))))
         (call-args (string-join arg-names ", "))
         (redispatch-args (string-join (cons "dispatchKeySet" arg-names) ", "))
         (call-defn
          (dispatcher-signature-defn func use-const-ref use-ilistref
                                     (string-append name "::call") #f))
         (redispatch-defn
          (dispatcher-signature-defn func use-const-ref use-ilistref
                                     (string-append name "::redispatch") #t)))
    (define (fn-body method args-str)
      (string-append
       "\n    static auto op = create_" name "_typed_handle();\n"
       "    return op." method "(" args-str ");"))
    (string-append
     "\n// aten::" schema "\n"
     "static C10_NOINLINE c10::TypedOperatorHandle<" name "::schema> create_"
     name "_typed_handle() {\n"
     "  return c10::Dispatcher::singleton()\n"
     "      .findSchemaOrThrow(" name "::name, " name "::overload_name)\n"
     "      .typed<" name "::schema>();\n"
     "}\n"
     "\n// aten::" schema "\n"
     call-defn " {\n"
     "    " (fn-body "call" call-args) "\n"
     "}\n"
     "\n// aten::" schema "\n"
     redispatch-defn " {\n"
     "    " (fn-body "redispatch" redispatch-args) "\n"
     "}\n")))

;; ---------------------------------------------------------------------------
;; Operators.h
;; ---------------------------------------------------------------------------

(define* (render-operators-h functions
                             #:key
                             (includes '("#include <ATen/MethodOperators.h>"))
                             (declarations #f))
  (let ((non-method
         (filter (lambda (f) (not (member "method" (native-function-variants f))))
                 functions)))
    (code-template-substitute
     %operators-h-template
     (lambda (key)
       (cond
        ((string=? key "generated_comment")
         (string-append "@generated by " %generator-path " from Operators.h"))
        ((string=? key "Operators_includes") includes)
        ((string=? key "Operators_declarations")
         (or declarations (map operator-declaration non-method)))
        (else (error 'unknown-template-key key)))))))

;; ---------------------------------------------------------------------------
;; Operators.cpp (sharded x5)
;; ---------------------------------------------------------------------------

(define (render-operators-cpp-shards functions num-shards)
  ;; returns ((suffix . content) ...) for "Everything" and "_0".."_n-1".
  (let* ((shard-ids (cons "Everything"
                          (map (lambda (i)
                                 (string-append "_" (number->string i)))
                               (iota num-shards))))
         (shard->defs (make-hash-table))
         (shard->headers (make-hash-table)))
    (for-each
     (lambda (sid)
       (hash-set! shard->defs sid '())
       (hash-set! shard->headers sid '()))
     shard-ids)
    (for-each
     (lambda (f)
       (let* ((root (root-name f))
              (sid (number->string (shard-index root num-shards)))
              (suffix (string-append "_" sid))
              (header (string-append "#include <ATen/ops/" root ".h>"))
              (defn (operator-definition f)))
         (hash-set! shard->headers suffix
                    (append (hash-ref shard->headers suffix) (list header)))
         (hash-set! shard->defs suffix
                    (append (hash-ref shard->defs suffix) (list defn)))
         (hash-set! shard->headers "Everything"
                    (append (hash-ref shard->headers "Everything") (list header)))
         (hash-set! shard->defs "Everything"
                    (append (hash-ref shard->defs "Everything") (list defn)))))
     functions)
    (map
     (lambda (sid)
       (let ((env (list (cons "operator_headers" (hash-ref shard->headers sid))
                        (cons "static_dispatch_extra_headers" '())
                        (cons "definitions" (hash-ref shard->defs sid)))))
         (cons sid
               (code-template-substitute
                %operators-cpp-template
                (lambda (key)
                  (cond
                   ((string=? key "generated_comment")
                    (string-append "@generated by " %generator-path
                                   " from Operators.cpp"))
                   (else
                    (let ((entry (assoc key env)))
                      (if entry (cdr entry)
                          (error 'unknown-template-key key))))))))))
     shard-ids)))

(define (render-operators-cpp-files functions)
  ;; returns ((filename . content) ...) sorted by filename, for the
  ;; Everything + _0.._4 shards.  The Everything shard is rendered (and
  ;; discarded from the compile manifest) exactly as the frozen oracle does.
  (sort
   (map (lambda (entry)
          (cons (string-append "Operators" (car entry) ".cpp") (cdr entry)))
        (render-operators-cpp-shards functions 5))
   (lambda (a b) (string<? (car a) (car b)))))
