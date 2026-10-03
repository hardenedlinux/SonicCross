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

;; G11 (structured composite kernels) + G12 (view_copy kernels).  Both emitters
;; write into the single artifact CompositeViewCopyKernels.cpp:
;;
;;   - ${CompositeViewCopyKernel_Definitions}  -> gen_composite_view_copy_kernel
;;     (torchgen gen_functionalization_type.GenCompositeViewCopyKernel)
;;   - ${GeneratedCompositeFunctional_Definitions} -> gen_composite_functional_kernel
;;     (torchgen native_function_generation.gen_composite_functional_kernel)
;;   - ${GeneratedCompositeOut_Definitions} -> gen_composite_out_kernel
;;     (torchgen native_function_generation.gen_composite_out_kernel)
;;
;; No new IR is introduced: the emitters consume the frozen Semantic IR directly
;; and reproduce the exact byte stream of the torchgen emitters.

(define-module (sonic-cross composite-view-copy-kernels)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-13)
  #:use-module (sonic-cross core-ir)
  #:use-module (sonic-cross native-function)
  #:use-module (sonic-cross generated-functions)
  #:use-module (sonic-cross code-template)
  #:use-module ((sonic-cross api) #:prefix api:)
  #:export (render-composite-view-copy-kernels-cpp))

(define %generator-path "torchgen/gen.py")

;; ---------------------------------------------------------------------------
;; FunctionSchema.aliased_return_names()
;; ---------------------------------------------------------------------------

(define (aliased-return-names func)
  ;; For every return: the input name it aliases, or #f when it aliases nothing.
  (let ((args (arguments-all (function-schema-arguments func))))
    (map
     (lambda (r)
       (let* ((ann (return-annotation r))
              (matches
               (filter (lambda (a)
                         (let ((a-ann (argument-annotation a)))
                           (and a-ann (equal? a-ann ann))))
                       args)))
         (cond
          ((null? matches) #f)
          ((null? (cdr matches)) (argument-name (car matches)))
          (else
           (error 'aliased-return-multiple
                  (return-name r)
                  (string-join (map argument-name matches) ", "))))))
     (function-schema-returns func))))

;; ---------------------------------------------------------------------------
;; native_function_generation.return_str / gather_nonaliased_inner_rets
;; ---------------------------------------------------------------------------

(define (return-str returns names)
  (unless (= (length returns) (length names))
    (error 'return-str-length-mismatch (length returns) (length names)))
  (cond
   ((null? returns) "")
   ((null? (cdr returns)) (string-append "return " (car names) ";"))
   (else
    (string-append "return "
                   (api:ctype-cpp-type (api:cpp-returns-type returns #t #f))
                   "(" (string-join names ", ") ");"))))

(define (gather-nonaliased-inner-rets func out-var)
  (let* ((aliased (aliased-return-names func))
         (is-tuple (> (length (function-schema-returns func)) 1)))
    (let loop ((i 0) (rs aliased) (result '()))
      (if (null? rs)
          (reverse result)
          (loop (1+ i) (cdr rs)
                (if (car rs)
                    result
                    (cons (if is-tuple
                              (string-append "std::get<" (number->string i)
                                             ">(" out-var ")")
                              out-var)
                          result)))))))

;; ---------------------------------------------------------------------------
;; Signature defn helpers carrying an explicit (non-prefix) name override.
;; ---------------------------------------------------------------------------

(define (dispatcher-defn func symint name use-const-ref use-ilistref)
  ;; DispatcherSignature.defn(name=...) -> "{returns} {name}({args defn})".
  (string-append
   (api:ctype-cpp-type
    (api:dispatcher-signature-returns-type func symint use-const-ref))
   " " name "("
   (string-join
    (map api:binding-defn
         (api:dispatcher-signature-arguments
          func symint use-const-ref use-ilistref))
    ", ")
   ")"))

(define (native-defn func symint name use-const-ref use-ilistref)
  ;; NativeSignature.defn(name=...) -> "{returns} {name}({args defn})".
  (string-append
   (api:ctype-cpp-type
    (api:native-signature-returns-type func symint use-const-ref))
   " " name "("
   (string-join
    (map api:binding-defn
         (api:native-signature-arguments func symint use-const-ref use-ilistref))
    ", ")
   ")"))

;; ---------------------------------------------------------------------------
;; G11: gen_composite_functional_kernel
;; ---------------------------------------------------------------------------

(define (gen-composite-functional-kernel g)
  (let ((functional (native-functions-group-functional g))
        (inplace (native-functions-group-inplace g))
        (mutable (native-functions-group-mutable g))
        (out (native-functions-group-out g)))
    (and (member "generated" (native-function-tags functional))
         (let* ((target-f
                 (cond
                  ((and inplace
                        (not (member "generated" (native-function-tags inplace))))
                   inplace)
                  ((and mutable
                        (not (member "generated" (native-function-tags mutable))))
                   mutable)
                  (else
                   (error 'composite-functional-no-target
                          (function-schema->string (native-function-func functional))))))
                (functional-func (native-function-func functional))
                (target-func (native-function-func target-f))
                (out-func (native-function-func out))
                (pairs (map cons
                            (api:dispatcher-jit-arguments functional-func)
                            (api:dispatcher-jit-arguments target-func)))
                (use-const-ref (api:group-use-const-ref-context g))
                (use-ilistref (api:group-use-ilistref-context g)))
           (let loop ((ps pairs) (clone-inputs '()) (ctx '()) (cloned-returns '()))
             (if (null? ps)
                 (let* ((clone-inputs-str (string-join (reverse clone-inputs) "\n"))
                        (exprs
                         (string-join
                          (map api:expr-expr
                               (api:translate
                                (reverse ctx)
                                (api:dispatcher-signature-arguments
                                 target-func #t use-const-ref use-ilistref)))
                          ", "))
                        (out-name "output")
                        (maybe-assign
                         (if (> (length (function-schema-returns target-func)) 0)
                             (string-append "auto " out-name " = ")
                             ""))
                        (inner-return-names
                         (gather-nonaliased-inner-rets target-func out-name))
                        (ret-str
                         (return-str
                          (function-schema-returns functional-func)
                          (append inner-return-names (reverse cloned-returns))))
                        (sig-name
                         (string-append
                          (api:dispatcher-name functional-func)
                          (if (function-schema-has-symint? out-func) "_symint" ""))))
                   (string-append
                    "\n"
                    (dispatcher-defn functional-func #t sig-name
                                     use-const-ref use-ilistref) " {\n"
                    "  " clone-inputs-str "\n"
                    "  " maybe-assign "at::_ops::" (unambiguous-name target-func)
                    "::call(" exprs ");\n"
                    "  " ret-str "\n"
                    "}\n"))
                 (let* ((a-curr (caar ps))
                        (a-tgt (cdar ps))
                        (a-tgt-ann (argument-annotation a-tgt)))
                   (if (and a-tgt-ann (annotation-is-write? a-tgt-ann))
                       (let ((name (argument-name a-curr)))
                         (loop (cdr ps)
                               (cons (string-append "auto " name
                                                    "_clone = clone_arg(" name ");")
                                     clone-inputs)
                               (cons (api:make-expr
                                      (string-append name "_clone")
                                      (api:cpp-argumenttype-type
                                       (argument-type a-curr)
                                       (argument-is-write? a-curr)
                                       name #f #t use-const-ref use-ilistref))
                                     ctx)
                               (cons (string-append name "_clone") cloned-returns)))
                       (loop (cdr ps)
                             clone-inputs
                             (cons (api:make-binding
                                    (argument-name a-curr)
                                    (api:cpp-argumenttype-type
                                     (argument-type a-curr)
                                     (argument-is-write? a-curr)
                                     (argument-name a-curr) #f #t
                                     use-const-ref use-ilistref)
                                    #f)
                                   ctx)
                             cloned-returns)))))))))

;; ---------------------------------------------------------------------------
;; G11: gen_composite_out_kernel
;; ---------------------------------------------------------------------------

(define (gen-composite-out-kernel g)
  (let ((out (native-functions-group-out g)))
    (and (member "generated" (native-function-tags out))
         (let* ((functional (native-functions-group-functional g))
                (out-func (native-function-func out))
                (functional-func (native-function-func functional))
                (use-const-ref (api:group-use-const-ref-context g))
                (use-ilistref (api:group-use-ilistref-context g))
                (exprs
                 (string-join
                  (map api:expr-expr
                       (api:translate
                        (api:dispatcher-signature-arguments
                         out-func #t use-const-ref use-ilistref)
                        (api:dispatcher-signature-arguments
                         functional-func #t use-const-ref use-ilistref)))
                  ", "))
                (out-name "tmp_output")
                (num-functional-returns
                 (length (function-schema-returns functional-func)))
                (out-args (arguments-out (function-schema-arguments out-func)))
                (copy-outs
                 (let loop ((i 0) (args out-args) (acc '()))
                   (if (null? args)
                       (reverse acc)
                       (let* ((out-arg (car args))
                              (ret-name
                               (if (= num-functional-returns 1)
                                   out-name
                                   (string-append "std::get<" (number->string i)
                                                  ">(" out-name ")"))))
                         (loop (1+ i) (cdr args)
                               (cons
                                (string-append
                                 "  resize_out_helper(" (argument-name out-arg)
                                 ", " ret-name ");\n"
                                 "  copy_arg(" (argument-name out-arg)
                                 ", " ret-name ");")
                                acc))))))
                (rets
                 (let loop ((i 0) (names (aliased-return-names out-func)) (acc '()))
                   (if (null? names)
                       (reverse acc)
                       (let ((ret-name (car names)))
                         (loop (1+ i) (cdr names)
                               (cons (or ret-name
                                         (if (= num-functional-returns 1)
                                             out-name
                                             (string-append "std::get<"
                                                            (number->string i)
                                                            ">(" out-name ")")))
                                     acc))))))
                (copy-outs-str (string-join copy-outs "\n"))
                (sig-name
                 (string-append (unambiguous-name out-func)
                                (if (function-schema-has-symint? out-func)
                                    "_symint" ""))))
           (string-append
            "\n"
            (dispatcher-defn out-func #t sig-name use-const-ref use-ilistref) " {\n"
            "  auto " out-name " = at::_ops::" (unambiguous-name functional-func)
            "::call(" exprs ");\n"
            "  " copy-outs-str "\n"
            "  " (return-str (function-schema-returns out-func) rets) "\n"
            "}\n")))))

;; ---------------------------------------------------------------------------
;; G12: GenCompositeViewCopyKernel
;; ---------------------------------------------------------------------------

(define %special-view-copy
  (string-append
   "at::Tensor view_copy_symint(const at::Tensor & self, at::SymIntArrayRef size) {\n"
   "  c10::SymDimVector shape = infer_size_dv(size, self.sym_numel());\n"
   "  if (!at::detail::computeStride(self.sym_sizes(), self.sym_strides(), shape).has_value()) {\n"
   "    return self.reshape_symint(size);\n"
   "  } else {\n"
   "    auto output = at::_ops::view::call(self, size);\n"
   "    return output.clone(/*memory_format=*/at::MemoryFormat::Contiguous);\n"
   "  }\n"
   "}\n"))

(define (gen-composite-view-copy-kernel g backend-index)
  (let ((view (native-functions-view-group-view g))
        (view-copy (native-functions-view-group-view-copy g)))
    (and view-copy
         (let* ((view-func (native-function-func view))
                (view-copy-func (native-function-func view-copy))
                (view-copy-base
                 (base-operator-name-base
                  (operator-name-base (function-schema-name view-copy-func))))
                (use-const-ref (api:native-function-use-const-ref-context view))
                (use-ilistref (api:native-function-use-ilistref-context view)))
           (and (string=? view-copy-base
                          (string-append (base-operator-spelling view-func) "_copy"))
                (let* ((metadata (api:backend-index-get-kernel view-copy backend-index)))
                  (unless metadata
                    (error 'missing-view-copy-kernel
                           (operator-name->string (function-schema-name view-copy-func))))
                  (if (string=? (operator-name->string
                                 (function-schema-name view-copy-func))
                                "view_copy")
                      (begin
                        (unless (string=? (backend-metadata-kernel metadata)
                                          "view_copy_symint")
                          (error 'expected-view-copy-symint
                                 (backend-metadata-kernel metadata)))
                        %special-view-copy)
                      (let* ((symint (backend-metadata-supports-symint? metadata))
                             (exprs
                              (string-join
                               (map api:expr-expr
                                    (api:translate
                                     (api:native-signature-arguments
                                      view-copy-func symint use-const-ref use-ilistref)
                                     (api:dispatcher-signature-arguments
                                      view-func #t use-const-ref use-ilistref)))
                               ", "))
                             (view-api-name (unambiguous-name view-func))
                             (return-cloned-output
                              (if (equal? (return-type
                                           (car (function-schema-returns view-func)))
                                          tensor-type)
                                  "  return output.clone(/*memory_format=*/at::MemoryFormat::Contiguous);"
                                  (string-append
                                   "  "
                                   (api:ctype-cpp-type
                                    (api:native-signature-returns-type
                                     view-copy-func symint use-const-ref))
                                   " out_clone;\n"
                                   "  for (const auto i : c10::irange(output.size())) {\n"
                                   "    out_clone.push_back(output[i].clone(/*memory_format=*/at::MemoryFormat::Contiguous));\n"
                                   "  }\n"
                                   "  return out_clone;"))))
                        (string-append
                         "\n"
                         (native-defn view-copy-func symint
                                      (backend-metadata-kernel metadata)
                                      use-const-ref use-ilistref) " {\n"
                         "  auto output = at::_ops::" view-api-name "::call(" exprs ");\n"
                         "  " return-cloned-output "\n"
                         "}\n")))))))))

;; ---------------------------------------------------------------------------
;; ops_headers (the #else AT_PER_OPERATOR_HEADERS branch)
;; ---------------------------------------------------------------------------

(define (root-name nf)
  (base-operator-name-base
   (operator-name-base (function-schema-name (native-function-func nf)))))

(define (op-header-block nf)
  (string-append "#include <ATen/ops/" (root-name nf) "_ops.h>\n"
                 "#include <ATen/ops/" (root-name nf) "_native.h>"))

(define (view-groups-header-str view-groups)
  (string-join
   (append-map
    (lambda (g)
      (let ((view (native-functions-view-group-view g))
            (view-copy (native-functions-view-group-view-copy g)))
        (if view-copy
            (list (op-header-block view) (op-header-block view-copy))
            (list (op-header-block view)))))
    view-groups)
   "\n"))

(define (structured-header-str structured)
  (string-join
   (append-map
    (lambda (g)
      (append-map
       (lambda (f)
         (if (and f (not (member "generated" (native-function-tags f))))
             (list (op-header-block f))
             '()))
       (list (native-functions-group-inplace g)
             (native-functions-group-mutable g)
             (native-functions-group-functional g))))
    structured)
   "\n"))

;; ---------------------------------------------------------------------------
;; template + render
;; ---------------------------------------------------------------------------

(define %composite-view-copy-kernels-template
  (string-append
   "#define TORCH_ASSERT_ONLY_METHOD_OPERATORS\n"
   "// ${generated_comment}\n"
   "\n"
   "#include <ATen/InferSize.h>\n"
   "#include <ATen/Tensor.h>\n"
   "#include <ATen/native/Resize.h>\n"
   "\n"
   "#ifndef AT_PER_OPERATOR_HEADERS\n"
   "#include <ATen/Operators.h>\n"
   "#else\n"
   "#include <ATen/ops/clone.h>\n"
   "$ops_headers\n"
   "#endif\n"
   "\n"
   "namespace at {\n"
   "namespace native {\n"
   "\n"
   "// This file contains a number of kernels for aten functions that are fully code-generated.\n"
   "// TODO: rename this file to something more generic.\n"
   "\n"
   "namespace {\n"
   "at::Tensor clone_arg(const at::Tensor& t) {\n"
   "    return t.clone();\n"
   "}\n"
   "\n"
   "std::vector<at::Tensor> clone_arg(const at::TensorList& t_list) {\n"
   "    std::vector<at::Tensor> out(t_list.size());\n"
   "    for (const auto& i : c10::irange(t_list.size())) {\n"
   "        out[i] = t_list[i].clone();\n"
   "    }\n"
   "    return out;\n"
   "}\n"
   "\n"
   "// duped with gen_resize_out_helper from structured kernels\n"
   "void copy_arg(const at::Tensor& dst, const at::Tensor& src) {\n"
   "    TORCH_CHECK(src.dtype() == dst.dtype(),\n"
   "        \"Expected out tensor to have dtype \", src.dtype(), \", but got \", dst.dtype(), \" instead\");\n"
   "    TORCH_CHECK(src.device() == dst.device(),\n"
   "        \"Expected out tensor to have device \", src.device(), \", but got \", dst.device(), \" instead\");\n"
   "    dst.copy_(src);\n"
   "}\n"
   "\n"
   "void copy_arg(const at::TensorList& dst, const at::TensorList& src) {\n"
   "    TORCH_INTERNAL_ASSERT(dst.size() == src.size());\n"
   "    for (const auto& i : c10::irange(dst.size())) {\n"
   "        copy_arg(dst[i], src[i]);\n"
   "    }\n"
   "}\n"
   "\n"
   "// TODO: this doesn't handle restriding empty tensors correctly; see\n"
   "// gen_resize_out_helper for the correct algorithm\n"
   "\n"
   "void resize_out_helper(const at::Tensor& dst, const at::Tensor& src) {\n"
   "    at::native::resize_output(dst, src.sizes());\n"
   "}\n"
   "\n"
   "void resize_out_helper(const at::TensorList& dst, const at::TensorList& src) {\n"
   "    TORCH_INTERNAL_ASSERT(dst.size() == src.size());\n"
   "    for (const auto& i : c10::irange(dst.size())) {\n"
   "        at::native::resize_output(dst[i], src[i].sizes());\n"
   "    }\n"
   "}\n"
   "}\n"
   "\n"
   "\n"
   "${CompositeViewCopyKernel_Definitions}\n"
   "\n"
   "${GeneratedCompositeFunctional_Definitions}\n"
   "\n"
   "${GeneratedCompositeOut_Definitions}\n"
   "\n"
   "} // namespace native\n"
   "} // namespace at\n"))

(define (render-composite-view-copy-kernels-cpp structured view-groups backend-index)
  (code-template-substitute
   %composite-view-copy-kernels-template
   (lambda (key)
     (cond
      ((string=? key "generated_comment")
       (string-append "@generated by " %generator-path
                      " from CompositeViewCopyKernels.cpp"))
      ((string=? key "ops_headers")
       (list (view-groups-header-str view-groups)
             (structured-header-str structured)))
      ((string=? key "CompositeViewCopyKernel_Definitions")
       (filter-map
        (lambda (g) (gen-composite-view-copy-kernel g backend-index))
        view-groups))
      ((string=? key "GeneratedCompositeFunctional_Definitions")
       (filter-map gen-composite-functional-kernel structured))
      ((string=? key "GeneratedCompositeOut_Definitions")
       (filter-map gen-composite-out-kernel structured))
      (else (error 'unknown-template-key key))))))
