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

;; G2: Register{DispatchKey}.cpp.  Faithful port of torchgen
;; dest/register_dispatch_key.py + gen.get_native_function_definitions +
;; gen.get_kernel_namespace + utils.write_sharded_with_template, driven over the
;; 22 filtered dispatch keys.  The rendered C++ is the sole output: templates
;; are embedded verbatim from the frozen commit and the text is byte-identical
;; to `torchgen/gen.py` at 41ffbc4a994e058af9fe00ed5caba73fc1033359.

(define-module (sonic-cross register-dispatch-key)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-13)
  #:use-module (sonic-cross core-ir)
  #:use-module (sonic-cross native-function)
  #:use-module (sonic-cross api)
  #:use-module (sonic-cross code-template)
  #:use-module (sonic-cross orchestration)
  #:use-module (sonic-cross file-manager)
  #:export (make-backend-index
            backend-index-effective-key
            render-register-dispatch-key-files
            render-dispatch-key-functions))

(define %generator-path "torchgen/gen.py")

;; ---------------------------------------------------------------------------
;; small helpers
;; ---------------------------------------------------------------------------

(define (string-replace-all s target replacement)
  (let loop ((result "") (rest s))
    (let ((idx (string-contains rest target)))
      (if (not idx)
          (string-append result rest)
          (loop (string-append result (substring rest 0 idx) replacement)
                (substring rest (+ idx (string-length target))))))))

(define (char-whitespace-only? line)
  (string-every char-whitespace? line))

(define (py-splitlines s)
  (let ((parts (string-split s #\newline)))
    (if (and (pair? parts) (string-null? (car (last-pair parts))))
        (drop-right parts 1)
        parts)))

(define (py-splitlines-keepends s)
  ;; str.splitlines(keepends=True) over content whose only terminator is \n.
  (let loop ((i 0) (start 0) (out '()))
    (if (>= i (string-length s))
        (reverse (if (> i start) (cons (substring s start i) out) out))
        (if (char=? (string-ref s i) #\newline)
            (loop (1+ i) (1+ i) (cons (substring s start (1+ i)) out))
            (loop (1+ i) start out)))))

(define (textwrap-indent* text prefix)
  ;; textwrap.indent(text, prefix) with the default predicate line.strip().
  (if (string-null? text)
      ""
      (string-concatenate
       (map (lambda (line)
              (if (char-whitespace-only? line)
                  line
                  (string-append prefix line)))
            (py-splitlines-keepends text)))))

(define (schema-kind-name kind)
  (cond
   ((eq? kind schema-kind-functional) "functional")
   ((eq? kind schema-kind-inplace) "inplace")
   ((eq? kind schema-kind-mutable) "mutable")
   ((eq? kind schema-kind-out) "out")
   ((eq? kind schema-kind-scratch) "scratch")
   (else (error 'bad-schema-kind kind))))

(define (has-composite-kernel? f)
  (or (native-function-has-composite-implicit-autograd-kernel? f)
      (native-function-has-composite-explicit-autograd-kernel? f)
      (native-function-has-composite-explicit-autograd-non-functional-kernel? f)))

(define (arguments-flat-positional args)
  (append (arguments-pre-self-positional args)
          (if (arguments-self-arg args)
              (list (self-argument-argument (arguments-self-arg args)))
              '())
          (arguments-post-self-positional args)))

(define (self-arg-name func)
  (let ((sa (arguments-self-arg (function-schema-arguments func))))
    (and sa (argument-name (self-argument-argument sa)))))

;; ---------------------------------------------------------------------------
;; BackendIndex: (effective-key index device-guard)
;; ---------------------------------------------------------------------------

(define (make-backend-index key indices)
  (let ((index (assoc-ref indices key)))
    (if index
        (list key index (is-cuda-dispatch-key? key))
        (list "Undefined" '() #f))))

(define (backend-index-effective-key bi) (car bi))
(define (backend-index-index bi) (cadr bi))
(define (backend-index-device-guard? bi) (caddr bi))

(define (backend-index-get-kernel f bi)
  (let ((name (operator-name->string
               (function-schema-name (native-function-func f)))))
    (assoc-ref (backend-index-index bi) name)))

(define (backend-index-has-kernel f bi)
  (and (backend-index-get-kernel f bi) #t))

(define (backend-index-get-kernel-group group bi)
  (backend-index-get-kernel (native-functions-group-out group) bi))

;; ---------------------------------------------------------------------------
;; registration headers / helpers (dest/register_dispatch_key.py)
;; ---------------------------------------------------------------------------

(define %extra-cuda-headers
  (string-append
   "#include <c10/cuda/CUDAGuard.h>\n"
   "#include <ATen/cuda/ATenCUDAGeneral.h>\n"
   "#include <ATen/cuda/CUDADevice.h>\n"
   "#include <ATen/cuda/CUDAContext.h>"))

(define (gen-registration-headers dispatch-key)
  ;; per_operator_headers=False, rocm=False.
  (let ((headers (list "#include <ATen/NativeFunctions.h>")))
    (cond
     ((member dispatch-key '("CPU" "Meta"))
      (set! headers (append headers (list "#include <ATen/EmptyTensor.h>"))))
     ((string=? dispatch-key "CUDA")
      (set! headers (append headers (list "#include <ATen/cuda/EmptyTensor.h>"))))
     ((string=? dispatch-key "MPS")
      (set! headers (append headers (list "#include <ATen/mps/EmptyTensor.h>"))))
     ((string=? dispatch-key "XPU")
      (set! headers (append headers (list "#include <ATen/xpu/EmptyTensor.h>"))))
     ((string=? dispatch-key "MTIA")
      (set! headers (append headers (list "#include <ATen/native/mtia/EmptyTensor.h>"))))
     (else
      (set! headers (append headers (list "#include <ATen/Functions.h>")))))
    (append headers (list "#include <c10/macros/Macros.h>"))))

(define (gen-operator-headers dispatch-key)
  ;; per_operator_headers=False.
  (let ((headers (list "#include <ATen/NativeFunctions.h>")))
    (if (string=? dispatch-key "CompositeExplicitAutogradNonFunctional")
        (set! headers (append headers (list "#include <ATen/Functions.h>"))))
    (if (member dispatch-key functions-keys)
        (set! headers (append headers
                              (list (string-append "#include <ATen/" dispatch-key
                                                   "Functions.h>")))))
    headers))

(define (gen-empty-impl-names dispatch-key)
  (cond
   ((member dispatch-key '("Meta" "CPU" "CUDA" "MPS" "XPU" "MTIA"))
    (let ((d (string-downcase dispatch-key)))
      (values (string-append "at::detail::empty_" d)
              (string-append "at::detail::empty_strided_" d))))
   ((member dispatch-key '("CompositeExplicitAutogradNonFunctional"
                           "QuantizedCPU" "QuantizedCUDA"))
    (values "at::empty" "at::empty_strided"))
   (else (values #f #f))))

(define (gen-create-out-helper dispatch-key)
  (let ((empty-options (if (string=? dispatch-key "Meta")
                           "options.device(at::kMeta)"
                           "options")))
    (call-with-values (lambda () (gen-empty-impl-names dispatch-key))
      (lambda (empty-impl empty-strided-impl)
        (if (not empty-impl)
            '()
            (list
             (string-append
              "\n"
              "Tensor create_out(IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {\n"
              "  if (strides.empty()) {\n"
              "      return " empty-impl "(sizes, " empty-options ");\n"
              "  } else {\n"
              "      return " empty-strided-impl "(sizes, strides, " empty-options ");\n"
              "  }\n"
              "}\n")))))))

(define (gen-maybe-create-proxy-helper dispatch-key)
  (call-with-values (lambda () (gen-empty-impl-names dispatch-key))
    (lambda (empty-impl empty-strided-impl)
      (if (not empty-strided-impl)
          '()
          (list
           (string-append
            "\n"
            "std::optional<Tensor> maybe_create_proxy(const Tensor &out, IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {\n"
            "  if (out.strides() != strides) {\n"
            "    return " empty-strided-impl "(sizes, strides, options);\n"
            "  }\n"
            "  return std::nullopt;\n"
            "}\n"))))))

(define (gen-resize-out-helper dispatch-key)
  (if (string=? dispatch-key "CompositeExplicitAutogradNonFunctional")
      '()
      (list
       (string-append
        "\n"
        "void resize_out(const Tensor &out, IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {\n"
        "  TORCH_CHECK(options.dtype() == out.dtype(),\n"
        "      \"Expected out tensor to have dtype \", options.dtype(), \", but got \", out.dtype(), \" instead\");\n"
        "  TORCH_CHECK(options.device() == out.device(),\n"
        "      \"Expected out tensor to have device \", options.device(), \", but got \", out.device(), \" instead\");\n"
        "  const bool resized = at::native::resize_output(out, sizes);\n"
        "  // Only restride if a resize occurred; otherwise we ignore the (advisory)\n"
        "  // strides from the meta function and directly use the output tensor's\n"
        "  // preexisting strides\n"
        "  if (resized) {\n"
        "    if (!strides.empty()) {\n"
        "      TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());\n"
        "      // TODO: avoid the redispatch here\n"
        "      out.as_strided_(sizes, strides);\n"
        "    } else if (options.memory_format_opt().has_value()) {\n"
        "      out.unsafeGetTensorImpl()->empty_tensor_restride(*options.memory_format_opt());\n"
        "    }\n"
        "  }\n"
        "}\n"))))

(define (gen-check-inplace-helper dispatch-key)
  (list
   (string-append
    "\n"
    "void check_inplace(const Tensor &self, IntArrayRef sizes, const TensorOptions &options) {\n"
    "  // These checks are needed on those operators that:\n"
    "  //   1) don't use 'TensorIterator' (e.g. 'addmm' and 'baddbmm')\n"
    "  //   2) have particular typing rules (e.g. 'cumsum' and 'cumprod')\n"
    "  // For other operators (e.g. 'add'), 'TensorIterator' already checks\n"
    "  // these things separately.\n"
    "  TORCH_CHECK(options.dtype() == self.dtype(),\n"
    "      \"Bad in-place call: \",\n"
    "      \"input tensor dtype \", self.dtype(), \" and output tensor dtype \", options.dtype(), \" should match\");\n"
    "  TORCH_CHECK(options.device() == self.device(),\n"
    "      \"Bad in-place call: \",\n"
    "      \"input tensor device \", self.device(), \" and output tensor device \", options.device(), \" should match\");\n"
    "  TORCH_CHECK(sizes == self.sizes(),\n"
    "      \"Bad in-place call: \",\n"
    "      \"input tensor size \", self.sizes(), \" and output tensor size \", sizes, \" should match\");\n"
    "}\n")))

(define (gen-registration-helpers dispatch-key)
  (append
   (list "C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED(\"-Wunused-function\")")
   (gen-create-out-helper dispatch-key)
   (gen-resize-out-helper dispatch-key)
   (gen-check-inplace-helper dispatch-key)
   (gen-maybe-create-proxy-helper dispatch-key)
   (list "C10_DIAGNOSTIC_POP()")))

;; ---------------------------------------------------------------------------
;; device check / guard (RegisterDispatchKey.gen_device_check + guard)
;; ---------------------------------------------------------------------------

(define (gen-device-check device-check args method-name)
  (if (string=? device-check "NoCheck")
      "  // No device check\n"
      (let ((header (string-append
                     "std::optional<Device> common_device = std::nullopt;\n"
                     "(void)common_device; // Suppress unused variable warning\n")))
        (string-append
         header
         (string-concatenate
          (map (lambda (arg)
                 (if (type-is-tensor-like? (argument-type arg))
                     (string-append
                      "\n  c10::impl::check_and_update_common_device(common_device, "
                      (argument-name arg) ", \"" method-name "\", \""
                      (argument-name arg) "\");")
                     ""))
               args))))))

(define (gen-device-guard f device-guard? dispatch-key)
  ;; returns the device_guard string for gen_unstructured.
  (if (and (native-function-device-guard? f) device-guard?)
      (let* ((args (function-schema-arguments (native-function-func f)))
             (has-tensor-options (and (arguments-tensor-options args) #t)))
        (if has-tensor-options
            (if (is-cuda-dispatch-key? dispatch-key)
                (string-append
                 "globalContext().lazyInitDevice(c10::DeviceType::CUDA);\n"
                 "\n"
                 "  const DeviceGuard device_guard(device_or_default(device));")
                "\n  const DeviceGuard device_guard(device_or_default(device));")
            (let* ((self-name (self-arg-name (native-function-func f)))
                   (candidates (append (if self-name (list self-name) '())
                                       (map argument-name (arguments-out args))
                                       (map argument-name
                                            (arguments-flat-positional args))))
                   (device-of (find (lambda (name)
                                      (let ((arg (find (lambda (a)
                                                         (string=? (argument-name a) name))
                                                       (append (arguments-out args)
                                                               (arguments-flat-positional args))
                                                       )))
                                        (and arg (type-is-tensor-like? (argument-type arg)))))
                                    candidates)))
              (if device-of
                  (string-append "const OptionalDeviceGuard device_guard(device_of("
                                 device-of "));")
                  "// DeviceGuard omitted"))))
      "// DeviceGuard omitted"))

;; ---------------------------------------------------------------------------
;; signatures (wrapper_kernel_sig etc.)
;; ---------------------------------------------------------------------------

(define (wrapper-prefix dispatch-key func)
  (string-append "wrapper_" dispatch-key "_"
                 (operator-name-overload-name (function-schema-name func)) "_"))

;; The `sig` for unstructured is a DispatcherSignature (symint=self.symint=True).
(define (unstructured-sig-defn func dispatch-key use-const-ref use-ilistref)
  (dispatcher-signature-defn func (wrapper-prefix dispatch-key func) #t
                             use-const-ref use-ilistref))

(define (unstructured-sig-name func dispatch-key)
  (dispatcher-signature-name func (wrapper-prefix dispatch-key func)))

(define (unstructured-sig-arguments func dispatch-key use-const-ref use-ilistref)
  (dispatcher-signature-arguments func #t use-const-ref use-ilistref))

(define (unstructured-sig-returns-type func use-const-ref)
  (ctype-cpp-type (dispatcher-signature-returns-type func #t use-const-ref)))

;; ---------------------------------------------------------------------------
;; RegisterDispatchKey.gen_unstructured
;; ---------------------------------------------------------------------------

(define (gen-unstructured f dispatch-key backend-index target)
  ;; f :: <native-function>; target :: 'namespaced-definition |
  ;; 'anonymous-definition | 'registration.  Returns a list of strings.
  (let* ((func (native-function-func f))
         (use-const-ref (native-function-use-const-ref-context f))
         (use-ilistref (native-function-use-ilistref-context f))
         (has-kernel (backend-index-has-kernel f backend-index))
         (kind (function-schema-kind func))
         (returns (function-schema-returns func)))
    (define (inplace-meta?)
      (and (not has-kernel)
           (string=? (backend-index-effective-key backend-index) "Meta")
           (eq? kind schema-kind-inplace)
           (not (has-composite-kernel? f))
           (= (length returns) 1)))
    ;; use_out_as_primary is always True in-tree, so the out/inplace wrapper
    ;; branch (gets_out_inplace_wrapper) is unreachable and omitted.
    (cond
     ((and (not has-kernel) (not (inplace-meta?))) '())
     ((native-function-manual-kernel-registration? f) '())
     (else
      (let* ((sig-name (unstructured-sig-name func dispatch-key))
             (returns-type (unstructured-sig-returns-type func use-const-ref))
             (sig-args (unstructured-sig-arguments func dispatch-key
                                                   use-const-ref use-ilistref))
             (args-str (string-join (map binding-defn sig-args) ", "))
             (cpp-sigs (cpp-signature-group-signatures
                        func #t #t #f (native-function-cpp-no-default-args f))))
        (case target
          ((namespaced-definition)
           (append-map
            (lambda (cpp-sig)
              (list
               (string-append
                "\n"
                (cpp-signature-defn cpp-sig use-const-ref use-ilistref) " {\n"
                "return " sig-name "("
                (string-join
                 (map expr-expr
                      (translate (cpp-signature-arguments cpp-sig use-const-ref
                                                          use-ilistref)
                                 sig-args))
                 ", ")
                ");\n"
                "}\n")))
            cpp-sigs))
          ((anonymous-definition)
           (if (inplace-meta?)
               (let ((self-name (self-arg-name func)))
                 (unless self-name (error 'inplace-meta-no-self func))
                 (list
                  (string-append
                   "\n"
                   returns-type " " sig-name "(" args-str ") {\n"
                   "  TORCH_CHECK_NOT_IMPLEMENTED(" self-name ".is_meta(),\n"
                   "    \"Cannot inplace into non-meta tensor with meta tensor argument\");\n"
                   "  return " self-name ";\n"
                   "}\n")))
               (let ((meta (backend-index-get-kernel f backend-index)))
                 (if (not meta)
                     '()
                     (let* ((impl-name (string-append
                                        (backend-metadata-cpp-namespace meta) "::"
                                        (backend-metadata-kernel meta)))
                            (symint (backend-metadata-supports-symint? meta))
                            (kernel-args (native-signature-arguments func symint
                                                                     use-const-ref
                                                                     use-ilistref))
                            (args-exprs-str
                             (string-join
                              (map expr-expr (translate sig-args kernel-args))
                              ", "))
                            (device-check
                             (if (backend-index-device-guard? backend-index)
                                 (gen-device-check
                                  (native-function-device-check f)
                                  (append (arguments-out
                                           (function-schema-arguments func))
                                          (arguments-flat-positional
                                           (function-schema-arguments func)))
                                  sig-name)
                                 "  // No device check\n"))
                            (device-guard
                             (gen-device-guard f
                                               (backend-index-device-guard?
                                                backend-index)
                                               (backend-index-effective-key
                                                backend-index))))
                       (list
                        (string-append
                         "namespace {\n"
                         "\n"
                         returns-type " " sig-name "(" args-str ") {\n"
                         "  " device-check "\n"
                         "\n"
                         "  " device-guard "\n"
                         "  return " impl-name "(" args-exprs-str ");\n"
                         "}\n"
                         "\n"
                         "} // anonymous namespace\n")))))))
          ((registration)
           (if (native-function-manual-kernel-registration? f)
               '()
               (list
                (string-append
                 "m.impl(\""
                 (operator-name->string (function-schema-name func))
                 "\",\nTORCH_FN(" sig-name "));\n"))))
          ((namespaced-declaration)
           ;; Target.NAMESPACED_DECLARATION: one multi-line string of
           ;; "TORCH_API {decl};" lines, one per CppSignature (incl. symint).
           (list
            (string-concatenate
             (map (lambda (cpp-sig)
                    (string-append
                     "TORCH_API "
                     (cpp-signature-decl cpp-sig use-const-ref use-ilistref)
                     ";\n"))
                  cpp-sigs))))
          (else (error 'bad-target target))))))))

;; ---------------------------------------------------------------------------
;; structured (StructuredRegisterDispatchKey)
;; ---------------------------------------------------------------------------

(define (meta-name g)
  (string-replace-all
   (operator-name->string
    (function-schema-name
     (native-function-func (native-functions-group-functional g))))
   "." "_"))

(define (structured-class-name-parent dispatch-key g metadata k)
  ;; returns (values class-name parent-class)
  (cond
   ((string=? dispatch-key "Meta")
    (let ((mn (meta-name g)))
      (values (string-append "structured_" mn "_meta_" k)
              (string-append "at::meta::structured_" mn))))
   ((string=? dispatch-key "CompositeExplicitAutogradNonFunctional")
    (let ((mn (meta-name g)))
      (values (string-append "structured_" mn "_default_backend_" k)
              (string-append "at::meta::structured_" mn))))
   (else
    (let ((kernel (backend-metadata-kernel metadata))
          (ns (backend-metadata-cpp-namespace metadata)))
      (values (string-append "structured_" kernel "_" k)
              (string-append ns "::structured_" kernel))))))

(define (gen-class-ctor k class-name returns)
  (cond
   ((eq? k schema-kind-functional) "")
   ((eq? k schema-kind-inplace)
    (string-append class-name "(Tensor& self) : outputs_{std::ref(self)} {}"))
   ((eq? k schema-kind-out)
    (let* ((out-args (string-join
                      (map (lambda (i) (string-append "Tensor& out" (number->string i)))
                           (iota returns))
                      ", "))
           (out-refs (string-join
                      (map (lambda (i) (string-append "std::ref(out" (number->string i) ")"))
                           (iota returns))
                      ", ")))
      (string-append class-name "(" out-args ") : outputs_{ " out-refs " } {}")))
   (else (error 'bad-structured-ctor k))))

(define %structured-set-guard
  (string-append
   "\n"
   "auto current_device = guard_.current_device();\n"
   "if (C10_UNLIKELY(current_device.has_value())) {\n"
   "  TORCH_INTERNAL_ASSERT(*current_device == options.device(),\n"
   "    \"structured kernels don't support multi-device outputs\");\n"
   "} else {\n"
   "  guard_.reset_device(options.device());\n"
   "}\n"))

(define %structured-create-proxy
  (string-append
   "\n"
   "auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);\n"
   "if (C10_UNLIKELY(maybe_proxy.has_value())) {\n"
   "    proxy_outputs_[output_idx] = std::move(maybe_proxy).value();\n"
   "}\n"))

(define (gen-class-set-output-body dispatch-key k maybe-create-proxy)
  (define (uses-guard?)
    (member dispatch-key '("CUDA" "MPS" "XPU" "CompositeExplicitAutogradNonFunctional")))
  (let ((maybe-set-guard-line (if (uses-guard?) (string-append %structured-set-guard "\n") ""))
        (create-proxy (if maybe-create-proxy %structured-create-proxy "")))
    (cond
     ((eq? k schema-kind-functional)
      (unless (member dispatch-key '("Meta" "CPU" "CUDA" "MPS" "XPU" "MTIA"
                                     "CompositeExplicitAutogradNonFunctional"))
        (error 'structured-functional-bad-key dispatch-key))
      (string-append maybe-set-guard-line
                     "\noutputs_[output_idx] = create_out(sizes, strides, options);"))
     ((eq? k schema-kind-inplace)
      (string-append maybe-set-guard-line
                     "\nconst auto& out = outputs_[output_idx].get();\n"
                     "check_inplace(out, sizes, options);\n"
                     create-proxy))
     ((eq? k schema-kind-out)
      (string-append maybe-set-guard-line
                     "\nconst auto& out = outputs_[output_idx].get();\n"
                     "resize_out(out, sizes, strides, options);\n"
                     create-proxy))
     (else (error 'bad-structured-body k)))))

(define (gen-set-output-function name maybe-create-proxy k parent-class
                                 generate-super dispatch-key)
  (let* ((set-output-super
          (if generate-super
              (string-append parent-class
                             "::set_output_raw_strided(output_idx, sizes, strides, options);")
              ""))
         (body (textwrap-indent*
                (gen-class-set-output-body dispatch-key k maybe-create-proxy)
                "    ")))
    (string-append
     "\n"
     "void set_output_" name "(\n"
     "    int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,\n"
     "    TensorOptions options\n"
     ") override {\n"
     body
     "\n"
     "    // super must happen after, so that downstream can use maybe_get_output\n"
     "    // to retrieve the output\n"
     (textwrap-indent* set-output-super "    ")
     "\n"
     "}\n")))

(define (gen-class-set-output-functions k parent-class generate-super dispatch-key)
  (string-append
   "\n"
   (gen-set-output-function "strided" #t k parent-class generate-super dispatch-key)
   "\n"
   (gen-set-output-function "raw_strided" #f k parent-class generate-super
                            dispatch-key)
   "\n"))

(define (gen-class func k class-name parent-class generate-super dispatch-key)
  (let* ((returns (length (function-schema-returns func)))
         (output-type (if (eq? k schema-kind-functional)
                          "Tensor"
                          "std::reference_wrapper<Tensor>"))
         (output-value
          (if (eq? k schema-kind-functional)
              "outputs_[output_idx]"
              "proxy_outputs_[output_idx].has_value() ? *proxy_outputs_[output_idx] : outputs_[output_idx].get()"))
         (proxy-field
          (if (eq? k schema-kind-functional)
              ""
              (string-append "std::array<::std::optional<Tensor>, "
                             (number->string returns) "> proxy_outputs_;")))
         (guard-field
          (cond
           ((string=? dispatch-key "CUDA") "c10::cuda::OptionalCUDAGuard guard_;")
           ((member dispatch-key '("CompositeExplicitAutogradNonFunctional" "MPS"
                                   "XPU" "MTIA"))
            "c10::OptionalDeviceGuard guard_;")
           (else "")))
         (ctor (gen-class-ctor k class-name returns))
         (set-fns (gen-class-set-output-functions k parent-class generate-super
                                                  dispatch-key))
         (lines
          (list (string-append "struct " class-name " final : public " parent-class " {")
                (textwrap-indent* ctor "    ")
                (textwrap-indent* set-fns "    ")
                "    const Tensor& maybe_get_output(int64_t output_idx) override {"
                (string-append "      return " output-value ";\n")
                "    }"
                (string-append "    std::array<" output-type ", "
                               (number->string returns) "> outputs_;")
                (textwrap-indent* proxy-field "    ")
                (textwrap-indent* guard-field "    ")
                "};")))
    (string-join (filter (lambda (line) (not (string-null? line))) lines) "\n")))

;; StructuredRegisterDispatchKey.gen_one, NAMESPACED_DEFINITION branch.
(define (structured-namespaced-definition f func dispatch-key backend-index)
  (let* ((use-const-ref (native-function-use-const-ref-context f))
         (use-ilistref (native-function-use-ilistref-context f))
         (kern (backend-index-get-kernel f backend-index))
         (symint (and kern (backend-metadata-supports-symint? kern)))
         (prefix (string-append "wrapper_" dispatch-key "_"))
         (sig-args (native-signature-arguments func symint use-const-ref
                                               use-ilistref))
         (sig-name (native-signature-name func prefix))
         (cpp-sigs (cpp-signature-group-signatures
                    func #t #t #f (native-function-cpp-no-default-args f))))
    (append-map
     (lambda (cpp-sig)
       (list
        (string-append
         "\n"
         (cpp-signature-defn cpp-sig use-const-ref use-ilistref) " {\n"
         "return " sig-name "("
         (string-join
          (map expr-expr
               (translate (cpp-signature-arguments cpp-sig use-const-ref
                                                   use-ilistref)
                          sig-args))
          ", ")
         ");\n"
         "}\n")))
     cpp-sigs)))

;; StructuredRegisterDispatchKey.gen_one, REGISTRATION branch.
(define (structured-registration f func dispatch-key)
  (let ((sig-name (native-signature-name func
                                          (string-append "wrapper_" dispatch-key "_"))))
    (list (string-append "m.impl(\""
                         (operator-name->string (function-schema-name func))
                         "\", TORCH_FN(" sig-name "));"))))

;; StructuredRegisterDispatchKey.gen_one, dispatched on target.
(define (structured-gen-one f g dispatch-key backend-index target)
  (let* ((func (native-function-func f))
         (kind (function-schema-kind func)))
    (if (and (string=? dispatch-key "CompositeExplicitAutogradNonFunctional")
             (eq? kind schema-kind-out))
        '()
        (case target
          ((namespaced-definition)
           (structured-namespaced-definition f func dispatch-key backend-index))
          ((registration)
           (structured-registration f func dispatch-key))
          ((anonymous-definition)
           (structured-anonymous-definition f g dispatch-key backend-index))
          ((namespaced-declaration)
           ;; Target.NAMESPACED_DECLARATION (structured): same "TORCH_API
           ;; {decl};" per CppSignature, no has_kernel precondition.
           (let ((cpp-sigs (cpp-signature-group-signatures
                            func #t #t #f (native-function-cpp-no-default-args f)))
                 (use-const-ref (native-function-use-const-ref-context f))
                 (use-ilistref (native-function-use-ilistref-context f)))
             (list
              (string-concatenate
               (map (lambda (cpp-sig)
                      (string-append
                       "TORCH_API "
                       (cpp-signature-decl cpp-sig use-const-ref use-ilistref)
                       ";\n"))
                    cpp-sigs)))))
          (else (error 'bad-structured-target target))))))

;; StructuredRegisterDispatchKey.gen_one, ANONYMOUS_DEFINITION branch.
(define (structured-anonymous-definition f g dispatch-key backend-index)
  ;; returns a list of strings (the rendered anonymous definition).
  (let* ((func (native-function-func f))
         (kind (function-schema-kind func)))
    (if (and (string=? dispatch-key "CompositeExplicitAutogradNonFunctional")
             (eq? kind schema-kind-out))
        ;; Never generate a default implementation for out; that is what a
        ;; backend implementer has to define.
        '()
        (let* ((use-const-ref (native-function-use-const-ref-context f))
               (use-ilistref (native-function-use-ilistref-context f))
               (kern (backend-index-get-kernel f backend-index))
               (symint (and kern (backend-metadata-supports-symint? kern)))
               (group-kern (backend-index-get-kernel-group g backend-index))
               (prefix (string-append "wrapper_" dispatch-key "_"))
               (sig-args (native-signature-arguments func symint use-const-ref
                                                     use-ilistref))
               (sig-name (native-signature-name func prefix))
               (sig-defn (native-signature-defn func prefix symint use-const-ref
                                                use-ilistref))
               (k (schema-kind-name kind))
               (context sig-args)
               (sig-body '()))
          ;; class / parent (metadata is the group's = out operator's kernel)
          (call-with-values
              (lambda () (structured-class-name-parent dispatch-key g group-kern k))
            (lambda (class-name parent-class)
        (when (and (not (member dispatch-key '("Meta"
                                               "CompositeExplicitAutogradNonFunctional")))
                   (not group-kern))
          (error 'structured-no-kernel-metadata
                 (operator-name->string
                  (function-schema-name
                   (native-function-func (native-functions-group-functional g))))))
        ;; device check
        (when (backend-index-device-guard? backend-index)
          (set! sig-body
                (append sig-body
                        (list (gen-device-check
                               (native-function-device-check f)
                               (append (arguments-out (function-schema-arguments func))
                                       (arguments-flat-positional
                                        (function-schema-arguments func)))
                               sig-name)))))
        ;; construct op
        (cond
         ((eq? kind schema-kind-functional)
          (set! sig-body (append sig-body (list (string-append class-name " op;")))))
         ((eq? kind schema-kind-inplace)
          (set! sig-body (append sig-body (list (string-append class-name " op(self);")))))
         ((eq? kind schema-kind-out)
          (set! sig-body
                (append sig-body
                        (list (string-append
                               class-name " op("
                               (string-join
                                (map argument-name
                                     (arguments-out (function-schema-arguments func)))
                                ", ")
                               ");"))))))
        ;; meta call
        (let* ((meta-exprs
                (string-join
                 (map expr-expr
                      (translate context (structured-meta-arguments g)))
                 ", ")))
          (let ((precomputed (native-function-precomputed (native-functions-group-out g))))
            (if precomputed
                (call-with-values
                    (lambda () (parse-precomputed precomputed))
                  (lambda (replace add)
                    (set! sig-body
                          (append sig-body
                                  (list (string-append "auto precompute = op.meta(" meta-exprs ");"))))
                    ;; Put all of the contents of the precompute struct into
                    ;; the context so that translate will be able to return the
                    ;; correct args for the call to the impl.
                    (for-each
                     (lambda (precomputed-elems)
                       (set! context
                             (append context
                                     (map (lambda (arg)
                                            (make-expr
                                             (string-append "precompute." (argument-name arg))
                                             (structured-argument-type arg (argument-name arg))))
                                          precomputed-elems))))
                     (append (map cdr replace) (list add)))
                    ;; Add a use of the precompute struct so FB internal
                    ;; compilers don't complain that there is an unused variable.
                    (set! sig-body (append sig-body (list "(void)precompute;")))))
                (set! sig-body (append sig-body (list (string-append "op.meta(" meta-exprs ");")))))))
        ;; out args context
        (let loop ((out-args (structured-out-arguments g)) (i 0))
          (unless (null? out-args)
            (let* ((out-arg (car out-args))
                   (expr (if (eq? kind schema-kind-out)
                             (string-append "op.maybe_get_output(" (number->string i) ")")
                             (string-append "op.outputs_[" (number->string i) "]")))
                   (nct (make-named-ctype
                         (named-ctype-name (binding-nctype out-arg))
                         (make-mut-ref-ctype (make-base-ctype %tensor-t)))))
              (set! context (append context (list (make-expr expr nct))))
              (loop (cdr out-args) (1+ i)))))
        ;; impl call
        (cond
         ((string=? dispatch-key "CompositeExplicitAutogradNonFunctional")
          (let* ((out-func (native-function-func (native-functions-group-out g)))
                 (out-cpp-sigs (cpp-signature-group-signatures
                                out-func #t #t
                                (native-function-manual-cpp-binding? f)
                                (native-function-cpp-no-default-args
                                 (native-functions-group-out g))))
                 ;; most_faithful_signature(): g.out always has out args, so
                 ;; the faithful signature is always present (element 2).
                 (out-sig (cadr out-cpp-sigs))
                 (api-name (cpp-signature-name out-sig #f))
                 (out-exprs
                  (string-join
                   (map expr-expr
                        (translate context
                                   (cpp-signature-arguments out-sig use-const-ref
                                                            use-ilistref)))
                   ", ")))
            (set! sig-body (append sig-body
                                   (list (string-append "at::" api-name "(" out-exprs ");"))))))
         ((not (string=? dispatch-key "Meta"))
          (let ((impl-exprs
                 (string-join
                  (map expr-expr
                       (translate context (structured-impl-arguments g)))
                  ", ")))
            (set! sig-body (append sig-body
                                   (list (string-append "op.impl(" impl-exprs ");")))))))
        ;; proxy copy
        (when (or (eq? kind schema-kind-out) (eq? kind schema-kind-inplace))
          (for-each
           (lambda (i)
             (set! sig-body
                   (append sig-body
                           (list
                            (string-append
                             "if (op.proxy_outputs_[" (number->string i)
                             "].has_value()) op.outputs_[" (number->string i)
                             "].get().copy_(*op.proxy_outputs_[" (number->string i) "]);")))))
           (iota (length (function-schema-returns func)))))
        ;; return
        (let* ((nreturns (length (function-schema-returns func)))
               (ret-expr
                (cond
                 ((eq? kind schema-kind-functional)
                  (if (= nreturns 1)
                      "std::move(op.outputs_[0])"
                      (string-append "std::make_tuple("
                                     (string-join
                                      (map (lambda (i)
                                             (string-append "std::move(op.outputs_["
                                                            (number->string i) "])"))
                                           (iota nreturns))
                                      ", ")
                                     ")")))
                 ((eq? kind schema-kind-inplace) "self")
                 ((eq? kind schema-kind-out)
                  (if (= nreturns 1)
                      (argument-name (car (arguments-out
                                           (function-schema-arguments func))))
                      (string-append "std::forward_as_tuple("
                                     (string-join
                                      (map argument-name
                                           (arguments-out
                                            (function-schema-arguments func)))
                                      ", ")
                                     ")")))
                 (else (error 'bad-structured-return kind)))))
          (set! sig-body (append sig-body (list (string-append "return " ret-expr ";")))))
        (list
         (string-append
          (gen-class func kind class-name parent-class
                     (and (native-function-structured-inherits
                           (native-functions-group-out g))
                          #t)
                     dispatch-key)
          "\n"
          "\n"
          sig-defn " {\n"
          (string-join sig-body "\n")
          "\n"
          "}\n"))))))))

;; ---------------------------------------------------------------------------
;; dispatch (RegisterDispatchKey.__call__)
;; ---------------------------------------------------------------------------

(define (gen-one-structured-group g dispatch-key backend-index target)
  ;; For a structured group, dispatch to gen_structured.
  (cond
   ((string=? (backend-index-effective-key backend-index) "Meta")
    (when (backend-index-has-kernel (native-functions-group-out g) backend-index)
      (error 'structured-meta-explicit-dispatch))
    (append-map (lambda (f) (structured-gen-one f g dispatch-key backend-index target))
                (native-functions-group-functions g)))
   ((string=? (backend-index-effective-key backend-index)
              "CompositeExplicitAutogradNonFunctional")
    (when (backend-index-has-kernel (native-functions-group-out g) backend-index)
      (error 'structured-cea-explicit-dispatch))
    (append-map (lambda (f) (structured-gen-one f g dispatch-key backend-index target))
                (native-functions-group-functions g)))
   (else
    (let ((metadata (backend-index-get-kernel-group g backend-index)))
      (if (or (not metadata) (not (backend-metadata-structured metadata)))
          (append-map (lambda (f) (gen-unstructured f dispatch-key backend-index
                                                    target))
                      (native-functions-group-functions g))
          (append-map (lambda (f) (structured-gen-one f g dispatch-key backend-index target))
                      (native-functions-group-functions g)))))))

(define (gen-dispatch item dispatch-key backend-index target)
  ;; item :: native-function | native-functions-group
  (if (native-functions-group? item)
      (if (native-functions-group-structured? item)
          (gen-one-structured-group item dispatch-key backend-index target)
          (append-map (lambda (f) (gen-unstructured f dispatch-key backend-index
                                                    target))
                      (native-functions-group-functions item)))
      (gen-unstructured item dispatch-key backend-index target)))

;; ---------------------------------------------------------------------------
;; get_kernel_namespace / get_native_function_definitions
;; ---------------------------------------------------------------------------

(define %default-kernel-namespace "at::native")

(define (get-kernel-namespace item backend-index)
  (let ((metadata (if (native-functions-group? item)
                      (backend-index-get-kernel-group item backend-index)
                      (backend-index-get-kernel item backend-index))))
    (if metadata (backend-metadata-cpp-namespace metadata)
        %default-kernel-namespace)))

(define (item-namespace item)
  (if (native-functions-group? item)
      (native-function-namespace (native-functions-group-functional item))
      (native-function-namespace item)))

(define %register-dispatch-definitions-template
  (string-append
   "${ns_prologue}\n"
   "\n"
   "// NB: TORCH_LIBRARY_IMPL must be in an anonymous namespace to avoid\n"
   "// ambiguity with conflicting identifiers that may have been defined in\n"
   "// at namespace already.\n"
   "namespace {\n"
   "\n"
   "${dispatch_anonymous_definitions}\n"
   "\n"
   "${static_init_dispatch_registrations}\n"
   "\n"
   "} // anonymous namespace\n"
   "\n"
   "${deferred_dispatch_registrations}\n"
   "\n"
   "namespace ${dispatch_namespace} {\n"
   "\n"
   "${dispatch_namespaced_definitions}\n"
   "\n"
   "} // namespace ${dispatch_namespace}\n"
   "\n"
   "${ns_epilogue}"))

(define (get-native-function-definitions item dispatch-key backend-index)
  ;; returns a list of lines (strings) for a single item.
  (let* ((kernel-namespace (string-replace-all
                            (get-kernel-namespace item backend-index)
                            "::native" ""))
         (ns-defs (gen-dispatch item dispatch-key backend-index
                                'namespaced-definition))
         (anon-defs (gen-dispatch item dispatch-key backend-index
                                  'anonymous-definition))
         (regs (gen-dispatch item dispatch-key backend-index 'registration)))
    (if (null? ns-defs)
        '()
        (let* ((namespace (item-namespace item))
               (registration-body
                (if (null? regs)
                    ""
                    (string-append
                     "\nTORCH_LIBRARY_IMPL(" namespace ", " dispatch-key ", m) {\n    "
                     (string-join regs "\n") "\n}")))
               (ns-prologue (string-append "namespace " kernel-namespace " {"))
               (ns-epilogue (string-append "} // namespace " kernel-namespace))
               (rendered
                (code-template-substitute
                 %register-dispatch-definitions-template
                 (lambda (key)
                   (cond
                    ((string=? key "ns_prologue") ns-prologue)
                    ((string=? key "ns_epilogue") ns-epilogue)
                    ((string=? key "dispatch_anonymous_definitions") anon-defs)
                    ((string=? key "static_init_dispatch_registrations")
                     registration-body)
                    ((string=? key "deferred_dispatch_registrations") "")
                    ((string=? key "dispatch_namespace")
                     (string-downcase dispatch-key))
                    ((string=? key "dispatch_namespaced_definitions") ns-defs)
                    (else (error 'unknown-ini-key key)))))))
          (py-splitlines rendered)))))

;; ---------------------------------------------------------------------------
;; top-level: render Register{key}.cpp for all keys + shards
;; ---------------------------------------------------------------------------

(define %register-dispatch-key-template
  (string-append
   "// an external backend might generate file within its code tree\n"
   "// and check all the source files within the tree with clang-format.\n"
   "// so, disable it since the backend might have a different config.\n"
   "// clang-format off\n"
   "\n"
   "// NOTE: This condition is true for all PyTorch internal libraries, it\n"
   "//       just excludes external projects such as torch_xla which\n"
   "//       reuse some of the PyTorch codegen machinery.\n"
   "#if defined(CAFFE2_BUILD_MAIN_LIB)        || \\\n"
   "    defined(TORCH_CUDA_BUILD_MAIN_LIB)    || \\\n"
   "    defined(TORCH_XPU_BUILD_MAIN_LIB)\n"
   "#define TORCH_ASSERT_ONLY_METHOD_OPERATORS\n"
   "#endif\n"
   "\n"
   "// ${generated_comment}\n"
   "\n"
   "#include <c10/core/TensorImpl.h>\n"
   "#include <c10/core/Allocator.h>\n"
   "#include <ATen/DeviceGuard.h>\n"
   "#include <ATen/Utils.h>\n"
   "#include <ATen/WrapDimUtils.h>\n"
   "#include <ATen/Dispatch.h>\n"
   "#include <c10/util/ExclusivelyOwned.h>\n"
   "#include <c10/util/Half.h>\n"
   "#include <c10/core/UndefinedTensorImpl.h>\n"
   "#include <optional>\n"
   "#include <ATen/Tensor.h>\n"
   "#include <ATen/native/Resize.h>\n"
   "\n"
   "#include <cstddef>\n"
   "#include <functional>\n"
   "#include <memory>\n"
   "#include <utility>\n"
   "\n"
   "#include <ATen/Config.h>\n"
   "#include <ATen/core/op_registration/adaption.h>\n"
   "#include <torch/library.h>\n"
   "$extra_cuda_headers\n"
   "$external_backend_headers\n"
   "$dispatch_headers\n"
   "$ops_headers\n"
   "\n"
   "namespace at {\n"
   "namespace {\n"
   "$dispatch_helpers\n"
   "} // namespace\n"
   "} // namespace at\n"
   "\n"
   "// See template file RegisterDispatchDefinitions.ini\n"
   "$dispatch_definitions\n"))

(define (render-register-dispatch-key-shards grouped dispatch-key backend-index
                                             num-shards)
  ;; returns ((suffix . content) ...) for "Everything" and "_0".."_n-1".
  (let* ((effective-key (backend-index-effective-key backend-index))
         (base-env
          (list
           (cons "extra_cuda_headers"
                 (if (is-cuda-dispatch-key? dispatch-key) %extra-cuda-headers ""))
           (cons "external_backend_headers" "")
           (cons "dispatch_headers" (gen-registration-headers effective-key))
           (cons "ops_headers" (gen-operator-headers dispatch-key))
           (cons "dispatch_helpers"
                 (if (string=? dispatch-key "CompositeImplicitAutogradNestedTensor")
                     '()
                     (gen-registration-helpers effective-key)))))
         (shard-ids (cons "Everything"
                          (map (lambda (i)
                                 (string-append "_" (number->string i)))
                               (iota num-shards))))
         (shard->defs (make-hash-table)))
    (for-each (lambda (sid) (hash-set! shard->defs sid '())) shard-ids)
    (for-each
     (lambda (item)
       (let* ((root-name (item-root-name item))
              (sid (number->string (shard-index root-name num-shards)))
              (suffix (string-append "_" sid))
              (defs (get-native-function-definitions item dispatch-key
                                                     backend-index)))
         (hash-set! shard->defs suffix
                    (append (hash-ref shard->defs suffix) defs))
         (hash-set! shard->defs "Everything"
                    (append (hash-ref shard->defs "Everything") defs))))
     grouped)
    (map
     (lambda (sid)
       (let* ((env (cons (cons "dispatch_definitions" (hash-ref shard->defs sid))
                         base-env))
              (content
               (code-template-substitute
                %register-dispatch-key-template
                (lambda (key)
                  (cond
                   ((string=? key "generated_comment")
                    (string-append "@generated by " %generator-path
                                   " from RegisterDispatchKey.cpp"))
                   (else
                    (let ((entry (assoc key env)))
                      (if entry (cdr entry)
                          (error 'unknown-template-key key)))))))))
         (cons sid content)))
     shard-ids)))

(define (render-register-dispatch-key-files grouped indices)
  ;; returns ((filename . content) ...) sorted by filename, for all 22 keys.
  (sort
   (append-map
    (lambda (key)
      (let* ((backend-index (make-backend-index key indices))
             (num-shards (if (string=? key "CPU") 4 1))
             (shards (render-register-dispatch-key-shards grouped key
                                                          backend-index
                                                          num-shards)))
        (map (lambda (entry)
               (let ((suffix (car entry)) (content (cdr entry)))
                 (cons (string-append "Register" key
                                      (if (string=? suffix "Everything") "" suffix)
                                      ".cpp")
                       content)))
             shards)))
    (filtered-dispatch-keys))
   (lambda (a b) (string<? (car a) (car b)))))

;; ---------------------------------------------------------------------------
;; G8: {DispatchKey}Functions.h / _inl.h (torchgen gen_aggregated_headers)
;; ---------------------------------------------------------------------------

;; Template verbatim from aten/src/ATen/templates/DispatchKeyFunctions.h at the
;; frozen commit.  Only ${inline_headers} is substituted (block, column 0).
(define %dispatch-key-functions-template
  (string-append
   "#include <ATen/core/TensorBody.h>\n"
   "\n"
   "// TODO Undo all logic introduced for Note [Avoiding Include Cycles In Static Dispatch]\n"
   "// Code introduced to avoid cyclic dependency in static dispatch is no longer\n"
   "// needed as static dispatch logic is moved from TensorBody.h, which caused cycles in the first place,\n"
   "// to Operators.cpp for supporting multiple backends with multiple kernels.\n"
   "//\n"
   "// Note [Avoiding Include Cycles In Static Dispatch]\n"
   "// In order to avoid #include cycles in the static dispatch build, we've carefully split out\n"
   "// the static function definition files into {DispatchKey}Functions.h and {DispatchKey}Functions_inl.h.\n"
   "//\n"
   "// Without this split, the include cycle looks like TensorBody.h -> CPUFunctions.h -> TensorBody.h.\n"
   "// - TensorBody.h #includes CPUFunctions.h in the static dispatch build, because the tensor methods\n"
   "//   all need to call into the fastpath C++ API defined in CPUFunctions.h. The methods are also all\n"
   "//   directly inlined into TensorBody.h.\n"
   "// - CPUFunctions.h #includes TensorBody.h because it contains function declarations for the entire C++ API,\n"
   "//   which include functions that have defaultable std::optional<Tensor> arguments.\n"
   "//   That requires knowing the full Tensor class definition.\n"
   "//\n"
   "// We break the cycle by doing the following:\n"
   "// - Split out CPUFunction.h into two files: CPUFunctions.h and CPUFunctions_inl.h\n"
   "// - CPUFunction.h is a dummy file that just includes the Tensor class and includes CPUFunctions_inl.,\n"
   "// - CPUFunctions_inl.h includes everything else\n"
   "// - (only in the static dispatch build) TensorBody.h makes sure to finish defining the Tensor class,\n"
   "//   and then it includes CPUFunctions_inl.h.\n"
   "// - All other files that want the cpu fastpath functions can include CPUFunctions.h directly.\n"
   "// - This also means that static dispatch build, CPUFunctions.h only needs to\n"
   "//   #include TensorBody.h, and it will automatically bring in CPUFunctions_inl.h.\n"
   "${inline_headers}\n"))

;; Template verbatim from aten/src/ATen/templates/DispatchKeyFunctions_inl.h.
(define %dispatch-key-functions-inl-template
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
   "#if defined(AT_PER_OPERATOR_HEADERS) && defined(TORCH_ASSERT_ONLY_METHOD_OPERATORS)\n"
   "#error This change adds a dependency on all pytorch operators, meaning the     \\\n"
   "  file will need to be re-compiled every time an operator is changed or added. \\\n"
   "  Consider including a specific operator from                                  \\\n"
   "  <ATen/ops/{my_operator}_${dispatch_namespace}_dispatch.h>.                   \\\n"
   "  See NOTE [TORCH_ASSERT_ONLY_METHOD_OPERATORS].\n"
   "#endif\n"
   "\n"
   "${DispatchKeyFunctions_inl_includes}\n"
   "\n"
   "\n"
   "${dispatch_namespaced_declarations}\n"))

(define (get-namespaced-declaration grouped dispatch-key backend-index)
  ;; torchgen gen.get_namespaced_declaration: gather the NAMESPACED_DECLARATION
  ;; text for every item, dedupe preserving order, and wrap in the single
  ;; "at::{lower}" namespace.  0 dispatch values in the frozen corpus carry a
  ;; custom "::" namespace, so get_kernel_namespace(...).replace("native",
  ;; lower) is always "at::{lower}" and there is exactly one namespace.
  (let* ((lower (string-downcase dispatch-key))
         (kernels (append-map
                   (lambda (item)
                     (gen-dispatch item dispatch-key backend-index
                                   'namespaced-declaration))
                   grouped)))
    (if (null? kernels)
        '()
        (let* ((prologue (string-append "namespace at {\nnamespace " lower " {"))
               (epilogue (string-append "} // namespace " lower
                                        "\n} // namespace at"))
               (ordered (delete-duplicates kernels string=?)))
          ;; Exact .split("\n") of "\n{prologue}\n{join}\n{epilogue}\n        ".
          (string-split
           (string-append "\n" prologue "\n" (string-join ordered "\n")
                          "\n" epilogue "\n" "        ")
           #\newline)))))

(define (render-dispatch-key-functions-h dispatch-key)
  ;; {dispatch_key}Functions.h: DispatchKeyFunctions.h template with the single
  ;; inline_headers value "#include <ATen/{key}Functions_inl.h>".
  (code-template-substitute
   %dispatch-key-functions-template
   (lambda (key)
     (cond
      ((string=? key "inline_headers")
       (string-append "#include <ATen/" dispatch-key "Functions_inl.h>"))
      (else (error 'unknown-functions-h-key key))))))

(define (render-dispatch-key-functions-inl-h grouped dispatch-key backend-index)
  ;; {dispatch_key}Functions_inl.h: DispatchKeyFunctions_inl.h template.
  (code-template-substitute
   %dispatch-key-functions-inl-template
   (lambda (key)
     (cond
      ((string=? key "generated_comment")
       (string-append "@generated by " %generator-path
                      " from DispatchKeyFunctions_inl.h"))
      ((string=? key "DispatchKeyFunctions_inl_includes") '())
      ((string=? key "dispatch_namespace") (string-downcase dispatch-key))
      ((string=? key "dispatch_namespaced_declarations")
       (get-namespaced-declaration grouped dispatch-key backend-index))
      (else (error 'unknown-functions-inl-key key))))))

(define (render-dispatch-key-functions grouped indices)
  ;; returns ((filename . content) ...) sorted by filename, 7 keys x 2 files.
  (sort
   (append-map
    (lambda (dispatch-key)
      (let ((backend-index (make-backend-index dispatch-key indices)))
        (list
         (cons (string-append dispatch-key "Functions.h")
               (render-dispatch-key-functions-h dispatch-key))
         (cons (string-append dispatch-key "Functions_inl.h")
               (render-dispatch-key-functions-inl-h grouped dispatch-key
                                                    backend-index)))))
    functions-keys)
   (lambda (a b) (string<? (car a) (car b)))))
