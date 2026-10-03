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

;; G13: RegisterFunctionalization.cpp (sharded x4).  Faithful port of torchgen
;; gen_functionalization_type.py:
;;
;;   - gen_functionalization_definition  -> ${func_definitions}
;;   - gen_functionalization_registration -> ${func_registrations}
;;   - gen_op_headers                    -> ${ops_headers}
;;
;; driven over `all_groups` (structured + view groups + ungrouped native
;; functions) exactly as gen.py's `cpu_fm.write_sharded(..., num_shards=4,
;; key_fn=lambda x: x.root_name, sharded_keys={ops_headers, func_definitions,
;; func_registrations, func_add_back_views_definitions,
;; func_add_back_views_registrations})`.  The rendered C++ is byte-identical to
;; `torchgen/gen.py` at 41ffbc4a994e058af9fe00ed5caba73fc1033359.

(define-module (sonic-cross functionalization)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-13)
  #:use-module (sonic-cross core-ir)
  #:use-module (sonic-cross native-function)
  #:use-module (sonic-cross generated-functions)
  #:use-module (sonic-cross code-template)
  #:use-module (sonic-cross orchestration)
  #:use-module (sonic-cross file-manager)
  #:use-module ((sonic-cross api) #:prefix api:)
  #:export (render-register-functionalization-files
            render-functional-inverses
            render-view-meta-classes-h
            render-view-meta-classes-cpp))

(define %generator-path "torchgen/gen.py")

;; ---------------------------------------------------------------------------
;; frozen constants (native_function_generation.py, redefined locally)
;; ---------------------------------------------------------------------------

(define inplace-ops-that-dont-get-grouped-properly '("polygamma_"))

(define out-ops-that-dont-get-grouped-properly
  '("adaptive_avg_pool3d_backward.grad_input"
    "_slow_conv2d_backward.grad_input"))

(define mutable-ops-that-cannot-get-an-out-variant
  '("_cummax_helper" "_cummin_helper"))

(define mutable-ops-not-using-functionalization
  (append out-ops-that-dont-get-grouped-properly
          mutable-ops-that-cannot-get-an-out-variant
          inplace-ops-that-dont-get-grouped-properly
          '("record_stream" "resize_" "resize_as_"
            "_fill_mem_eff_dropout_mask_"
            "_flash_attention_forward_no_dropout_inplace")))

(define cumulative-out-ops-preserving-out-dtype '("cumsum.out" "cumprod.out"))

;; ---------------------------------------------------------------------------
;; small helpers (gen_functionalization_type.py)
;; ---------------------------------------------------------------------------

(define (wrapper-name func)
  ;; cpp.name(func) + "_<overload>" when an overload is present.
  (let ((n (api:cpp-name func)))
    (let ((overload (operator-name-overload-name (function-schema-name func))))
      (if (string-null? overload) n (string-append n "_" overload)))))

(define (modifies-arguments? f)
  (any argument-is-write?
       (arguments-all (function-schema-arguments (native-function-func f)))))

(define (functionalization-return-str returns names use-const-ref)
  (unless (= (length returns) (length names))
    (error 'return-str-length-mismatch (length returns) (length names)))
  (cond
   ((null? returns) "")
   ((null? (cdr returns)) (string-append "return " (car names) ";"))
   (else
    (string-append "return "
                   (api:ctype-cpp-type
                    (api:dispatcher-returns-type returns #t use-const-ref))
                   "(" (string-join names ", ") ");"))))

;; ---------------------------------------------------------------------------
;; FunctionSchema.aliased_return_names()
;; ---------------------------------------------------------------------------

(define (aliased-return-names func)
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
;; type predicates
;; ---------------------------------------------------------------------------

(define (type-is-list-like? t)
  (case (type-kind t)
    ((list) t)
    ((optional) (type-is-list-like? (car (type-arguments t))))
    (else #f)))

;; ---------------------------------------------------------------------------
;; get_owning_type
;; ---------------------------------------------------------------------------

(define (get-owning-type t)
  (cond
   ((equal? t (api:make-base-ctype api:%tensor-list-t))
    (values (api:make-vector-ctype (api:make-base-ctype api:%tensor-t))
            (lambda (x) (string-append x ".vec()"))))
   ((equal? t (api:make-base-ctype api:%i-tensor-list-ref-t))
    (values (api:make-vector-ctype (api:make-base-ctype api:%tensor-t))
            (lambda (x) (string-append "{" x ".begin(), " x ".end()}"))))
   (else (values t (lambda (x) x)))))

;; ---------------------------------------------------------------------------
;; unwrap_tensor_args / convert_to_meta_tensors
;; ---------------------------------------------------------------------------

;; Dispatch the tensor-likeness test against the JIT (flat_all) argument that
;; backs each dispatcher binding; bindings carry no `argument` field here.
(define (dispatcher-jit-args func)
  (api:dispatcher-jit-arguments func))

(define (unwrap-tensor-args func is-view-op use-const-ref use-ilistref)
  (let ((jit-args (dispatcher-jit-args func))
        (sig-args (api:dispatcher-signature-arguments func #t use-const-ref use-ilistref)))
    (let loop ((jargs jit-args) (sargs sig-args) (ctx '()) (unwrapped '()))
      (if (null? sargs)
          (values (string-join (reverse unwrapped) "\n      ") (reverse ctx))
          (let* ((binding (car sargs))
                 (jarg (car jargs))
                 (argname (api:binding-name binding))
                 (nct (api:binding-nctype binding)))
            (if (type-is-tensor-like? (argument-type jarg))
                (let* ((unwrapped-name (string-append argname "_"))
                       (maybe-sync
                        (if is-view-op
                            ""
                            (string-append "at::functionalization::impl::sync(" argname ");")))
                       (raw-type
                        (api:named-ctype-type (api:named-ctype-remove-const-ref nct))))
                  (call-with-values
                      (lambda () (get-owning-type raw-type))
                    (lambda (unwrapped-type conversion-fn)
                      (loop (cdr jargs) (cdr sargs)
                            (cons (api:binding-with-name binding unwrapped-name) ctx)
                            (cons
                             (string-append
                              "\n      " (api:ctype-cpp-type unwrapped-type) " " unwrapped-name ";\n"
                              "      if (at::functionalization::impl::isFunctionalTensor(" argname ")) {\n"
                              "        " maybe-sync "\n"
                              "        " unwrapped-name " = at::functionalization::impl::from_functional_tensor(" argname ");\n"
                              "      } else {\n"
                              "        " unwrapped-name " = " (conversion-fn argname) ";\n"
                              "      }")
                             unwrapped)))))
                (loop (cdr jargs) (cdr sargs)
                      (cons binding ctx) unwrapped)))))))

(define (convert-to-meta-tensors func use-const-ref use-ilistref)
  (let ((jit-args (dispatcher-jit-args func))
        (sig-args (api:dispatcher-signature-arguments func #t use-const-ref use-ilistref)))
    (let loop ((jargs jit-args) (sargs sig-args) (ctx '()) (elts '()))
      (if (null? sargs)
          (values (string-join (reverse elts) "\n        ") (reverse ctx))
          (let* ((binding (car sargs))
                 (jarg (car jargs))
                 (name (api:binding-name binding)))
            (if (type-is-tensor-like? (argument-type jarg))
                (let ((meta-name (string-append name "_meta")))
                  (loop (cdr jargs) (cdr sargs)
                        (cons (api:binding-with-name binding meta-name) ctx)
                        (cons (string-append "auto " meta-name " = to_meta(" name ");") elts)))
                (loop (cdr jargs) (cdr sargs) (cons binding ctx) elts)))))))

;; ---------------------------------------------------------------------------
;; emit_expr_has_symbolic_values / emit_has_symbolic_inputs
;; ---------------------------------------------------------------------------

(define (emit-expr-has-symbolic-values expr type)
  (cond
   ((equal? type (api:make-base-ctype api:%sym-int-t))
    (string-append expr ".is_symbolic()"))
   ((api:optional-ctype? type)
    (string-append expr ".has_value() ? "
                   (emit-expr-has-symbolic-values
                    (string-append "(*" expr ")")
                    (api:optional-ctype-elem type))
                   " : false"))
   ((equal? type (api:make-base-ctype api:%optional-sym-int-array-ref-t))
    (emit-expr-has-symbolic-values
     expr (api:make-optional-ctype (api:make-base-ctype api:%sym-int-array-ref-t))))
   ((or (equal? type (api:make-base-ctype api:%sym-int-array-ref-t))
        (equal? type (api:make-vector-ctype (api:make-base-ctype api:%sym-int-t))))
    (let* ((argname "arg")
           (lambda-check (emit-expr-has-symbolic-values
                          argname (api:make-base-ctype api:%sym-int-t))))
      (string-append "std::any_of(" expr ".begin(), " expr ".end(), "
                     "[=](auto& " argname ") { return " lambda-check "; })")))
   (else (error 'unsupported-symbolic-values-type (api:ctype-cpp-type type)))))

(define (emit-has-symbolic-inputs func use-const-ref use-ilistref)
  (let* ((name "has_symbolic_inputs")
         (jit-args (dispatcher-jit-args func))
         (sig-args (api:dispatcher-signature-arguments func #t use-const-ref use-ilistref)))
    (let ((statements
           (filter-map
            (lambda (binding jarg)
              (and (type-is-symint-like? (argument-type jarg))
                   (string-append name " = " name " | ("
                                  (emit-expr-has-symbolic-values
                                   (api:binding-name binding)
                                   (api:named-ctype-type (api:binding-nctype binding)))
                                  ");")))
            sig-args jit-args)))
      (string-append "\n      bool " name " = false;\n      "
                     (string-join statements "\n      ")))))

;; ---------------------------------------------------------------------------
;; signature defn helpers
;; ---------------------------------------------------------------------------

(define (redispatch-defn func name use-const-ref use-ilistref)
  ;; DispatcherSignature.defn(name=..., is_redispatching_fn=True).
  (string-append
   (api:ctype-cpp-type (api:dispatcher-signature-returns-type func #t use-const-ref))
   " " name "(c10::DispatchKeySet dispatchKeySet, "
   (string-join
    (map api:binding-defn
         (api:dispatcher-signature-arguments func #t use-const-ref use-ilistref))
    ", ")
   ")"))

(define (native-signature-ptr-type func symint use-const-ref use-ilistref)
  ;; NativeSignature.ptr_type(): "{returns} (*)({args defn})".
  (string-append
   (api:ctype-cpp-type (api:native-signature-returns-type func symint use-const-ref))
   " (*)("
   (string-join
    (map api:binding-defn
         (api:native-signature-arguments func symint use-const-ref use-ilistref))
    ", ")
   ")"))

;; ---------------------------------------------------------------------------
;; ViewMeta helper (functionalization.py)
;; ---------------------------------------------------------------------------

(define (functionalization-is-multi-output func)
  (let ((returns (function-schema-returns func)))
    (or (> (length returns) 1)
        (and (= (length returns) 1)
             (type-is-list-like? (return-type (car returns)))))))

(define (functionalization-classname func with-namespace)
  (string-append (if with-namespace "at::functionalization::" "")
                 (unambiguous-name func) "_ViewMeta"))

(define (view-meta-new func)
  ;; ViewMetaSpecialization.new(): the has_symbolic_inputs (+ out_index if
  ;; multi-output) + reapply_views/inverse_return_mode + arg names of args[1:].
  (string-append
   "std::make_shared<" (functionalization-classname func #t) ">("
   (string-join
    (append (list "has_symbolic_inputs")
            (if (functionalization-is-multi-output func) (list "0") '())
            (list "reapply_views" "inverse_return_mode")
            (map argument-name
                 (cdr (arguments-all (function-schema-arguments func)))))
    ", ")
   ")"))

;; ---------------------------------------------------------------------------
;; emit_view_functionalization_body
;; ---------------------------------------------------------------------------

(define (emit-view-functionalization-body g view-inplace?)
  (let* ((view (native-functions-view-group-view g))
         (view-inplace (native-functions-view-group-view-inplace g))
         (view-copy (native-functions-view-group-view-copy g))
         (f (if view-inplace? view-inplace view))
         (f-func (native-function-func f))
         (view-copy-func (native-function-func view-copy))
         (use-const-ref (api:native-function-use-const-ref-context f))
         (use-ilistref (api:native-function-use-ilistref-context f))
         (api-name (unambiguous-name view-copy-func))
         (noop-api-name (unambiguous-name f-func))
         (view-tensor-name
          (api:binding-name
           (car (api:dispatcher-signature-arguments
                 f-func #t use-const-ref use-ilistref))))
         (return-type
          (api:ctype-cpp-type
           (api:ctype-remove-const-ref
            (api:dispatcher-signature-returns-type f-func #t use-const-ref))))
         (defn (redispatch-defn f-func (wrapper-name f-func)
                                use-const-ref use-ilistref))
         (symbolic-inputs-check
          (emit-has-symbolic-inputs view-copy-func use-const-ref use-ilistref))
         (view-meta-str (view-meta-new f-func))
         (view-copy-sig-args
          (api:dispatcher-signature-arguments
           view-copy-func #t use-const-ref use-ilistref)))
    (call-with-values
        (lambda () (unwrap-tensor-args f-func #t use-const-ref use-ilistref))
      (lambda (unwrap-tensor-args-str unwrapped-ctx)
        (let* ((view-redispatch-args
                (map api:expr-expr (api:translate unwrapped-ctx view-copy-sig-args)))
               (view-redispatch-args-str (string-join view-redispatch-args ", ")))
          (call-with-values
              (lambda () (convert-to-meta-tensors f-func use-const-ref use-ilistref))
            (lambda (meta-conversion-str meta-call-ctx)
              (let ((meta-call-args-str
                     (string-join
                      (map api:expr-expr (api:translate meta-call-ctx view-copy-sig-args))
                      ", ")))
                (if (member "inplace_view" (native-function-tags f))
                    (string-append
                     "\n    " defn " {\n"
                     "      if (!at::functionalization::impl::isFunctionalTensor(" view-tensor-name ")) {\n"
                     "        // functionalization is re-entrant, but will no-op if it wasn't passed a FunctionalTensorWrapper.\n"
                     "        " unwrap-tensor-args-str "\n"
                     "        at::AutoDispatchSkipFunctionalize guard;\n"
                     "        return at::_ops::" noop-api-name "::call(" view-redispatch-args-str ");\n"
                     "      }\n"
                     "      auto reapply_views = at::functionalization::impl::getFunctionalizationReapplyViewsTLS();\n"
                     "      auto inverse_return_mode = (\n"
                     "          reapply_views ? at::functionalization::InverseReturnMode::ViewOrScatterInverse\n"
                     "            : at::functionalization::InverseReturnMode::NeverView\n"
                     "      );\n"
                     "      " symbolic-inputs-check "\n"
                     "      auto view_meta = " view-meta-str ";\n"
                     "      auto compute_reference_meta =\n"
                     "        " view-tensor-name ".key_set().has_backend(c10::BackendComponent::XLABit) ||\n"
                     "        " view-tensor-name ".key_set().has_backend(c10::BackendComponent::LazyBit);\n"
                     "      " return-type " reference_tensor_output;\n"
                     "      if (compute_reference_meta && !disable_meta_reference()) {\n"
                     "        " meta-conversion-str "\n"
                     "        at::AutoDispatchSkipFunctionalize func_guard;\n"
                     "        c10::impl::ExcludeDispatchKeyGuard guard(exclude_keys_for_meta_dispatch);\n"
                     "        reference_tensor_output = at::_ops::" noop-api-name "::call(" meta-call-args-str ");\n"
                     "      }\n"
                     "      // This function adds the above view meta to the current tensor and replays them off the base,\n"
                     "      // mutating the size/stride info of the current FunctionalTensorWrapper.\n"
                     "      // Because of this, we need to make sure to run the reference shape function above,\n"
                     "      // BEFORE doing this (otherwise we'll end up running the reference function using the wrong sizes/strides)\n"
                     "      at::functionalization::impl::mutate_view_meta(" view-tensor-name ", view_meta);\n"
                     "      // See  Note [Propagating strides in the functionalization pass]\n"
                     "      // XLA/LTC don't implement the logic to propagate strides correctly, so we need to rely\n"
                     "      // on a reference implementation here (instead of relying on the output from the forward lambda\n"
                     "      // having the correct stride info)\n"
                     "      if (compute_reference_meta && !disable_meta_reference()) {\n"
                     "        at::functionalization::impl::set_sizes_strides_offset(" view-tensor-name ", reference_tensor_output);\n"
                     "      }\n"
                     "      return " view-tensor-name ";\n"
                     "    }\n")
                    (string-append
                     "\n    " defn " {\n"
                     "      " unwrap-tensor-args-str "\n"
                     "      if (!at::functionalization::impl::isFunctionalTensor(" view-tensor-name ")) {\n"
                     "        // functionalization is re-entrant, but will no-op if it wasn't passed a FunctionalTensorWrapper.\n"
                     "        at::AutoDispatchSkipFunctionalize guard;\n"
                     "        return at::_ops::" noop-api-name "::call(" view-redispatch-args-str ");\n"
                     "      }\n"
                     "      auto reapply_views = at::functionalization::impl::getFunctionalizationReapplyViewsTLS();\n"
                     "      auto inverse_return_mode = (\n"
                     "          reapply_views ? at::functionalization::InverseReturnMode::ViewOrScatterInverse\n"
                     "            : at::functionalization::InverseReturnMode::NeverView\n"
                     "      );\n"
                     "      auto compute_reference_meta =\n"
                     "        " view-tensor-name ".key_set().has_backend(c10::BackendComponent::XLABit) ||\n"
                     "        " view-tensor-name ".key_set().has_backend(c10::BackendComponent::LazyBit);\n"
                     "      " return-type " reference_tensor_output;\n"
                     "      if (compute_reference_meta && !disable_meta_reference()) {\n"
                     "        " meta-conversion-str "\n"
                     "        at::AutoDispatchSkipFunctionalize func_guard;\n"
                     "        c10::impl::ExcludeDispatchKeyGuard guard(exclude_keys_for_meta_dispatch);\n"
                     "        reference_tensor_output = at::_ops::" noop-api-name "::call(" meta-call-args-str ");\n"
                     "      }\n"
                     "      " return-type " tmp_output;\n"
                     "      {\n"
                     "        at::AutoDispatchSkipFunctionalize guard;\n"
                     "        if (reapply_views) {\n"
                     "          tmp_output = at::_ops::" noop-api-name "::call(" view-redispatch-args-str ");\n"
                     "        } else {\n"
                     "          tmp_output = at::_ops::" api-name "::call(" view-redispatch-args-str ");\n"
                     "        }\n"
                     "      }\n"
                     "      " symbolic-inputs-check "\n"
                     "      auto view_meta = " view-meta-str ";\n"
                     "      auto out = at::functionalization::impl::create_functional_tensor_with_view_meta(tmp_output, " view-tensor-name ", view_meta);\n"
                     "      // See  Note [Propagating strides in the functionalization pass]\n"
                     "      if (compute_reference_meta && !disable_meta_reference()) {\n"
                     "        at::functionalization::impl::set_sizes_strides_offset(out, reference_tensor_output);\n"
                     "      }\n"
                     "      return out;\n"
                     "    }\n"))))))))))

;; ---------------------------------------------------------------------------
;; emit_inplace_functionalization_body helpers
;; ---------------------------------------------------------------------------

(define (mutable-arg-names f)
  (map argument-name
       (filter argument-is-write?
               (arguments-all (function-schema-arguments (native-function-func f))))))

(define (get-mutable-redispatch-return-names f inner-return-var)
  (let* ((func (native-function-func f))
         (returns (function-schema-returns func))
         (aliased (aliased-return-names func)))
    (let loop ((i 0) (names aliased) (aliased-out '()) (non-aliased-out '()))
      (if (null? names)
          (values (reverse aliased-out) (reverse non-aliased-out))
          (let ((name (car names)))
            (if name
                (loop (1+ i) (cdr names) (cons name aliased-out) non-aliased-out)
                (loop (1+ i) (cdr names) aliased-out
                      (cons (if (= (length returns) 1)
                                inner-return-var
                                (string-append "std::get<" (number->string i)
                                               ">(" inner-return-var ")"))
                            non-aliased-out))))))))

(define (maybe-create-output f var-name use-const-ref)
  (let ((returns (function-schema-returns (native-function-func f))))
    (if (null? returns)
        ""
        (string-append
         (api:ctype-cpp-type
          (api:ctype-remove-const-ref
           (api:dispatcher-returns-type returns #t use-const-ref)))
         " " var-name " = "))))

(define (return-from-mutable-noop-redispatch f inner-var use-const-ref)
  (call-with-values
      (lambda () (get-mutable-redispatch-return-names f inner-var))
    (lambda (aliased non-aliased)
      (functionalization-return-str
       (function-schema-returns (native-function-func f))
       (append aliased non-aliased)
       use-const-ref))))

(define (wrap-propagate-mutations-and-return f functional-op inner-var use-const-ref)
  (let* ((func (native-function-func f))
         (functional-func (native-function-func functional-op))
         (mutable-args (mutable-arg-names f)))
    (call-with-values
        (lambda () (get-mutable-redispatch-return-names f inner-var))
      (lambda (aliased-outer non-aliased-outer)
        (call-with-values
            (lambda () (get-mutable-redispatch-return-names functional-op inner-var))
          (lambda (_ non-aliased-inner)
            (let* ((n-outer (length non-aliased-outer))
                   (updates
                    (append
                     (map (lambda (i inner-ret)
                            (string-append
                             "  auto output_" (number->string i)
                             " = at::functionalization::impl::to_functional_tensor("
                             inner-ret ");"))
                          (iota n-outer)
                          (take non-aliased-inner n-outer))
                     (map (lambda (outer-arg inner-ret)
                            (string-append
                             "  auto " outer-arg "_inner = at::functionalization::impl::from_functional_tensor(" outer-arg ");\n"
                             "  at::functionalization::impl::replace_(" outer-arg ", " inner-ret ");\n"
                             "  at::functionalization::impl::commit_update(" outer-arg ");\n"
                             "  at::functionalization::impl::sync(" outer-arg ");\n"
                             "  auto " outer-arg "_inner_updated = at::functionalization::impl::from_functional_tensor(" outer-arg ");\n"
                             "  at::functionalization::impl::propagate_xla_data_direct(" outer-arg "_inner, " outer-arg "_inner_updated);"))
                          mutable-args
                          (drop non-aliased-inner n-outer))))
                   (wrapped-names
                    (map (lambda (i) (string-append "output_" (number->string i)))
                         (iota n-outer)))
                   (returns-str
                    (functionalization-return-str
                     (function-schema-returns func)
                     (append aliased-outer wrapped-names)
                     use-const-ref)))
              (string-append (string-join updates "\n")
                             "\n    " returns-str))))))))

(define (list-replace lst idx val)
  (let loop ((i 0) (xs lst) (acc '()))
    (cond
     ((null? xs) (reverse acc))
     ((= i idx) (loop (1+ i) (cdr xs) (cons val acc)))
     (else (loop (1+ i) (cdr xs) (cons (car xs) acc))))))

(define (maybe-replace-cumulative-out-dtype-exprs f functional-sig-args functional-exprs)
  (let* ((func (native-function-func f))
         (out-args (arguments-out (function-schema-arguments func))))
    (if (and (eq? (function-schema-kind func) schema-kind-out)
             (member (operator-name->string (function-schema-name func))
                     cumulative-out-ops-preserving-out-dtype))
        (let ((dtype-idx
               (list-index (lambda (binding)
                             (string=? (api:binding-name binding) "dtype"))
                           functional-sig-args)))
          (unless dtype-idx
            (error 'missing-dtype (operator-name->string (function-schema-name func))))
          (let* ((out-name (argument-name (car out-args)))
                 (dtype-expr (list-ref functional-exprs dtype-idx))
                 (replacement
                  (string-append dtype-expr ".has_value() ? " dtype-expr " : "
                                 "std::optional<at::ScalarType>(" out-name "_.scalar_type())")))
            (list-replace functional-exprs dtype-idx replacement)))
        functional-exprs)))

;; ---------------------------------------------------------------------------
;; emit_inplace_functionalization_body
;; ---------------------------------------------------------------------------

(define (emit-inplace-functionalization-body f g)
  (unless (modifies-arguments? f)
    (error 'expected-mutating-function
           (operator-name->string (function-schema-name (native-function-func f)))))
  (let* ((func (native-function-func f))
         (functional (native-functions-group-functional g))
         (functional-func (native-function-func functional))
         (use-const-ref (api:native-function-use-const-ref-context f))
         (use-ilistref (api:native-function-use-ilistref-context f))
         (jit-args (dispatcher-jit-args func))
         (sig-args (api:dispatcher-signature-arguments func #t use-const-ref use-ilistref))
         (functional-sig-args
          (api:dispatcher-signature-arguments functional-func #t use-const-ref use-ilistref))
         (mutated-names
          (map argument-name
               (filter (lambda (a)
                         (and (type-is-tensor-like? (argument-type a))
                              (argument-annotation a)))
                       jit-args)))
         (non-mutated-names
          (map argument-name
               (filter (lambda (a)
                         (and (type-is-tensor-like? (argument-type a))
                              (not (argument-annotation a))))
                       jit-args)))
         (non-mutated-tensor-names
          (map argument-name
               (filter (lambda (a)
                         (and (equal? (argument-type a) tensor-type)
                              (not (argument-annotation a))))
                       jit-args)))
         (check-all-mutated
          (string-join
           (cons "true"
                 (map (lambda (a)
                        (string-append "at::functionalization::impl::isFunctionalTensor(" a ")"))
                      mutated-names))
           " && "))
         (check-any-non-mutated
          (string-join
           (cons "false"
                 (map (lambda (a)
                        (string-append "at::functionalization::impl::isFunctionalTensor(" a ")"))
                      non-mutated-names))
           " || "))
         (check-any-non-mutated-xla
          (string-join
           (cons "false"
                 (map (lambda (a)
                        (string-append a ".device().type() == c10::DeviceType::XLA"))
                      non-mutated-tensor-names))
           " || "))
         (return-type
          (api:ctype-cpp-type
           (api:ctype-remove-const-ref
            (api:dispatcher-returns-type
             (function-schema-returns functional-func) #t use-const-ref))))
         (any-storage-args
          (any (lambda (a)
                 (equal? (argument-type a) (make-base-type "Storage")))
               jit-args))
         (defn (redispatch-defn func (wrapper-name func) use-const-ref use-ilistref))
         (inplace-bool (if (and (not any-storage-args)
                                (eq? (function-schema-kind func) schema-kind-inplace))
                           "true" "false")))
    (call-with-values
        (lambda () (unwrap-tensor-args func #f use-const-ref use-ilistref))
      (lambda (unwrap-tensor-args-str unwrapped-ctx)
        (call-with-values
            (lambda () (convert-to-meta-tensors func use-const-ref use-ilistref))
          (lambda (meta-conversion-str meta-call-ctx)
            (let* ((inplace-exprs
                    (map api:expr-expr (api:translate unwrapped-ctx sig-args)))
                   (functional-exprs
                    (maybe-replace-cumulative-out-dtype-exprs
                     f functional-sig-args
                     (map api:expr-expr (api:translate unwrapped-ctx functional-sig-args))))
                   (meta-call-names-str
                    (string-join (map api:binding-name meta-call-ctx) ", "))
                   (noop-create (maybe-create-output f "tmp_output" use-const-ref))
                   (noop-return (return-from-mutable-noop-redispatch f "tmp_output" use-const-ref))
                   (wrap (wrap-propagate-mutations-and-return f functional "tmp_output" use-const-ref))
                   (api-name (unambiguous-name func))
                   (functional-api-name (unambiguous-name functional-func)))
              (string-append
               "\n    " defn " {\n"
               "      if (" inplace-bool " && !disable_meta_reference()) {\n"
               "        // Before converting the mutable op to its functional variant, run meta tensors through the original op.\n"
               "        // This will help us catch shape errors that apply to inplace ops that wouldn't apply to their functional variants.\n"
               "        // (We can only do this for inplace ops today though, because they technically all support meta tensors).\n"
               "        " meta-conversion-str "\n"
               "        at::AutoDispatchSkipFunctionalize func_guard;\n"
               "        c10::impl::ExcludeDispatchKeyGuard guard(exclude_keys_for_meta_dispatch);\n"
               "        at::_ops::" api-name "::call(" meta-call-names-str ");\n"
               "      }\n"
               "      " unwrap-tensor-args-str "\n"
               "      if (!(" check-all-mutated ")) {\n"
               "        // We want to disable this check if there are any XLA tensors.\n"
               "        // cpu_tensor.copy_(xla_tensor) is valid code.\n"
               "        if (!(" check-any-non-mutated-xla ") && (" check-any-non-mutated ")) {\n"
               "         // case 1: trying to mutate a non functional tensor with a functional tensor is an error\n"
               "         TORCH_INTERNAL_ASSERT(false,\n"
               "           \"mutating a non-functional tensor with a functional tensor is not allowed.\",\n"
               "           \" Please ensure that all of your inputs are wrapped inside of a functionalize() call.\");\n"
               "        } else {\n"
               "         // case 2: arguments are not functional tensors, so we no-op and redispatch.\n"
               "         at::AutoDispatchSkipFunctionalize guard;\n"
               "         " noop-create "at::_ops::" api-name "::call(" (string-join inplace-exprs ", ") ");\n"
               "         " noop-return "\n"
               "        }\n"
               "      } else {\n"
               "        " return-type " tmp_output;\n"
               "        {\n"
               "          at::AutoDispatchSkipFunctionalize guard;\n"
               "          tmp_output = at::_ops::" functional-api-name "::call(" (string-join functional-exprs ", ") ");\n"
               "        }\n"
               "        " wrap "\n"
               "      }\n"
               "    }\n"))))))))

;; ---------------------------------------------------------------------------
;; gen_functionalization_definition
;; ---------------------------------------------------------------------------

(define (gen-functionalization-definition g)
  (cond
   ((native-functions-view-group? g)
    (if (native-functions-view-group-composite? g)
        '()
        (let ((view-copy (native-functions-view-group-view-copy g)))
          (unless view-copy
            (error 'expected-view-copy
                   (operator-name->string
                    (function-schema-name
                     (native-function-func (native-functions-view-group-view g))))))
          (append
           (list (emit-view-functionalization-body g #f))
           (let ((view-inplace (native-functions-view-group-view-inplace g)))
             (if view-inplace
                 (list (emit-view-functionalization-body g #t))
                 '()))))))
   ((native-functions-group? g)
    (append
     (list (emit-inplace-functionalization-body (native-functions-group-out g) g))
     (let ((inplace (native-functions-group-inplace g)))
       (if inplace (list (emit-inplace-functionalization-body inplace g)) '()))
     (let ((mutable (native-functions-group-mutable g)))
       (if mutable (list (emit-inplace-functionalization-body mutable g)) '()))))
   (else
    ;; ungrouped NativeFunction: only error-checking, no definitions.
    (let ((name (operator-name->string
                 (function-schema-name (native-function-func g)))))
      (when (and (not (member name mutable-ops-not-using-functionalization))
                 (not (member (base-operator-spelling (native-function-func g))
                              mutable-ops-not-using-functionalization)))
        (unless (or (native-function-has-composite-implicit-autograd-kernel? g)
                    (not (modifies-arguments? g)))
          (error 'expected-composite-or-non-modifying name))))
    '())))

;; ---------------------------------------------------------------------------
;; gen_functionalization_registration
;; ---------------------------------------------------------------------------

(define (emit-registration-helper f composite-implicit-autograd-index)
  (let* ((func (native-function-func f))
         (use-const-ref (api:native-function-use-const-ref-context f))
         (use-ilistref (api:native-function-use-ilistref-context f))
         (name (operator-name->string (function-schema-name func))))
    (if (native-function-has-composite-implicit-autograd-kernel? f)
        (let ((metadata (api:backend-index-get-kernel f composite-implicit-autograd-index)))
          (unless metadata
            (error 'missing-composite-kernel name))
          (string-append
           "m.impl(\"" name "\", static_cast<"
           (native-signature-ptr-type func
                                      (backend-metadata-supports-symint? metadata)
                                      use-const-ref use-ilistref)
           ">(at::native::" (backend-metadata-kernel metadata) "));"))
        (string-append
         "m.impl(\"" name "\", TORCH_FN(functionalization::"
         (wrapper-name func) "));"))))

(define (gen-functionalization-registration g composite-implicit-autograd-index)
  (cond
   ((native-functions-view-group? g)
    (if (string=? (operator-name->string
                   (function-schema-name
                    (native-function-func (native-functions-view-group-view g))))
                  "lift_fresh")
        '()
        (append
         (list (emit-registration-helper (native-functions-view-group-view g)
                                         composite-implicit-autograd-index))
         (let ((view-inplace (native-functions-view-group-view-inplace g)))
           (if view-inplace
               (begin
                 (unless (native-function-is-view-op? view-inplace)
                   (error 'expected-view-inplace
                          (operator-name->string
                           (function-schema-name (native-function-func view-inplace)))))
                 (list (emit-registration-helper view-inplace
                                                 composite-implicit-autograd-index)))
               '())))))
   (else
    (let ((fns
           (cond
            ((native-functions-group? g)
             (let ((inplace (native-functions-group-inplace g)))
               (if (and inplace
                        (string=? (operator-name->string
                                   (function-schema-name (native-function-func inplace)))
                                  "set_.source_Tensor"))
                   '()
                   (native-functions-group-functions g))))
            (else
             (let ((name (operator-name->string
                          (function-schema-name (native-function-func g)))))
               (if (member name mutable-ops-not-using-functionalization)
                   '()
                   (list g)))))))
      (let loop ((fns fns) (regs '()))
        (if (null? fns)
            (reverse regs)
            (let* ((f (car fns))
                   (func (native-function-func f))
                   (name (operator-name->string (function-schema-name func)))
                   (base (base-operator-spelling func)))
              (cond
               ((native-function-has-composite-implicit-autograd-kernel? f)
                (loop (cdr fns) regs))
               ((string=? name "lift") '())
               ((string=? name "resize_") '())
               (else
                (unless (string=? base "set_")
                  (when (native-function-is-view-op? f)
                    (error 'unexpected-view-op name)))
                (if (modifies-arguments? f)
                    (loop (cdr fns)
                          (cons (emit-registration-helper
                                 f composite-implicit-autograd-index)
                                regs))
                    (loop (cdr fns) regs)))))))))))

;; ---------------------------------------------------------------------------
;; gen_op_headers
;; ---------------------------------------------------------------------------

(define (root-name-of f)
  (base-operator-name-base
   (operator-name-base (function-schema-name (native-function-func f)))))

(define (op-header-lines root-name)
  (list (string-append "#include <ATen/ops/" root-name "_native.h>")
        (string-append "#include <ATen/ops/" root-name "_ops.h>")))

(define (gen-op-headers g)
  (cond
   ((native-functions-view-group? g)
    (let ((view (native-functions-view-group-view g))
          (view-copy (native-functions-view-group-view-copy g)))
      (append (op-header-lines (root-name-of view))
              (if view-copy (op-header-lines (root-name-of view-copy)) '()))))
   ((native-functions-group? g)
    (append (op-header-lines (root-name-of (native-functions-group-functional g)))
            (op-header-lines (root-name-of (native-functions-group-out g)))
            (let ((inplace (native-functions-group-inplace g)))
              (if inplace (op-header-lines (root-name-of inplace)) '()))
            (let ((mutable (native-functions-group-mutable g)))
              (if mutable (op-header-lines (root-name-of mutable)) '()))))
   (else
    (op-header-lines (root-name-of g)))))

;; ---------------------------------------------------------------------------
;; template + sharded render
;; ---------------------------------------------------------------------------

(define %register-functionalization-template
  (string-append
   (string-join
    (list
     "#define TORCH_ASSERT_ONLY_METHOD_OPERATORS"
     "// ${generated_comment}"
     ""
     "#include <ATen/core/LegacyTypeDispatch.h>"
     "#include <ATen/EmptyTensor.h>"
     "#include <ATen/FunctionalTensorWrapper.h>"
     "#include <ATen/ViewMetaClasses.h>"
     "#include <ATen/MemoryOverlap.h>"
     "#include <torch/library.h>"
     ""
     "#include <c10/util/env.h>"
     "#ifndef AT_PER_OPERATOR_HEADERS"
     "#include <ATen/Operators.h>"
     "#include <ATen/NativeFunctions.h>"
     "#else"
     "// needed for the meta tensor calls to get stride info in functionalization"
     "#include <ATen/ops/empty_strided_native.h>"
     "// needed for special handling of copy_()."
     "// See Note [functionalizating copy_() and not preserving strides]"
     "#include <ATen/ops/to_ops.h>"
     "#include <ATen/ops/expand_copy_ops.h>"
     ""
     "$ops_headers"
     "#endif"
     ""
     "namespace at {"
     "namespace functionalization {"
     ""
     "// This keyset is used by functionalization when it calls into meta kernels"
     "// to accurately propagate stride metadata."
     "// Exclude any modes: the purpose of calling into meta kernels is only as an implementation"
     "// detail to perform shape inference, and we don't want any modal keys to run."
     "// Specifically, we want to prevent functionalization and Python modes from running."
     "constexpr auto exclude_keys_for_meta_dispatch ="
     "    c10::functorch_transforms_ks |"
     "    c10::DispatchKeySet({"
     "        c10::DispatchKey::FuncTorchDynamicLayerBackMode,"
     "        c10::DispatchKey::FuncTorchDynamicLayerFrontMode,"
     "        c10::DispatchKey::Python,"
     "        c10::DispatchKey::PreDispatch,"
     ""
     "    });"
     ""
     "// Helper around at::has_internal_overlap."
     "// The ATen util is used in hot-path eager mode: it's always fast,"
     "// but might return TOO_HARD sometimes."
     "// During functionalization, we're ok taking a bit longer"
     "// to detect memory overlap."
     "inline bool has_internal_overlap_helper(const at::Tensor t) {"
     "  auto has_overlap = at::has_internal_overlap(t);"
     "  if (has_overlap == at::MemOverlap::Yes) return true;"
     "  if (has_overlap == at::MemOverlap::No) return false;"
     "  return false;"
     "}"
     ""
     ""
     "inline Tensor to_meta(const Tensor& t) {"
     "    if (!t.defined()) return t;"
     "    return at::native::empty_strided_meta_symint(t.sym_sizes(), t.sym_strides(),"
     "/*dtype=*/t.scalar_type(), /*layout=*/t.layout(),"
     "/*device=*/c10::Device(kMeta), /*pin_memory=*/std::nullopt);"
     "}"
     ""
     "inline std::optional<Tensor> to_meta(const std::optional<Tensor>& t) {"
     "  if (t.has_value()) {"
     "    return to_meta(*t);"
     "  }"
     "  return std::nullopt;"
     "}"
     ""
     "inline std::vector<Tensor> to_meta(at::ITensorListRef t_list) {"
     "  std::vector<Tensor> outputs;"
     "  outputs.reserve(t_list.size());"
     "  for (const auto& tensor : t_list) {"
     "    outputs.push_back(to_meta(tensor));"
     "  }"
     "  return outputs;"
     "}"
     ""
     "inline c10::List<Tensor> to_meta(const c10::List<Tensor>& t_list) {"
     "  c10::List<Tensor> outputs;"
     "  outputs.reserve(t_list.size());"
     "  for (const auto i : c10::irange(t_list.size())) {"
     "    outputs.push_back(to_meta(t_list[i]));"
     "  }"
     "  return outputs;"
     "}"
     ""
     "inline c10::List<::std::optional<Tensor>> to_meta(const c10::List<::std::optional<Tensor>>& t_list) {"
     "  c10::List<::std::optional<Tensor>> outputs;"
     "  outputs.reserve(t_list.size());"
     "  for (const auto i : c10::irange(t_list.size())) {"
     "    outputs.push_back(to_meta(t_list[i]));"
     "  }"
     "  return outputs;"
     "}"
     ""
     "static bool disable_meta_reference() {"
     "  static auto env = c10::utils::get_env(\"TORCH_DISABLE_FUNCTIONALIZATION_META_REFERENCE\");"
     "  return env == \"1\";"
     "}"
     ""
     ""
     "${func_definitions}"
     ""
     "}  // namespace functionalization"
     ""
     "namespace {"
     ""
     "TORCH_LIBRARY_IMPL(aten, Functionalize, m) {"
     "  ${func_registrations};"
     "}"
     ""
     "}  // namespace"
     ""
     "} // namespace at")
    "\n")
   "\n"))

(define (render-register-functionalization-files all-groups composite-implicit-autograd-index)
  ;; returns ((suffix . content) ...) for "Everything" and "_0".."_3".
  (let* ((num-shards 4)
         (shard-ids (cons "Everything"
                          (map (lambda (i) (string-append "_" (number->string i)))
                               (iota num-shards))))
         (shard->env (make-hash-table)))
    (for-each
     (lambda (sid)
       (hash-set! shard->env sid
                  (list (cons "ops_headers" '())
                        (cons "func_definitions" '())
                        (cons "func_registrations" '()))))
     shard-ids)
    (for-each
     (lambda (item)
       (let* ((root-name (item-root-name item))
              (sid (number->string (shard-index root-name num-shards)))
              (suffix (string-append "_" sid))
              (env (list (cons "ops_headers" (gen-op-headers item))
                         (cons "func_definitions"
                               (gen-functionalization-definition item))
                         (cons "func_registrations"
                               (gen-functionalization-registration
                                item composite-implicit-autograd-index)))))
         (for-each
          (lambda (shard-suffix)
            (let ((acc (hash-ref shard->env shard-suffix)))
              (hash-set! shard->env shard-suffix
                         (map (lambda (entry)
                                (let ((key (car entry)))
                                  (cons key
                                        (append (cdr entry)
                                                (or (assoc-ref env key) '())))))
                              acc))))
          (list suffix "Everything"))))
     all-groups)
    (map
     (lambda (sid)
       (let* ((env (hash-ref shard->env sid))
              (content
               (code-template-substitute
                %register-functionalization-template
                (lambda (key)
                  (cond
                   ((string=? key "generated_comment")
                    (string-append "@generated by " %generator-path
                                   " from RegisterFunctionalization.cpp"))
                   ((string=? key "ops_headers") (assoc-ref env "ops_headers"))
                   ((string=? key "func_definitions") (assoc-ref env "func_definitions"))
                   ((string=? key "func_registrations") (assoc-ref env "func_registrations"))
                   (else (error 'unknown-template-key key)))))))
         (cons sid content)))
     shard-ids)))

;; ---------------------------------------------------------------------------
;; G14/G15: FunctionalInverses.h + ViewMetaClasses.h/.cpp
;; ---------------------------------------------------------------------------

;; Synthetic bindings (torchgen api/functionalization.py).
(define %inverse-return-mode-t "at::functionalization::InverseReturnMode")

(define (functionalization-base-binding)
  (api:make-binding "base"
                    (api:make-named-ctype "base"
                                          (api:make-const-ref-ctype (api:make-base-ctype api:%tensor-t)))
                    #f))

(define (functionalization-has-symbolic-inputs-binding)
  (api:make-binding "has_symbolic_inputs"
                    (api:make-named-ctype "has_symbolic_inputs" (api:make-base-ctype api:%bool-t))
                    #f))

(define (functionalization-mutated-view-binding)
  (api:make-binding "mutated_view"
                    (api:make-named-ctype "mutated_view"
                                          (api:make-const-ref-ctype (api:make-base-ctype api:%tensor-t)))
                    #f))

(define (functionalization-out-index-binding)
  (api:make-binding "out_index"
                    (api:make-named-ctype "out_index" (api:make-base-ctype api:%long-t))
                    #f))

(define (functionalization-reapply-views-binding)
  (api:make-binding "reapply_views"
                    (api:make-named-ctype "reapply_views" (api:make-base-ctype api:%bool-t))
                    #f))

(define (functionalization-inverse-return-mode-binding)
  (api:make-binding "inverse_return_mode"
                    (api:make-named-ctype "inverse_return_mode" (api:make-base-ctype %inverse-return-mode-t))
                    #f))

(define (functionalization-reverse-name func include-namespace)
  ;; functionalization.reverse_name(f, include_namespace).
  (let ((api-name (unambiguous-name func)))
    (if include-namespace
        (string-append "at::functionalization::FunctionalInverses::" api-name "_inverse")
        (string-append api-name "_inverse"))))

(define (functionalization-op-name g is-reverse reapply-views)
  ;; functionalization.name(g, is_reverse, include_namespace=True, reapply_views).
  (if is-reverse
      (functionalization-reverse-name
       (native-function-func (native-functions-view-group-view g)) #t)
      (let* ((view (native-functions-view-group-view g))
             (view-copy (native-functions-view-group-view-copy g))
             (f (if reapply-views view view-copy))
             (api-name (unambiguous-name (native-function-func f))))
        (string-append "at::_ops::" api-name "::call"))))

(define (functionalization-base-ctor-arguments func)
  (append (list (functionalization-has-symbolic-inputs-binding))
          (if (functionalization-is-multi-output func)
              (list (functionalization-out-index-binding))
              '())))

(define (functionalization-attributes func owning use-const-ref use-ilistref)
  (append (list (functionalization-reapply-views-binding)
                (functionalization-inverse-return-mode-binding))
          (map (lambda (a)
                 (api:dispatcher-argument a owning #t use-const-ref use-ilistref))
               (cdr (arguments-all (function-schema-arguments func))))))

(define (functionalization-extra-ctor-arguments func use-const-ref use-ilistref)
  (functionalization-attributes func #f use-const-ref use-ilistref))

(define (functionalization-op-arguments func is-reverse use-const-ref use-ilistref)
  (let* ((args (arguments-all (function-schema-arguments func)))
         (non-self-args (cdr args))
         (non-self-bindings
          (map (lambda (a) (api:dispatcher-argument a #f #t use-const-ref use-ilistref))
               non-self-args)))
    (if is-reverse
        (append (list (functionalization-base-binding)
                      (functionalization-mutated-view-binding)
                      (functionalization-inverse-return-mode-binding))
                (if (functionalization-is-multi-output func)
                    (cons (functionalization-out-index-binding) non-self-bindings)
                    non-self-bindings))
        (cons (functionalization-base-binding) non-self-bindings))))

;; ---------------------------------------------------------------------------
;; G14: FunctionalInverses.h  (gen_functionalization_view_inverse_declaration)
;; ---------------------------------------------------------------------------

(define (gen-functionalization-view-inverse-declaration g)
  (let ((view (native-functions-view-group-view g)))
    (if (native-function-has-composite-implicit-autograd-kernel? view)
        #f
        (let* ((func (native-function-func view))
               (use-const-ref (api:native-function-use-const-ref-context view))
               (use-ilistref (api:native-function-use-ilistref-context view))
               (decls (map api:binding-decl
                           (functionalization-op-arguments func #t
                                                           use-const-ref use-ilistref))))
          (string-append "static at::Tensor "
                         (functionalization-reverse-name func #f)
                         "(" (string-join decls ", ") ");")))))

;; ---------------------------------------------------------------------------
;; G15: ViewMetaClasses.h/.cpp  (ViewMetaSpecialization)
;; ---------------------------------------------------------------------------

(define (view-meta-new-with-out-index func out-index)
  ;; ViewMetaSpecialization.new(out_index): base ctor args + extra ctor args,
  ;; with the out_index parameter replaced by the given out_index.
  (string-append
   "std::make_shared<" (functionalization-classname func #t) ">("
   (string-join
    (append (list "has_symbolic_inputs")
            (if (functionalization-is-multi-output func) (list out-index) '())
            (list "reapply_views" "inverse_return_mode")
            (map argument-name (cdr (arguments-all (function-schema-arguments func)))))
    ", ")
   ")"))

(define (view-meta-opcall g f is-reverse reapply-views use-const-ref use-ilistref)
  ;; ViewMetaSpecialization.opcall(is_reverse, reapply_views).
  (let* ((func (native-function-func f))
         (view-copy-func
          (native-function-func (native-functions-view-group-view-copy g)))
         (opname (functionalization-op-name g is-reverse reapply-views))
         (op-arguments
          (functionalization-op-arguments view-copy-func is-reverse
                                          use-const-ref use-ilistref))
         (context
          (append (list (functionalization-base-binding))
                  (functionalization-base-ctor-arguments func)
                  (functionalization-attributes func #t use-const-ref use-ilistref)
                  (if is-reverse (list (functionalization-mutated-view-binding)) '())))
         (arguments
          (string-join (map api:expr-expr (api:translate context op-arguments)) ", "))
         (maybe-index
          (if (and (not is-reverse) (functionalization-is-multi-output func))
              "[out_index]"
              "")))
    (string-append opname "(" arguments ")" maybe-index)))

(define (view-meta-decl g f use-const-ref use-ilistref)
  ;; ViewMetaSpecialization.decl().
  (let* ((func (native-function-func f))
         (classname (functionalization-classname func #f))
         (is-multi-output (functionalization-is-multi-output func))
         (is-as-strided
          (string=? (operator-name->string (function-schema-name func)) "as_strided"))
         (base-ctor-arguments (functionalization-base-ctor-arguments func))
         (extra-ctor-arguments
          (functionalization-extra-ctor-arguments func use-const-ref use-ilistref))
         (attributes (functionalization-attributes func #t use-const-ref use-ilistref))
         (serializable-tuple-args
          (string-join
           (map (lambda (b)
                  (string-append "      " (api:binding-type b)
                                 " /* " (api:binding-name b) " */"))
                (append base-ctor-arguments attributes))
           ",\n"))
         (destructure-tuple-args
          (string-join
           (map (lambda (i) (string-append "std::get<" (number->string i) ">(tpl)"))
                (iota (length (append base-ctor-arguments extra-ctor-arguments))))
           ", "))
         (ctor-parameters
          (string-join (map api:binding-decl (append base-ctor-arguments extra-ctor-arguments))
                       ", "))
         (base-ctor-bindings
          (string-append
           "has_symbolic_inputs, /*is_multi_output=*/" (if is-multi-output "true" "false")
           ", /*is_as_strided=*/" (if is-as-strided "true" "false")
           ", /*out_index=*/" (if is-multi-output "out_index" "0")))
         (ctor-assignments
          (string-join
           (map (lambda (e)
                  (string-append "        " (api:named-ctype-name (api:expr-type e))
                                 "(" (api:expr-expr e) ")"))
                (api:translate extra-ctor-arguments attributes
                               #:allow-expensive-conversions #t))
           ",\n"))
         (tuple-arguments
          (string-join (map api:binding-name (append base-ctor-arguments attributes)) ", "))
         (attr-declarations
          (string-join (map (lambda (b) (string-append "  " (api:binding-decl b) ";"))
                            attributes)
                       "\n"))
         (to-out-index-decl
          (if is-multi-output
              "  std::shared_ptr<ViewMeta> to_out_index(int64_t out_idx) override;"
              "")))
    (list
     (string-append
      "\nstruct TORCH_API " classname " : public ViewMeta {\n"
      "  FUNCTIONALIZATION_VIEWMETA_NAME(" classname ")\n"
      "  FUNCTIONALIZATION_VIEWMETA_SERIALIZABLE_TUPLE(\n" serializable-tuple-args ");\n"
      "\n"
      "  " classname "(const SerializableTuple& tpl)\n"
      "      : " classname "(" destructure-tuple-args ") {}\n"
      "\n"
      "  " classname "(" ctor-parameters ")\n"
      "      : at::functionalization::ViewMeta(" base-ctor-bindings "),\n"
      ctor-assignments " {}\n"
      "\n"
      "  Tensor forward(const Tensor& base) override;\n"
      "  Tensor reverse(const Tensor& base, const Tensor& mutated_view) override;\n"
      to-out-index-decl "\n"
      "\n"
      "  SerializableTuple to_serializable_tuple() {\n"
      "    return std::make_tuple(" tuple-arguments ");\n"
      "  }\n"
      "\n"
      attr-declarations "\n"
      "};\n"))))

(define (view-meta-impl g f use-const-ref use-ilistref)
  ;; ViewMetaSpecialization.impl().
  (let* ((func (native-function-func f))
         (classname (functionalization-classname func #f))
         (is-multi-output (functionalization-is-multi-output func)))
    (append
     (list
      (string-append
       "\nat::Tensor " classname "::forward(const at::Tensor& base) {\n"
       "  if (reapply_views) {\n"
       "    return " (view-meta-opcall g f #f #t use-const-ref use-ilistref) ";\n"
       "  } else {\n"
       "    return " (view-meta-opcall g f #f #f use-const-ref use-ilistref) ";\n"
       "  }\n"
       "}")
      (string-append
       "\nat::Tensor " classname "::reverse(const at::Tensor& base, const Tensor& mutated_view) {\n"
       "  return " (view-meta-opcall g f #t #t use-const-ref use-ilistref) ";\n"
       "}"))
     (if is-multi-output
         (list
          (string-append
           "\nstd::shared_ptr<at::functionalization::ViewMeta> " classname "::to_out_index(int64_t out_index) {\n"
           "  return " (view-meta-new-with-out-index func "out_index") ";\n"
           "}\n"))
         '()))))

(define (view-meta-map g run)
  ;; ViewMetaSpecialization.map(g, run): run over g.view and g.view_inplace.
  (append
   (let ((view (native-functions-view-group-view g)))
     (run g view
          (api:native-function-use-const-ref-context view)
          (api:native-function-use-ilistref-context view)))
   (let ((view-inplace (native-functions-view-group-view-inplace g)))
     (if view-inplace
         (run g view-inplace
              (api:native-function-use-const-ref-context view-inplace)
              (api:native-function-use-ilistref-context view-inplace))
         '()))))

(define (gen-functionalization-view-meta-classes-base g run)
  (if (native-functions-view-group-composite? g)
      '()
      (view-meta-map g run)))

(define (gen-functionalization-view-meta-classes-decl g)
  (gen-functionalization-view-meta-classes-base g view-meta-decl))

(define (gen-functionalization-view-meta-classes-impl g)
  (gen-functionalization-view-meta-classes-base g view-meta-impl))

;; ---------------------------------------------------------------------------
;; templates + render
;; ---------------------------------------------------------------------------

(define %functional-inverses-template
  (string-append
   (string-join
    (list
     "#pragma once"
     ""
     "// ${generated_comment}"
     ""
     "#include <ATen/FunctionalStorageImpl.h>"
     "#include <ATen/Tensor.h>"
     ""
     "namespace at {"
     "namespace functionalization {"
     ""
     "struct FunctionalInverses {"
     ""
     "${view_inverse_declarations}"
     ""
     "// NB: These are not generated! They're manually implemented in the template."
     "// TODO: Change codegen to generate these. See the following link:"
     "// https://github.com/pytorch/pytorch/blob/main/torchgen/model.py#L2583-L2585"
     "static at::Tensor chunk_inverse(const at::Tensor & base, const at::Tensor & mutated_view, InverseReturnMode inverse_return_mode, int64_t mutated_view_idx, int chunks, int dim);"
     "static at::Tensor narrow_inverse(const at::Tensor & base, const at::Tensor & mutated_view, InverseReturnMode inverse_return_mode, int dim, c10::SymInt start, c10::SymInt length);"
     ""
     "};"
     "}"
     "}")
    "\n")
   "\n"))

(define %view-meta-classes-h-template
  (string-append
   (string-join
    (list
     "#define TORCH_ASSERT_ONLY_METHOD_OPERATORS"
     "// ${generated_comment}"
     ""
     "#include <ATen/FunctionalStorageImpl.h>"
     ""
     "namespace at {"
     "namespace functionalization {"
     ""
     "${view_meta_declarations}"
     ""
     "} // namespace functionalization"
     "} // namespace at")
    "\n")
   "\n"))

(define %view-meta-classes-cpp-template
  (string-append
   (string-join
    (list
     "// ${generated_comment}"
     ""
     "#include <ATen/FunctionalInverses.h>"
     "#include <ATen/ViewMetaClasses.h>"
     ""
     "#ifndef AT_PER_OPERATOR_HEADERS"
     "#include <ATen/Operators.h>"
     "#include <ATen/NativeFunctions.h>"
     "#else"
     "${op_headers}"
     "#endif"
     ""
     "namespace at {"
     "namespace functionalization {"
     ""
     "${view_meta_implementations}"
     ""
     "} // namespace functionalization"
     "} // namespace at")
    "\n")
   "\n"))

(define (render-functional-inverses view-groups)
  (code-template-substitute
   %functional-inverses-template
   (lambda (key)
     (cond
      ((string=? key "generated_comment")
       (string-append "@generated by " %generator-path " from FunctionalInverses.h"))
      ((string=? key "view_inverse_declarations")
       (filter-map gen-functionalization-view-inverse-declaration view-groups))
      (else (error 'unknown-template-key key))))))

(define (render-view-meta-classes-h view-groups)
  (code-template-substitute
   %view-meta-classes-h-template
   (lambda (key)
     (cond
      ((string=? key "generated_comment")
       (string-append "@generated by " %generator-path " from ViewMetaClasses.h"))
      ((string=? key "view_meta_declarations")
       (append-map gen-functionalization-view-meta-classes-decl view-groups))
      (else (error 'unknown-template-key key))))))

(define (render-view-meta-classes-cpp view-groups)
  (code-template-substitute
   %view-meta-classes-cpp-template
   (lambda (key)
     (cond
      ((string=? key "generated_comment")
       (string-append "@generated by " %generator-path " from ViewMetaClasses.cpp"))
      ((string=? key "view_meta_implementations")
       (append-map gen-functionalization-view-meta-classes-impl view-groups))
      ((string=? key "op_headers")
       (append-map gen-op-headers view-groups))
      (else (error 'unknown-template-key key))))))
