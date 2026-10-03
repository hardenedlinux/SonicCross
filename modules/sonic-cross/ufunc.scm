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

;; G10: structured ufunc kernels (torchgen dest/ufunc.py + api/ufunc.py).
;;
;; Renders UfuncCPU_{name}.cpp, UfuncCPUKernel_{name}.cpp and
;; UfuncCUDA_{name}.cu for each structured operator with a non-empty
;; ufunc_inner_loop block (only `add.out` in the frozen 2.14 manifest).

(define-module (sonic-cross ufunc)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-13)
  #:use-module (sonic-cross core-ir)
  #:use-module (sonic-cross native-function)
  #:use-module (sonic-cross generated-functions)
  #:use-module (sonic-cross yaml)
  #:use-module (sonic-cross code-template)
  #:use-module ((sonic-cross api) #:prefix api:)
  #:use-module (sonic-cross emitter)
  #:export (render-ufunc-cpu-cpp
            render-ufunc-cpu-kernel-cpp
            render-ufunc-cuda-cu))

;; ---------------------------------------------------------------------------
;; model: UfuncInnerLoop parse + DTYPE_CLASSES expansion (torchgen model.py).
;; ---------------------------------------------------------------------------

(define %dtype-classes
  '(("Integral" . ("Byte" "Char" "Int" "Long" "Short"))
    ("Floating" . ("Float" "Double"))
    ("Complex" . ("ComplexFloat" "ComplexDouble"))
    ("All" . ("Byte" "Char" "Int" "Long" "Short" "Float" "Double"))
    ("AllAndComplex" . ("Byte" "Char" "Int" "Long" "Short" "Float" "Double"
                        "ComplexFloat" "ComplexDouble"))
    ("FloatingAndComplex" . ("Float" "Double" "ComplexFloat" "ComplexDouble"))))

(define (expand-dtype-name name)
  ;; ScalarType.parse_set: a DTYPE_CLASSES key expands, otherwise it is a bare
  ;; ScalarType name.
  (let ((entry (assoc-ref %dtype-classes name)))
    (if entry entry (list name))))

(define (parse-ufunc-inner-loop-value s)
  ;; "add (AllAndComplex, BFloat16, Half, ComplexHalf)" -> ("add" . (dtype ...))
  ;; UfuncInnerLoop.parse splits on the first " " and then on ", ".
  (let* ((space (string-index s #\space))
         (name (substring s 0 space))
         (dtypes-str (substring s (+ space 1)))
         (inner (substring dtypes-str 1 (- (string-length dtypes-str) 1)))
         (dtypes (map string-trim-both (string-split inner #\,))))
    (cons name dtypes)))

(define (parse-ufunc-inner-loop raw)
  ;; raw :: <ufunc-inner-loop-placeholder>.  Returns alist
  ;; ((key-string . (name . (dtype ...))) ...) in YAML order.
  (map (lambda (entry)
         (let* ((key (yaml-scalar-value (car entry)))
                (value (yaml-scalar-value (cadr entry))))
           (cons key (parse-ufunc-inner-loop-value value))))
       (yaml-mapping-entries (ufunc-inner-loop-placeholder-raw raw))))

(define (loop-supported-dtypes loop-entry)
  ;; loop-entry :: (name . (dtype-or-class ...)) -> (dtype ...) expanded in order.
  (append-map expand-dtype-name (cdr loop-entry)))

(define (ufunc-name loops)
  ;; The Generic / ScalarOnly name must agree; return it.
  (let ((generic (assoc-ref loops "Generic"))
        (scalar-only (assoc-ref loops "ScalarOnly")))
    (let ((name (or (and generic (car generic))
                    (and scalar-only (car scalar-only)))))
      (unless name (error 'ufunc-name-missing loops))
      name)))

(define (ufunc-functor-dtypes loops)
  ;; OrderedSet(ScalarOnly) | OrderedSet(Generic), first-occurrence order.
  (let ((scalar-only (assoc-ref loops "ScalarOnly"))
        (generic (assoc-ref loops "Generic")))
    (delete-duplicates
     (append (if scalar-only (loop-supported-dtypes scalar-only) '())
             (if generic (loop-supported-dtypes generic) '()))
     string=?)))

;; ---------------------------------------------------------------------------
;; api/ufunc.py: names + argument conversion.
;; ---------------------------------------------------------------------------

(define (ufunc-base-name g)
  ;; str(BaseOperatorName) of the functional variant ("add").
  (base-operator-spelling
   (native-function-func (native-functions-group-functional g))))

(define (ufunc-kernel-name g key)
  ;; schema_kernel_name: f"ufunc_{func.name.name}_{dispatch_key}".
  (string-append "ufunc_" (ufunc-base-name g) "_" key))

(define (stub-arguments g)
  ;; Stubs drop all tensor arguments (implicit in TensorIterator) and keep the
  ;; rest, converted via structured.argument.
  (append-map
   api:structured-argument
   (filter (lambda (a) (not (type-is-tensor-like? (argument-type a))))
           (arguments-flat-non-out
            (function-schema-arguments
             (native-function-func (native-functions-group-out g)))))))

(define (ufunc-type t binds compute-t)
  ;; api/ufunc.py ufunc_type: value types pass through; Scalar/Tensor -> compute_t.
  (or (api:cpp-valuetype-type t binds #f #f)
      (case (type-kind t)
        ((scalar tensor) (api:make-named-ctype binds compute-t))
        (else (error 'ufunc-type t)))))

(define (ufunc-argument a compute-t)
  (api:make-binding (argument-name a)
                    (ufunc-type (argument-type a) (argument-name a) compute-t)
                    #f))

(define (ufunc-arguments g compute-t)
  (map (lambda (a) (ufunc-argument a compute-t))
       (arguments-flat-non-out
        (function-schema-arguments
         (native-function-func (native-functions-group-functional g))))))

(define (ufunctor-ctor-type t binds)
  ;; Scalar/Tensor ctor args widen to opmath_t; value types pass through.
  (or (api:cpp-valuetype-type t binds #f #f)
      (case (type-kind t)
        ((scalar tensor) (api:make-named-ctype binds (api:make-base-ctype "opmath_t")))
        (else (error 'ufunctor-ctor-type t)))))

(define (ufunctor-ctor-argument a)
  (api:make-binding (argument-name a)
                    (ufunctor-ctor-type (argument-type a) (argument-name a))
                    #f))

(define (ufunctor-apply-argument a)
  (api:make-binding (argument-name a)
                    (api:make-named-ctype (argument-name a)
                                          (api:make-base-ctype "scalar_t"))
                    #f))

(define (ufunctor-arguments g scalar-tensor-idx)
  ;; api/ufunc.py ufunctor_arguments -> (ctor . apply).  scalar-tensor-idx is
  ;; either an integer index or #f (CUDAFunctor has no scalar tensor).
  (let loop ((args (arguments-flat-non-out
                    (function-schema-arguments
                     (native-function-func (native-functions-group-functional g)))))
             (idx scalar-tensor-idx)
             (ctor '())
             (apply '()))
    (if (null? args)
        (begin
          (when (number? idx)
            (error 'ufunctor-scalar-tensor-idx-unresolved g scalar-tensor-idx))
          (cons (reverse ctor) (reverse apply)))
        (let* ((a (car args))
               (t (argument-type a)))
          (if (type-is-tensor-like? t)
              (if (and (number? idx) (= idx 0))
                  (loop (cdr args) #f
                        (cons (ufunctor-ctor-argument a) ctor) apply)
                  (loop (cdr args) (and (number? idx) (- idx 1))
                        ctor (cons (ufunctor-apply-argument a) apply)))
              (loop (cdr args) idx
                    (cons (ufunctor-ctor-argument a) ctor) apply))))))

(define (ufunc-call ufunc-name ctx goals)
  ;; UfuncSignature.call: f"{name}({translate(ctx, arguments())})".
  (string-append ufunc-name "("
                 (string-join (map api:expr-expr (api:translate ctx goals)) ", ")
                 ")"))

;; ---------------------------------------------------------------------------
;; dest/ufunc.py: StubSignature + StructuredImplSignature.
;; ---------------------------------------------------------------------------

(define (stub-name g) (string-append (ufunc-base-name g) "_stub"))
(define (stub-kernel-name g) (string-append (ufunc-base-name g) "_kernel"))
(define (stub-type-name g) (string-append (ufunc-base-name g) "_fn"))

(define (stub-type g)
  (string-append "void(*)(TensorIteratorBase&, "
                 (string-join (map api:binding-type (stub-arguments g)) ", ")
                 ")"))

(define (stub-type-defn g)
  (string-append "using " (stub-type-name g) " = " (stub-type g)))

(define (stub-dispatch-decl g)
  (string-append "DECLARE_DISPATCH(" (stub-type-name g) ", " (stub-name g) ")"))

(define (stub-dispatch-defn g)
  (string-append "DEFINE_DISPATCH(" (stub-name g) ")"))

(define (stub-kernel-defn g)
  (string-append "void " (stub-kernel-name g) "(TensorIteratorBase& iter, "
                 (string-join (map api:binding-defn (stub-arguments g)) ", ")
                 ")"))

(define (stub-call g ctx)
  (string-append (stub-name g) "(device_type(), *this, "
                 (string-join
                  (map api:expr-expr (api:translate ctx (stub-arguments g))) ", ")
                 ")"))

(define (stub-direct-call g ctx)
  (string-append (stub-kernel-name g) "(*this, "
                 (string-join
                  (map api:expr-expr (api:translate ctx (stub-arguments g))) ", ")
                 ")"))

(define (structured-impl-defn g name)
  ;; StructuredImplSignature.defn: TORCH_IMPL_FUNC({name})({impl args defn}).
  (string-append "TORCH_IMPL_FUNC(" name ")("
                 (string-join (map api:binding-defn (api:structured-impl-arguments g))
                              ", ")
                 ")"))

;; ---------------------------------------------------------------------------
;; dest/ufunc.py: UfunctorSignature (CUDA functor structs).
;; ---------------------------------------------------------------------------

(define (make-ufunctor-sig name scalar-tensor-idx ctor apply)
  (list name scalar-tensor-idx ctor apply))
(define (ufunctor-sig-name s) (list-ref s 0))
(define (ufunctor-sig-ctor s) (list-ref s 2))
(define (ufunctor-sig-apply s) (list-ref s 3))

(define (rename-binding b suffix)
  ;; Binding.rename changes only the name, not the nctype.
  (api:make-binding (string-append (api:binding-name b) suffix)
                    (api:binding-nctype b)
                    (api:binding-default b)))

(define (ufunctor-sig-fields s)
  (map (lambda (b) (rename-binding b "_")) (ufunctor-sig-ctor s)))

(define (ufunctor-decl-fields s)
  (string-join (map (lambda (f)
                      (string-append (api:binding-type f) " " (api:binding-name f) ";"))
                    (ufunctor-sig-fields s))
               "\n"))

(define (ufunctor-inline-defn-ctor s)
  (let* ((ctor (ufunctor-sig-ctor s))
         (args-str (string-join (map api:binding-decl ctor) ", "))
         (init-str (string-join
                    (map (lambda (a)
                           (string-append (api:binding-name a) "_("
                                          (api:binding-name a) ")"))
                         ctor)
                    ", ")))
    (string-append (ufunctor-sig-name s) "(" args-str ") : " init-str " {}")))

(define (ufunctor-decl-apply s)
  (string-append "scalar_t operator()("
                 (string-join (map api:binding-decl (ufunctor-sig-apply s)) ", ")
                 ") const"))

(define (ufunctor-to-string s g)
  ;; The functor struct body, verbatim (dest/ufunc.py compute_ufunc_cuda_functors).
  (let* ((ufunc-name (string-append "ufunc::" (ufunc-base-name g)))
         (apply-ctx (append (ufunctor-sig-fields s) (ufunctor-sig-apply s)))
         (call (ufunc-call ufunc-name apply-ctx
                           (ufunc-arguments g (api:make-base-ctype "opmath_t")))))
    (string-append
     "\ntemplate <typename scalar_t>\n"
     "struct " (ufunctor-sig-name s) " {\n"
     "  using opmath_t = at::opmath_type<scalar_t>;\n"
     "  " (ufunctor-decl-fields s) "\n"
     "  " (ufunctor-inline-defn-ctor s) "\n"
     "  __device__ " (ufunctor-decl-apply s) " {\n"
     "    return " call ";\n"
     "  }\n"
     "};\n")))

;; ---------------------------------------------------------------------------
;; dest/ufunc.py: CUDA.
;; ---------------------------------------------------------------------------

(define (eligible-for-binary-scalar-specialization? g)
  (= (length (filter (lambda (a) (type-is-tensor-like? (argument-type a)))
                     (arguments-flat-non-out
                      (function-schema-arguments
                       (native-function-func (native-functions-group-functional g))))))
     2))

(define (build-cuda-functors g)
  ;; Returns (values ufunctor-sigs ufunctors-str).  ufunctor-sigs ::
  ;; ((dtype . ((functor-key . ufunctor-sig) ...)) ...).
  (let* ((loops (parse-ufunc-inner-loop
                 (native-function-ufunc-inner-loop (native-functions-group-out g))))
         (idx-lookup '(("CUDAFunctorOnSelf" . 1)
                       ("CUDAFunctorOnOther" . 0)
                       ("CUDAFunctor" . #f)))
         (keys '("CUDAFunctorOnSelf" "CUDAFunctorOnOther" "CUDAFunctor"))
         (ufunc-name (ufunc-name loops))
         (supported-dtypes (ufunc-functor-dtypes loops))
         (sigs (map (lambda (k)
                      (let* ((idx (assoc-ref idx-lookup k))
                             (bindings (ufunctor-arguments g idx)))
                        (cons k (make-ufunctor-sig
                                 (string-append k "_" ufunc-name)
                                 idx (car bindings) (cdr bindings)))))
                    keys)))
    (values
     (map (lambda (dtype) (cons dtype sigs)) supported-dtypes)
     (string-join (map (lambda (e) (ufunctor-to-string (cdr e) g)) sigs) "\n"))))

(define (compute-ufunc-cuda-dtype-body g inner-loops parent-ctx)
  ;; body built by string accumulation; see dest/ufunc.py.
  (let ((configs '(("CUDAFunctorOnOther" . (0 . "self"))
                   ("CUDAFunctorOnSelf" . (1 . "other")))))
    (let loop ((cfgs configs)
               (body (string-append "using opmath_t = at::opmath_type<scalar_t>;"
                                    "if (false) {}\n")))
      (if (null? cfgs)
          (let* ((fsig (assoc-ref inner-loops "CUDAFunctor"))
                 (ctor-exprs (string-join
                              (map api:expr-expr
                                   (api:translate parent-ctx (ufunctor-sig-ctor fsig)))
                              ", ")))
            (string-append body
                           "\nelse {\n"
                           "  gpu_kernel(iter, " (ufunctor-sig-name fsig)
                           "<scalar_t>(" ctor-exprs "));\n"
                           "}\n"
                           "    "))
          (let* ((cfg (car cfgs))
                 (key (car cfg))
                 (scalar-idx0 (car (cdr cfg)))
                 (ctor-tensor (cdr (cdr cfg)))
                 (fsig (assoc-ref inner-loops key)))
            (if fsig
                (let* ((scalar-idx (+ scalar-idx0 1))
                       (ctx (append
                             parent-ctx
                             (list (api:make-expr
                                    (string-append "iter.scalar_value<opmath_t>("
                                                   (number->string scalar-idx) ")")
                                    (api:make-named-ctype
                                     ctor-tensor (api:make-base-ctype "opmath_t"))))))
                       (ctor-exprs (string-join
                                    (map api:expr-expr
                                         (api:translate ctx (ufunctor-sig-ctor fsig)))
                                    ", ")))
                  (loop (cdr cfgs)
                        (string-append
                         body
                         "else if (iter.is_cpu_scalar(" (number->string scalar-idx) ")) {\n"
                         "  " (ufunctor-sig-name fsig) "<scalar_t> ufunctor(" ctor-exprs ");\n"
                         "  iter.remove_operand(" (number->string scalar-idx) ");\n"
                         "  gpu_kernel(iter, ufunctor);\n"
                         "}")))
                (loop (cdr cfgs) body)))))))

(define (compute-ufunc-cuda g)
  (let ((sig-name (ufunc-kernel-name g "CUDA"))
        (sig-args (api:structured-impl-arguments g)))
    (call-with-values (lambda () (build-cuda-functors g))
      (lambda (ufunctor-sigs ufunctors)
        (let* ((dtype-cases (map (lambda (entry)
                                   (string-append
                                    "\nAT_DISPATCH_CASE(at::ScalarType::" (car entry) ",\n"
                                    "  [&]() {\n"
                                    "    " (compute-ufunc-cuda-dtype-body
                                             g (cdr entry) sig-args) "\n"
                                    "  }\n"
                                    ")\n"))
                                 ufunctor-sigs))
               (dtype-cases-str (string-join dtype-cases "\n")))
          (string-append
           "\n" ufunctors "\n\n"
           (stub-type-defn g) ";\n"
           (stub-dispatch-decl g) "\n\n"
           (stub-kernel-defn g) " {\n"
           "  AT_DISPATCH_SWITCH(iter.common_dtype(), \"" sig-name "\",\n"
           "    " dtype-cases-str "\n"
           "  );\n"
           "}\n"
           "REGISTER_DISPATCH(" (stub-name g) ", &" (stub-kernel-name g) ")\n\n"
           (structured-impl-defn g sig-name) " {\n"
           "  " (stub-direct-call g sig-args) ";\n"
           "}\n"))))))

;; ---------------------------------------------------------------------------
;; dest/ufunc.py: CPU.
;; ---------------------------------------------------------------------------

(define (compute-ufunc-cpu g)
  (let ((sig-name (ufunc-kernel-name g "CPU"))
        (sig-args (api:structured-impl-arguments g)))
    (string-append
     "\n" (stub-type-defn g) ";\n"
     (stub-dispatch-decl g) "\n"
     (stub-dispatch-defn g) ";\n\n"
     (structured-impl-defn g sig-name) " {\n"
     "  " (stub-call g sig-args) ";\n"
     "}\n")))

(define (scalar-ref-binding? b)
  (equal? (api:named-ctype-type (api:binding-nctype b))
          (api:make-const-ref-ctype (api:make-base-ctype api:%scalar-t))))

(define (cpu-tensor-bindings g vectorized?)
  (map (lambda (a)
         (api:make-binding
          (argument-name a)
          (api:make-named-ctype
           (argument-name a)
           (if vectorized?
               (api:make-vectorized-ctype (api:make-base-ctype "scalar_t"))
               (api:make-base-ctype "scalar_t")))
          #f))
       (filter (lambda (a) (type-is-tensor-like? (argument-type a)))
               (arguments-flat-non-out
                (function-schema-arguments
                 (native-function-func (native-functions-group-functional g)))))))

(define (compute-ufunc-cpu-dtype-body g inner-loops parent-ctx)
  ;; inner-loops :: ((key . (ufunc-name . compute-t)) ...).
  (let* ((scalar-loop (assoc-ref inner-loops "CPUScalar"))
         (vec-loop (assoc-ref inner-loops "CPUVector"))
         (scalar-bindings (cpu-tensor-bindings g #f))
         (vec-bindings (and vec-loop (cpu-tensor-bindings g #t)))
         (scalar-bs (filter scalar-ref-binding? parent-ctx))
         (s-body (map (lambda (b)
                        (string-append "auto _s_" (api:binding-name b)
                                       " = " (api:binding-name b) ".to<scalar_t>();"))
                      scalar-bs))
         (s-ctx (map (lambda (b)
                       (api:make-expr
                        (string-append "_s_" (api:binding-name b))
                        (api:make-named-ctype
                         (api:named-ctype-name (api:binding-nctype b))
                         (api:make-base-ctype "scalar_t"))))
                     scalar-bs))
         (v-body (if vec-loop
                     (map (lambda (b)
                            (string-append "auto _v_" (api:binding-name b)
                                           " = at::vec::Vectorized<scalar_t>(_s_"
                                           (api:binding-name b) ");"))
                          scalar-bs)
                     '()))
         (v-ctx (if vec-loop
                    (map (lambda (b)
                           (api:make-expr
                            (string-append "_v_" (api:binding-name b))
                            (api:make-named-ctype
                             (api:named-ctype-name (api:binding-nctype b))
                             (api:make-vectorized-ctype (api:make-base-ctype "scalar_t")))))
                         scalar-bs)
                    '()))
         (body (append s-body v-body))
         (ctx (append s-ctx v-ctx))
         (body-str (string-join body "\n")))
    (if vec-loop
        (string-append
         "\n" body-str "\n"
         "cpu_kernel_vec(iter,\n"
         "  [=](" (string-join (map api:binding-decl scalar-bindings) ", ") ") { return "
         (ufunc-call (car scalar-loop) (append ctx scalar-bindings)
                     (ufunc-arguments g (cdr scalar-loop))) "; },\n"
         "  [=](" (string-join (map api:binding-decl vec-bindings) ", ") ") { return "
         (ufunc-call (car vec-loop) (append ctx vec-bindings)
                     (ufunc-arguments g (cdr vec-loop))) "; }\n"
         ");\n")
        (string-append
         "\n" body-str "\n"
         "cpu_kernel(iter,\n"
         "  [=](" (string-join (map api:binding-decl scalar-bindings) ", ") ") { return "
         (ufunc-call (car scalar-loop) (append ctx scalar-bindings)
                     (ufunc-arguments g (cdr scalar-loop))) "; }\n"
         ");\n"))))

(define (alist-append-new alist key value)
  ;; Ordered alist: append (key . value) only when key is absent, preserving
  ;; first-seen order (mirrors Python dict.setdefault).
  (if (assoc key alist) alist (append alist (list (cons key value)))))

(define (alist-replace-value alist key value)
  ;; Replace the value at key in place (key must already exist), preserving order.
  (map (lambda (e) (if (equal? (car e) key) (cons key value) e)) alist))

(define (build-cpu-ufunc-sigs g)
  ;; compute_ufunc_cpu_kernel reindexing: ((dtype . ((key . (name . compute-t)) ...)) ...).
  ;; dtypes are kept in first-seen order (Bool, then Generic's expansion).
  (let* ((loops (parse-ufunc-inner-loop
                 (native-function-ufunc-inner-loop (native-functions-group-out g)))))
    (let loop-keys ((ks '("CPUScalar" "CPUVector")) (sigs '()))
      (if (null? ks)
          sigs
          (let* ((k (car ks))
                 (compute-t (if (string=? k "CPUScalar")
                                (api:make-base-ctype "scalar_t")
                                (api:make-vectorized-ctype (api:make-base-ctype "scalar_t"))))
                 (lks (append
                       (if (assoc-ref loops k) (list k) '())
                       (if (and (assoc-ref loops "ScalarOnly")
                                (string=? k "CPUScalar"))
                           (list "ScalarOnly") '())
                       (if (assoc-ref loops "Generic") (list "Generic") '()))))
            (loop-keys
             (cdr ks)
             (fold
              (lambda (lk acc)
                (let* ((loop-entry (assoc-ref loops lk))
                       (name (and loop-entry (car loop-entry)))
                       (dtypes (and loop-entry (loop-supported-dtypes loop-entry))))
                  (if loop-entry
                      (fold
                       (lambda (dtype acc2)
                         (let* ((sig (cons k (cons (string-append "ufunc::" name)
                                                   compute-t)))
                                (entry (assoc dtype acc2)))
                           (if entry
                               (let ((inner (cdr entry)))
                                 (if (assoc k inner)
                                     acc2
                                     (alist-replace-value acc2 dtype (cons sig inner))))
                               (alist-append-new acc2 dtype (list sig)))))
                       acc dtypes)
                      acc)))
              sigs lks)))))))

(define (compute-ufunc-cpu-kernel g)
  (let* ((ufunc-sigs (build-cpu-ufunc-sigs g))
         (parent-ctx (stub-arguments g))
         (dtype-cases (map (lambda (entry)
                             (string-append
                              "\nAT_DISPATCH_CASE(at::ScalarType::" (car entry) ",\n"
                              "  [&]() {\n"
                              "    " (compute-ufunc-cpu-dtype-body
                                       g (cdr entry) parent-ctx) "\n"
                              "  }\n"
                              ")\n"))
                           ufunc-sigs))
         (dtype-cases-str (string-join dtype-cases "\n")))
    (string-append
     "\nnamespace {\n\n"
     (stub-kernel-defn g) " {\n"
     "  AT_DISPATCH_SWITCH(iter.common_dtype(), \"" (stub-name g) "\",\n"
     "    " dtype-cases-str "\n"
     "  );\n"
     "}\n\n"
     "} // anonymous namespace\n\n"
     (stub-type-defn g) ";\n"
     (stub-dispatch-decl g) "\n"
     "REGISTER_DISPATCH(" (stub-name g) ", &" (stub-kernel-name g) ")\n")))

;; ---------------------------------------------------------------------------
;; templates (aten/src/ATen/templates/*, frozen verbatim).
;; ---------------------------------------------------------------------------

(define %ufunc-cpu-template
  (string-append
   "#define TORCH_ASSERT_NO_OPERATORS\n"
   "\n"
   "#include <ATen/native/DispatchStub.h>\n"
   "#include <ATen/TensorIterator.h>\n"
   "#include <ATen/TensorMeta.h>\n"
   "\n"
   "namespace at {\n"
   "\n"
   "// NB: this is explicitly copied here (via codegen) rather than\n"
   "// included via NativeFunctions.h to avoid recompiling this file when\n"
   "// NativeFunctions.h changes\n"
   "namespace meta {\n"
   "${meta_declaration}\n"
   "}\n"
   "\n"
   "namespace native {\n"
   "${native_declaration}\n"
   "${native_definitions}\n"
   "}} // namespace at::native\n"))

(define %ufunc-cpu-kernel-template
  (string-append
   "#define TORCH_ASSERT_NO_OPERATORS\n"
   "\n"
   "#include <ATen/native/ufunc/${name}.h>\n"
   "#include <ATen/native/DispatchStub.h>\n"
   "#include <ATen/TensorIterator.h>\n"
   "#include <ATen/native/cpu/Loops.h>\n"
   "#include <ATen/cpu/vec/vec.h>\n"
   "#include <ATen/Dispatch.h>\n"
   "#include <c10/core/Scalar.h>\n"
   "\n"
   "namespace at {\n"
   "namespace native {\n"
   "${native_definitions}\n"
   "}} // namespace at::native\n"))

(define %ufunc-cuda-template
  (string-append
   "#define TORCH_ASSERT_NO_OPERATORS\n"
   "\n"
   "#include <ATen/native/ufunc/${name}.h>\n"
   "#include <ATen/Dispatch.h>\n"
   "#include <ATen/native/DispatchStub.h>\n"
   "#include <c10/core/Scalar.h>\n"
   "${cuda_headers}\n"
   "\n"
   "namespace at {\n"
   "\n"
   "// NB: this is explicitly copied here (via codegen) rather than\n"
   "// included via NativeFunctions.h to avoid recompiling this file when\n"
   "// NativeFunctions.h changes\n"
   "namespace meta {\n"
   "${meta_declaration}\n"
   "}\n"
   "\n"
   "namespace native {\n"
   "${native_declaration}\n"
   "${native_definitions}\n"
   "}} // namespace at::native\n"))

;; ---------------------------------------------------------------------------
;; render entry points.
;; ---------------------------------------------------------------------------

(define (render-ufunc-cpu-cpp g cpu-bi)
  (code-template-substitute
   %ufunc-cpu-template
   (lambda (key)
     (cond
      ((string=? key "meta_declaration")
       (compute-meta-function-declaration g))
      ((string=? key "native_declaration")
       (compute-native-function-declaration g cpu-bi))
      ((string=? key "native_definitions")
       (compute-ufunc-cpu g))
      (else (error 'unknown-ufunc-template-key key))))))

(define (render-ufunc-cpu-kernel-cpp g)
  (code-template-substitute
   %ufunc-cpu-kernel-template
   (lambda (key)
     (cond
      ((string=? key "name") (ufunc-base-name g))
      ((string=? key "native_definitions") (compute-ufunc-cpu-kernel g))
      (else (error 'unknown-ufunc-template-key key))))))

(define (render-ufunc-cuda-cu g cuda-bi)
  (code-template-substitute
   %ufunc-cuda-template
   (lambda (key)
     (cond
      ((string=? key "name") (ufunc-base-name g))
      ((string=? key "cuda_headers") "#include <ATen/native/cuda/Loops.cuh>")
      ((string=? key "meta_declaration") (compute-meta-function-declaration g))
      ((string=? key "native_declaration")
       (compute-native-function-declaration g cuda-bi))
      ((string=? key "native_definitions") (compute-ufunc-cuda g))
      (else (error 'unknown-ufunc-template-key key))))))
