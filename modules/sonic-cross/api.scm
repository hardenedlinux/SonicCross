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

;; The typed API translation layer, mirroring torchgen/api/*.py at the frozen
;; commit.  It models the C++ type system (CType / NamedCType / Binding / Expr),
;; the cpp / native / structured / dispatcher type translations, the signatures
;; (CppSignature / DispatcherSignature / NativeSignature / kernel_signature) and
;; the translate synthesis engine.  Unlike dispatcher.scm (a string-level G3
;; shorthand), this layer carries the full NamedCType structure required by the
;; G2 Register{DispatchKey}.cpp emitters.

(define-module (sonic-cross api)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-13)
  #:use-module (sonic-cross core-ir)
  #:use-module (sonic-cross native-function)
  #:use-module (sonic-cross schema-parser)
  #:use-module (sonic-cross yaml)
  #:export (make-base-ctype
            base-ctype?
            base-ctype-string
            make-const-ref-ctype
            const-ref-ctype?
            const-ref-ctype-elem
            make-mut-ref-ctype
            mut-ref-ctype?
            mut-ref-ctype-elem
            make-optional-ctype
            optional-ctype?
            optional-ctype-elem
            make-vector-ctype
            vector-ctype?
            vector-ctype-elem
            make-vectorized-ctype
            vectorized-ctype?
            vectorized-ctype-elem
            make-array-ctype
            array-ctype?
            array-ctype-elem
            array-ctype-size
            make-tuple-ctype
            tuple-ctype?
            tuple-ctype-elems
            make-array-ref-ctype
            array-ref-ctype?
            array-ref-ctype-elem
            make-list-ctype
            list-ctype?
            list-ctype-elem
            ctype-cpp-type
            make-named-ctype
            named-ctype?
            named-ctype-name
            named-ctype-type
            named-ctype-cpp-type
            named-ctype-remove-const-ref
            ctype-remove-const-ref
            make-binding
            binding?
            binding-name
            binding-nctype
            binding-default
            binding-type
            binding-no-default
            binding-decl
            binding-defn
            binding-with-name
            make-expr
            expr?
            expr-expr
            expr-type
            ;; base type spellings (BaseCppType constants)
            %void-t %bool-t %long-t %double-t %float-t %string-t %generator-t
            %scalar-type-t %tensor-t %optional-tensor-ref-t %tensor-list-t
            %i-tensor-list-ref-t %i-opt-tensor-list-ref-t %dim-vector-t
            %layout-t %device-t %device-index-t %scalar-t %optional-scalar-ref-t
            %memory-format-t %qscheme-t %storage-t %stream-t %int-array-ref-t
            %optional-int-array-ref-t %optional-sym-int-array-ref-t
            %tensor-options-t %sym-int-t %sym-bool-t %sym-int-array-ref-t
            %opmath-t %scalar-t
            ;; cpp translation
            cpp-name
            cpp-valuetype-type
            cpp-argumenttype-type
            cpp-argument-type
            cpp-returntype-type
            cpp-return-type
            cpp-returns-type
            cpp-default-expr
            cpp-argument
            cpp-arguments
            cpp-return-names
            ;; native translation
            native-name
            native-argumenttype-type
            native-argument-type
            native-argument
            native-arguments
            native-returns-type
            ;; structured translation
            structured-argumenttype-type
            structured-argument-type
            structured-argument
            structured-impl-arguments
            structured-meta-arguments
            structured-out-arguments
            ;; dispatcher translation
            dispatcher-name
            dispatcher-argument
            dispatcher-arguments
            dispatcher-returns-type
            dispatcher-jit-arguments
            ;; signatures
            cpp-signature
            cpp-signature-arguments
            cpp-signature-name
            cpp-signature-decl
            cpp-signature-defn
            cpp-signature-returns-type
            cpp-signature-group-signatures
            dispatcher-signature-name
            dispatcher-signature-decl
            dispatcher-signature-defn
            dispatcher-signature-exprs
            dispatcher-signature-arguments
            dispatcher-signature-returns-type
            native-signature-name
            native-signature-decl
            native-signature-defn
            native-signature-arguments
            native-signature-returns-type
            kernel-signature
            backend-index-get-kernel
            backend-index-get-kernel-group
            ;; translate
            translate
            ;; precomputed
            parse-precomputed
            ;; arguments reconstruction
            arguments-positional
            arguments-kwarg-only
            arguments-non-out
            ;; context helpers
            native-function-use-const-ref-context
            native-function-use-ilistref-context
            group-use-const-ref-context
            group-use-ilistref-context))

;; Local helpers that core-ir does not export directly.

;; Mirrors torchgen str(BaseOperatorName): the full base spelling carrying the
;; "_"/"_functional" suffix and dunder-method wrapping.  Kept private here
;; (native-function.scm has an identical definition it does not export).
(define (base-operator-spelling name)
  (let* ((base (operator-name-base name))
         (base-name (base-operator-name-base base)))
    (if (base-operator-name-dunder-method? base)
        (string-append "__"
                       (if (base-operator-name-inplace? base)
                           (string-append "i" base-name)
                           base-name)
                       "__")
        (if (base-operator-name-inplace? base)
            (string-append base-name "_")
            (if (base-operator-name-functional-overload? base)
                (string-append base-name "_functional")
                base-name)))))

;; Element accessor for optional types: (optional elem) carries elem as the
;; single type-argument, exactly like list-type-element.
(define (optional-element value)
  (car (type-arguments value)))

;; torchgen Return.is_write == annotation is present and has the write flag.
(define (return-is-write? value)
  (let ((a (return-annotation value)))
    (and a (annotation-is-write? a))))

;; ---------------------------------------------------------------------------
;; CType model.  A CType is a tagged list; structural equality via equal? is
;; what translate's `==` comparisons need.
;; ---------------------------------------------------------------------------

(define (make-base-ctype s) (list 'base s))
(define (base-ctype? x) (and (pair? x) (eq? (car x) 'base)))
(define (base-ctype-string x) (cadr x))

(define (make-const-ref-ctype e) (list 'const-ref e))
(define (const-ref-ctype? x) (and (pair? x) (eq? (car x) 'const-ref)))
(define (const-ref-ctype-elem x) (cadr x))

(define (make-mut-ref-ctype e) (list 'mut-ref e))
(define (mut-ref-ctype? x) (and (pair? x) (eq? (car x) 'mut-ref)))
(define (mut-ref-ctype-elem x) (cadr x))

(define (make-optional-ctype e) (list 'optional e))
(define (optional-ctype? x) (and (pair? x) (eq? (car x) 'optional)))
(define (optional-ctype-elem x) (cadr x))

(define (make-vector-ctype e) (list 'vector e))
(define (vector-ctype? x) (and (pair? x) (eq? (car x) 'vector)))
(define (vector-ctype-elem x) (cadr x))

;; VectorizedCType (torchgen api/types/types.py): an explicitly-specialized
;; `at::vec::Vectorized<T>` template, distinct from VectorCType (std::vector).
(define (make-vectorized-ctype e) (list 'vectorized e))
(define (vectorized-ctype? x) (and (pair? x) (eq? (car x) 'vectorized)))
(define (vectorized-ctype-elem x) (cadr x))

(define (make-array-ctype e size) (list 'array e size))
(define (array-ctype? x) (and (pair? x) (eq? (car x) 'array)))
(define (array-ctype-elem x) (cadr x))
(define (array-ctype-size x) (caddr x))

(define (make-tuple-ctype elems) (cons 'tuple elems))
(define (tuple-ctype? x) (and (pair? x) (eq? (car x) 'tuple)))
(define (tuple-ctype-elems x) (cdr x))

(define (make-array-ref-ctype e) (list 'array-ref e))
(define (array-ref-ctype? x) (and (pair? x) (eq? (car x) 'array-ref)))
(define (array-ref-ctype-elem x) (cadr x))

(define (make-list-ctype e) (list 'list-of e))
(define (list-ctype? x) (and (pair? x) (eq? (car x) 'list-of)))
(define (list-ctype-elem x) (cadr x))

(define (ctype-cpp-type x . rest)
  ;; Mirror CType.cpp_type(strip_ref=False).  strip_ref only recurses for
  ;; const-ref/mut-ref; every templated wrapper ignores it.
  (let ((strip-ref (and (pair? rest) (car rest))))
    (cond
     ((base-ctype? x) (base-ctype-string x))
     ((const-ref-ctype? x)
      (if strip-ref
          (ctype-cpp-type (const-ref-ctype-elem x) strip-ref)
          (string-append "const "
                         (ctype-cpp-type (const-ref-ctype-elem x)) " &")))
     ((mut-ref-ctype? x)
      (if strip-ref
          (ctype-cpp-type (mut-ref-ctype-elem x) strip-ref)
          (string-append (ctype-cpp-type (mut-ref-ctype-elem x)) " &")))
     ((optional-ctype? x)
      (string-append "::std::optional<"
                     (ctype-cpp-type (optional-ctype-elem x)) ">"))
     ((vector-ctype? x)
      (string-append "::std::vector<"
                     (ctype-cpp-type (vector-ctype-elem x)) ">"))
     ((vectorized-ctype? x)
      (string-append "at::vec::Vectorized<"
                     (ctype-cpp-type (vectorized-ctype-elem x)) ">"))
     ((array-ctype? x)
      (string-append "::std::array<"
                     (ctype-cpp-type (array-ctype-elem x))
                     "," (number->string (array-ctype-size x)) ">"))
     ((tuple-ctype? x)
      (string-append "::std::tuple<"
                     (string-join (map ctype-cpp-type (tuple-ctype-elems x)) ",")
                     ">"))
     ((array-ref-ctype? x)
      (string-append "at::ArrayRef<"
                     (ctype-cpp-type (array-ref-ctype-elem x)) ">"))
     ((list-ctype? x)
      (string-append "c10::List<"
                     (ctype-cpp-type (list-ctype-elem x)) ">"))
     (else (error 'unknown-ctype x)))))

;; ---------------------------------------------------------------------------
;; BaseCppType spellings.
;; ---------------------------------------------------------------------------

(define %void-t "void")
(define %bool-t "bool")
(define %long-t "int64_t")
(define %double-t "double")
(define %float-t "float")
(define %string-t "c10::string_view")
(define %generator-t "at::Generator")
(define %scalar-type-t "at::ScalarType")
(define %tensor-t "at::Tensor")
(define %optional-tensor-ref-t "at::OptionalTensorRef")
(define %tensor-list-t "at::TensorList")
(define %i-tensor-list-ref-t "at::ITensorListRef")
(define %i-opt-tensor-list-ref-t "at::IOptTensorListRef")
(define %dim-vector-t "at::DimVector")
(define %layout-t "at::Layout")
(define %device-t "at::Device")
(define %device-index-t "at::DeviceIndex")
(define %scalar-t "at::Scalar")
(define %optional-scalar-ref-t "at::OptionalScalarRef")
(define %memory-format-t "at::MemoryFormat")
(define %qscheme-t "at::QScheme")
(define %storage-t "at::Storage")
(define %stream-t "at::Stream")
(define %int-array-ref-t "at::IntArrayRef")
(define %optional-int-array-ref-t "at::OptionalIntArrayRef")
(define %optional-sym-int-array-ref-t "at::OptionalSymIntArrayRef")
(define %tensor-options-t "at::TensorOptions")
(define %sym-int-t "c10::SymInt")
(define %sym-bool-t "c10::SymBool")
(define %sym-int-array-ref-t "c10::SymIntArrayRef")
(define %opmath-t "opmath_t")

(define (base-type-cpp-string name)
  ;; torchgen api/types.py BaseTypeToCppMapping for the base-type kinds that are
  ;; distinct core-ir records (Tensor/Scalar/int/float/bool/str/SymInt/
  ;; MemoryFormat are handled directly in valuetype-type).
  (cond
   ((string=? name "Generator") %generator-t)
   ((string=? name "ScalarType") %scalar-type-t)
   ((string=? name "DimVector") %dim-vector-t)
   ((string=? name "Layout") %layout-t)
   ((string=? name "Device") %device-t)
   ((string=? name "DeviceIndex") %device-index-t)
   ((string=? name "QScheme") %qscheme-t)
   ((string=? name "Storage") %storage-t)
   ((string=? name "Stream") %stream-t)
   ((string=? name "SymBool") %sym-bool-t)
   (else (error 'unknown-base-type name))))

;; ---------------------------------------------------------------------------
;; NamedCType / Binding / Expr.
;; ---------------------------------------------------------------------------

(define (make-named-ctype name type) (list 'named name type))
(define (named-ctype? x) (and (pair? x) (eq? (car x) 'named)))
(define (named-ctype-name x) (cadr x))
(define (named-ctype-type x) (caddr x))
(define (named-ctype-cpp-type x) (ctype-cpp-type (named-ctype-type x)))

;; CType.remove_const_ref() / NamedCType.remove_const_ref() (torchgen
;; api/types/types_base.py): strips ConstRef/MutRef wrappers, recursing into
;; the template wrappers that carry an inner CType.
(define (ctype-remove-const-ref x)
  (cond
   ((base-ctype? x) x)
   ((const-ref-ctype? x) (ctype-remove-const-ref (const-ref-ctype-elem x)))
   ((mut-ref-ctype? x) (ctype-remove-const-ref (mut-ref-ctype-elem x)))
   ((optional-ctype? x)
    (make-optional-ctype (ctype-remove-const-ref (optional-ctype-elem x))))
   ((vector-ctype? x)
    (make-vector-ctype (ctype-remove-const-ref (vector-ctype-elem x))))
   ((vectorized-ctype? x) x)
   ((array-ctype? x)
    (make-array-ctype (ctype-remove-const-ref (array-ctype-elem x))
                      (array-ctype-size x)))
   ((tuple-ctype? x)
    (make-tuple-ctype (map ctype-remove-const-ref (tuple-ctype-elems x))))
   ((array-ref-ctype? x)
    (make-array-ref-ctype (ctype-remove-const-ref (array-ref-ctype-elem x))))
   ((list-ctype? x)
    (make-list-ctype (ctype-remove-const-ref (list-ctype-elem x))))
   (else (error 'unknown-ctype-remove-const-ref x))))

(define (named-ctype-remove-const-ref x)
  (make-named-ctype (named-ctype-name x)
                    (ctype-remove-const-ref (named-ctype-type x))))

(define (make-binding name nctype default)
  (list 'binding name nctype default))
(define (binding? x) (and (pair? x) (eq? (car x) 'binding)))
(define (binding-name x) (cadr x))
(define (binding-nctype x) (caddr x))
(define (binding-default x) (cadddr x))
(define (binding-type x) (named-ctype-cpp-type (binding-nctype x)))
(define (binding-no-default x)
  (make-binding (binding-name x) (binding-nctype x) #f))
(define (binding-decl x)
  (string-append (binding-type x) " " (binding-name x)
                 (if (binding-default x)
                     (string-append "=" (binding-default x))
                     "")))
(define (binding-defn x)
  (string-append (binding-type x) " " (binding-name x)))

;; Binding.with_name(name): a binding with a new binding-site name but the same
;; semantic NamedCType (torchgen keeps nctype -- and thus the semantic name --
;; unchanged).
(define (binding-with-name x name)
  (make-binding name (binding-nctype x) (binding-default x)))

(define (make-expr expr type) (list 'expr expr type))
(define (expr? x) (and (pair? x) (eq? (car x) 'expr)))
(define (expr-expr x) (cadr x))
(define (expr-type x) (caddr x))

;; ---------------------------------------------------------------------------
;; Arguments reconstruction: torchgen Arguments.positional / kwarg_only /
;; non_out preserve the SelfArgument and TensorOptionsArguments wrappers.
;; ---------------------------------------------------------------------------

(define (arguments-positional args)
  (append (arguments-pre-self-positional args)
          (if (arguments-self-arg args)
              (list (arguments-self-arg args))
              '())
          (arguments-post-self-positional args)))

(define (arguments-kwarg-only args)
  (append (arguments-pre-tensor-options-kwarg-only args)
          (if (arguments-tensor-options args)
              (list (arguments-tensor-options args))
              '())
          (arguments-post-tensor-options-kwarg-only args)))

(define (arguments-non-out args)
  (append (arguments-positional args) (arguments-kwarg-only args)))

;; ---------------------------------------------------------------------------
;; Dynamic-scope flags (torchgen.local).  In G2 the flags are fixed per
;; native-function context: use_const_ref_for_mutable_tensors comes from the
;; function record, and use_ilistref_for_tensor_lists from part_of_structured_group.
;; ---------------------------------------------------------------------------

(define (native-function-use-const-ref-context f)
  (native-function-use-const-ref-for-mutable-tensors? f))

(define (native-function-use-ilistref-context f)
  (native-function-part-of-structured-group? f))

(define (group-use-const-ref-context g)
  (native-function-use-const-ref-for-mutable-tensors?
   (native-functions-group-out g)))

(define (group-use-ilistref-context g)
  (native-function-part-of-structured-group? (native-functions-group-out g)))

;; ---------------------------------------------------------------------------
;; cpp translation (torchgen api/cpp.py).
;; ---------------------------------------------------------------------------

(define* (cpp-name func #:key (faithful-name-for-out-overloads #f)
                   (symint-overload #f))
  (string-append
   (base-operator-spelling (function-schema-name func))
   (if symint-overload "_symint" "")
   (if (function-schema-is-out-fn? func)
       (if faithful-name-for-out-overloads "_outf" "_out")
       "")))

(define (cpp-valuetype-type type binds mutable symint)
  ;; Returns a NamedCType or #f.  `binds` is a name (string or the special
  ;; symbol 'possibly-redundant-memory-format).
  (case (type-kind type)
    ((tensor scalar) #f)
    ((int) (make-named-ctype binds (make-base-ctype %long-t)))
    ((float) (make-named-ctype binds (make-base-ctype %double-t)))
    ((bool) (make-named-ctype binds (make-base-ctype %bool-t)))
    ((string) (make-named-ctype binds (make-base-ctype %string-t)))
    ((sym-int) (make-named-ctype binds
                                 (make-base-ctype (if symint %sym-int-t %long-t))))
    ((memory-format) (make-named-ctype binds (make-base-ctype %memory-format-t)))
    ((base-type) (make-named-ctype binds
                                   (make-base-ctype
                                    (base-type-cpp-string (base-type-name type)))))
    ((optional)
     (let ((elem (cpp-valuetype-type (optional-element type) binds mutable symint)))
       (and elem
            (make-named-ctype binds
                              (make-optional-ctype (named-ctype-type elem))))))
    ((list)
     (if (string=? (type->string (list-type-element type)) "bool")
         (let ((size (list-type-size type)))
           (unless size (error 'bool-list-type-must-have-size type))
           (make-named-ctype binds
                             (make-array-ctype (make-base-ctype %bool-t) size)))
         #f))
    (else (error 'unrecognized-valuetype type))))

(define (cpp-argumenttype-type type mutable binds remove-non-owning-ref-types
                               symint use-const-ref use-ilistref)
  (or (cpp-valuetype-type type binds mutable symint)
      (case (type-kind type)
        ((tensor)
         (cond
          ((and mutable (not use-const-ref))
           (make-named-ctype binds (make-mut-ref-ctype (make-base-ctype %tensor-t))))
          (remove-non-owning-ref-types
           (make-named-ctype binds (make-base-ctype %tensor-t)))
          (else
           (make-named-ctype binds (make-const-ref-ctype (make-base-ctype %tensor-t))))))
        ((scalar)
         (if remove-non-owning-ref-types
             (make-named-ctype binds (make-base-ctype %scalar-t))
             (make-named-ctype binds (make-const-ref-ctype (make-base-ctype %scalar-t)))))
        ((optional)
         (let ((elem (optional-element type))
               (elem-str (type->string (optional-element type))))
           (cond
            ((string=? elem-str "Tensor")
             (cond
              ((and mutable (not use-const-ref))
               (make-named-ctype binds (make-mut-ref-ctype (make-base-ctype %tensor-t))))
              (remove-non-owning-ref-types
               (make-named-ctype binds (make-optional-ctype (make-base-ctype %tensor-t))))
              (else
               (make-named-ctype binds
                                 (make-const-ref-ctype
                                  (make-optional-ctype (make-base-ctype %tensor-t)))))))
            ((string=? elem-str "Scalar")
             (if remove-non-owning-ref-types
                 (make-named-ctype binds (make-optional-ctype (make-base-ctype %scalar-t)))
                 (make-named-ctype binds
                                   (make-const-ref-ctype
                                    (make-optional-ctype (make-base-ctype %scalar-t))))))
            ((and (eq? (type-kind elem) 'list)
                  (string=? (type->string (list-type-element elem)) "int"))
             (make-named-ctype binds (make-base-ctype %optional-int-array-ref-t)))
            ((and (eq? (type-kind elem) 'list)
                  (string=? (type->string (list-type-element elem)) "SymInt"))
             (make-named-ctype binds
                               (make-base-ctype
                                (if symint %optional-sym-int-array-ref-t
                                    %optional-int-array-ref-t))))
            (else
             (let ((elem-nct (cpp-argumenttype-type
                              elem mutable binds remove-non-owning-ref-types
                              symint use-const-ref use-ilistref)))
               (make-named-ctype binds
                                 (make-optional-ctype (named-ctype-type elem-nct))))))))
        ((list)
         (let ((elem-str (type->string (list-type-element type))))
           (cond
            ((string=? elem-str "int")
             (if remove-non-owning-ref-types
                 (make-named-ctype binds (make-vector-ctype (make-base-ctype %long-t)))
                 (make-named-ctype binds (make-base-ctype %int-array-ref-t))))
            ((string=? elem-str "SymInt")
             (if remove-non-owning-ref-types
                 (make-named-ctype binds
                                   (make-vector-ctype
                                    (make-base-ctype (if symint %sym-int-t %long-t))))
                 (make-named-ctype binds
                                   (make-base-ctype
                                    (if symint %sym-int-array-ref-t %int-array-ref-t)))))
            ((string=? elem-str "Tensor")
             (if use-ilistref
                 (make-named-ctype binds
                                   (make-const-ref-ctype
                                    (make-base-ctype %i-tensor-list-ref-t)))
                 (make-named-ctype binds (make-base-ctype %tensor-list-t))))
            ((string=? elem-str "Scalar")
             (make-named-ctype binds
                               (make-array-ref-ctype (make-base-ctype %scalar-t))))
            ((string=? elem-str "Tensor?")
             (make-named-ctype binds
                               (make-const-ref-ctype
                                (make-list-ctype
                                 (make-optional-ctype (make-base-ctype %tensor-t))))))
            (else
             (let ((elem-nct (cpp-argumenttype-type
                              (list-type-element type) mutable binds
                              remove-non-owning-ref-types symint
                              use-const-ref use-ilistref)))
               (make-named-ctype binds
                                 (make-array-ref-ctype (named-ctype-type elem-nct))))))))
        (else (error 'unrecognized-argumenttype type)))))

(define (cpp-argument-type argument binds symint use-const-ref use-ilistref)
  (cpp-argumenttype-type (argument-type argument)
                         (argument-is-write? argument) binds #f symint
                         use-const-ref use-ilistref))

(define (cpp-returntype-type type mutable symint use-const-ref)
  ;; symint is always respected for return types.
  (let ((vt (cpp-valuetype-type type "__placeholder__" mutable #t)))
    (or (and vt (named-ctype-type vt))
        (case (type-kind type)
        ((tensor)
         (if mutable
             (if use-const-ref
                 (make-const-ref-ctype (make-base-ctype %tensor-t))
                 (make-mut-ref-ctype (make-base-ctype %tensor-t)))
             (make-base-ctype %tensor-t)))
        ((scalar) (make-base-ctype %scalar-t))
        ((list)
         (when mutable (error 'mutable-tensor-list-return type))
         (when (list-type-size type) (error 'fixed-size-list-return type))
         (make-vector-ctype
          (cpp-returntype-type (list-type-element type) #f symint use-const-ref)))
        ((optional)
         (let ((elem (optional-element type)))
           (if (string=? (type->string elem) "Tensor")
               (make-optional-ctype
                (cpp-returntype-type elem mutable symint use-const-ref))
               (error 'unrecognized-return-type type))))
        (else (error 'unrecognized-return-type type))))))

(define (cpp-return-type return symint use-const-ref)
  (cpp-returntype-type (return-type return)
                       (return-is-write? return) symint use-const-ref))

(define (cpp-returns-type returns symint use-const-ref)
  (cond
   ((null? returns) (make-base-ctype %void-t))
   ((null? (cdr returns)) (cpp-return-type (car returns) symint use-const-ref))
   (else
    (make-tuple-ctype
     (map (lambda (r) (cpp-return-type r symint use-const-ref)) returns)))))

(define (jit-to-cpp-default d)
  (cond
   ((string=? d "False") "false")
   ((string=? d "True") "true")
   ((string=? d "None") "::std::nullopt")
   ((string=? d "Mean") "at::Reduction::Mean")
   ((string=? d "[]") "{}")
   ((string=? d "contiguous_format") "c10::MemoryFormat::Contiguous")
   ((string=? d "long") "at::kLong")
   (else d)))

(define (string-default-expr d)
  ;; Convert a single-quoted schema default to a double-quoted C++ literal.
  (let loop ((i 1) (out ""))
    (if (>= (1+ i) (string-length d))
        out
        (let ((c (string-ref d i)))
          (if (not (char=? c #\\))
              (loop (1+ i)
                    (string-append out
                                   (if (char=? c #\") "\\\"" (string c))))
              (if (char=? (string-ref d (1+ i)) #\')
                  (loop (+ i 2) (string-append out "'"))
                  (loop (+ i 2)
                        (string-append out (substring d i (+ i 2))))))))))

(define (cpp-default-expr d type symint)
  (cond
   ((and (string=? d "None") (string=? (type->string type) "Tensor?")) "{}")
   ((and (eq? (type-kind type) 'string)
         (>= (string-length d) 2)
         (char=? (string-ref d 0) #\')
         (char=? (string-ref d (1- (string-length d))) #\'))
    (string-append "\"" (string-default-expr d) "\""))
   ((eq? (type-kind type) 'optional)
    (if (string=? d "None")
        "::std::nullopt"
        (cpp-default-expr d (optional-element type) symint)))
   ((eq? (type-kind type) 'list)
    (cond
     ((and (string-prefix? "[" d) (string-suffix? "]" d))
      (string-append "{" (substring d 1 (1- (string-length d))) "}"))
     ((and symint (string-every char-set:digit d)
           (string=? (type->string (list-type-element type)) "SymInt"))
      (string-append "c10::SymInt(" d ")"))
     ((not (list-type-size type))
      (error 'expected-list-default d))
     (else (jit-to-cpp-default d))))
   (else (jit-to-cpp-default d))))

(define (cpp-argument argument cpp-no-default-args method faithful symint
                      has-tensor-options use-const-ref use-ilistref)
  (cond
   ((argument? argument)
    (let* ((binds (if (and (string=? (argument-name argument) "memory_format")
                           has-tensor-options)
                      'possibly-redundant-memory-format
                      (argument-name argument)))
           (default
            (and (not (member (argument-name argument) cpp-no-default-args))
                 (argument-default argument)
                 (cpp-default-expr (argument-default argument)
                                   (argument-type argument) symint))))
      (list (make-binding (argument-name argument)
                          (cpp-argument-type argument binds symint
                                             use-const-ref use-ilistref)
                          default))))
   ((tensor-options-arguments? argument)
    (if faithful
        (append (cpp-argument (tensor-options-arguments-dtype argument)
                              cpp-no-default-args method faithful symint
                              has-tensor-options use-const-ref use-ilistref)
                (cpp-argument (tensor-options-arguments-layout argument)
                              cpp-no-default-args method faithful symint
                              has-tensor-options use-const-ref use-ilistref)
                (cpp-argument (tensor-options-arguments-device argument)
                              cpp-no-default-args method faithful symint
                              has-tensor-options use-const-ref use-ilistref)
                (cpp-argument (tensor-options-arguments-pin-memory argument)
                              cpp-no-default-args method faithful symint
                              has-tensor-options use-const-ref use-ilistref))
        (let ((default
               (cond
                ((every (lambda (x)
                          (and (argument-default x)
                               (string=? (argument-default x) "None")))
                        (tensor-options-arguments-all argument))
                 "{}")
                ((and (argument-default (tensor-options-arguments-dtype argument))
                      (string=? (argument-default (tensor-options-arguments-dtype argument))
                                "long"))
                 "at::kLong")
                (else #f))))
          (list (make-binding
                 "options"
                 (make-named-ctype "options" (make-base-ctype %tensor-options-t))
                 default)))))
   ((self-argument? argument)
    (if method '() (cpp-argument (self-argument-argument argument)
                                 cpp-no-default-args method faithful symint
                                 has-tensor-options use-const-ref use-ilistref)))
   (else (error 'cpp-argument-unknown argument))))

(define (cpp-arguments arguments faithful symint method cpp-no-default-args
                       use-const-ref use-ilistref)
  (let* ((args (if faithful
                   (append (arguments-non-out arguments) (arguments-out arguments))
                   (append (arguments-out arguments) (arguments-non-out arguments))))
         (has-tensor-options (and (arguments-tensor-options arguments) #t))
         (bindings
          (append-map
           (lambda (a)
             (cpp-argument a cpp-no-default-args method faithful symint
                           has-tensor-options use-const-ref use-ilistref))
           args)))
    (if faithful (map binding-no-default bindings) bindings)))

(define (cpp-return-names native-function fallback-name)
  (let* ((func (native-function-func native-function))
         (returns (function-schema-returns func))
         (name (function-schema-name func))
         (inplace? (base-operator-name-inplace? (operator-name-base name))))
    (let loop ((remaining returns) (i 0) (result '()))
      (if (null? remaining)
          (reverse result)
          (let* ((r (car remaining))
                 (name
                  (cond
                   ((and inplace? (> i 0)) (error 'illegal-inplace-multi-return))
                   (inplace? "self")
                   ((function-schema-is-out-fn? func)
                    (argument-name (list-ref (arguments-out
                                              (function-schema-arguments func)) i)))
                   ((return-name r)
                    (let ((conflict?
                           (any (lambda (a)
                                  (string=? (return-name r) (argument-name a)))
                                (arguments-all (function-schema-arguments func)))))
                      (if (and conflict? (not (function-schema-is-out-fn? func)))
                          (string-append (return-name r) "_return")
                          (return-name r))))
                   ((= (length returns) 1) fallback-name)
                   (else (string-append fallback-name (number->string i))))))
            (loop (cdr remaining) (1+ i) (cons name result)))))))

;; ---------------------------------------------------------------------------
;; native translation (torchgen api/native.py).
;; ---------------------------------------------------------------------------

(define (native-name func)
  (let* ((name (function-schema-name func))
         (spelling (base-operator-spelling name))
         (overload (operator-name-overload-name name)))
    (string-append spelling
                   (if (function-schema-is-out-fn? func) "_out" "")
                   (if (string-null? overload) "" (string-append "_" overload)))))

(define (native-argumenttype-type type mutable binds symint
                                  use-const-ref use-ilistref)
  (let ((str (type->string type)))
    (cond
     ((string=? str "Tensor?")
      (let ((tensor-type (make-optional-ctype (make-base-ctype %tensor-t))))
        (if (and mutable (not use-const-ref))
            (make-named-ctype binds (make-mut-ref-ctype tensor-type))
            (make-named-ctype binds (make-const-ref-ctype tensor-type)))))
     ((string=? str "Tensor?[]")
      (make-named-ctype binds
                        (make-const-ref-ctype
                         (make-list-ctype
                          (make-optional-ctype (make-base-ctype %tensor-t))))))
     ((string=? str "Scalar")
      (make-named-ctype binds (make-const-ref-ctype (make-base-ctype %scalar-t))))
     ((string=? str "Scalar?")
      (make-named-ctype binds
                        (make-const-ref-ctype
                         (make-optional-ctype (make-base-ctype %scalar-t)))))
     (else
      (cpp-argumenttype-type type mutable binds #f symint
                             use-const-ref use-ilistref)))))

(define (native-argument-type argument binds symint use-const-ref use-ilistref)
  (native-argumenttype-type (argument-type argument)
                            (argument-is-write? argument) binds symint
                            use-const-ref use-ilistref))

(define (native-argument argument is-out symint use-const-ref use-ilistref)
  (define (should-default) (not is-out))
  (cond
   ((argument? argument)
    (let ((default (and (should-default)
                        (argument-default argument)
                        (cpp-default-expr (argument-default argument)
                                          (argument-type argument) symint))))
      (list (make-binding (argument-name argument)
                          (native-argument-type argument (argument-name argument)
                                                symint use-const-ref use-ilistref)
                          default))))
   ((self-argument? argument)
    (native-argument (self-argument-argument argument) is-out symint
                     use-const-ref use-ilistref))
   ((tensor-options-arguments? argument)
    (let ((default (and (should-default) "{}")))
      (list
       (make-binding "dtype"
                     (make-named-ctype "dtype"
                                       (make-optional-ctype (make-base-ctype %scalar-type-t)))
                     default)
       (make-binding "layout"
                     (make-named-ctype "layout"
                                       (make-optional-ctype (make-base-ctype %layout-t)))
                     default)
       (make-binding "device"
                     (make-named-ctype "device"
                                       (make-optional-ctype (make-base-ctype %device-t)))
                     default)
       (make-binding "pin_memory"
                     (make-named-ctype "pin_memory"
                                       (make-optional-ctype (make-base-ctype %bool-t)))
                     default))))
   (else (error 'native-argument-unknown argument))))

(define (native-arguments func symint use-const-ref use-ilistref)
  (let* ((arguments (function-schema-arguments func))
         (args (append (arguments-non-out arguments) (arguments-out arguments))))
    (append-map
     (lambda (a)
       (native-argument a (function-schema-is-out-fn? func) symint
                        use-const-ref use-ilistref))
     args)))

(define (native-returns-type returns symint use-const-ref)
  (cpp-returns-type returns symint use-const-ref))

;; ---------------------------------------------------------------------------
;; structured translation (torchgen api/structured.py).  symint is always off.
;; ---------------------------------------------------------------------------

(define (structured-argumenttype-type type mutable binds)
  (or (cpp-valuetype-type type binds mutable #f)
      (case (type-kind type)
        ((tensor) (make-named-ctype binds (make-const-ref-ctype (make-base-ctype %tensor-t))))
        ((scalar) (make-named-ctype binds (make-const-ref-ctype (make-base-ctype %scalar-t))))
        ((optional)
         (let ((elem (optional-element type)))
           (cond
            ((equal? elem (make-tensor-type))
             (make-named-ctype binds (make-base-ctype %optional-tensor-ref-t)))
            ((equal? elem (make-scalar-type))
             (make-named-ctype binds (make-base-ctype %optional-scalar-ref-t)))
            ((and (eq? (type-kind elem) 'list)
                  (string=? (type->string (list-type-element elem)) "int"))
             (make-named-ctype binds (make-base-ctype %optional-int-array-ref-t)))
            (else
             (let ((elem-nct (structured-argumenttype-type elem mutable binds)))
               (make-named-ctype binds (make-optional-ctype (named-ctype-type elem-nct))))))))
        ((list)
         (let ((elem (list-type-element type)))
           (cond
            ((equal? elem (make-tensor-type))
             (make-named-ctype binds (make-const-ref-ctype (make-base-ctype %i-tensor-list-ref-t))))
            ((equal? elem (make-optional-type (make-tensor-type)))
             (make-named-ctype binds (make-base-ctype %i-opt-tensor-list-ref-t)))
            ((string=? (type->string elem) "int")
             (make-named-ctype binds (make-base-ctype %int-array-ref-t)))
            (else
             (let ((elem-nct (structured-argumenttype-type elem mutable binds)))
               (make-named-ctype binds (make-array-ref-ctype (named-ctype-type elem-nct))))))))
        (else (error 'unrecognized-structured-argumenttype type)))))

(define (structured-argument-type argument binds)
  (structured-argumenttype-type (argument-type argument)
                                (argument-is-write? argument) binds))

(define (structured-argument argument)
  (cond
   ((argument? argument)
    (list (make-binding (argument-name argument)
                        (structured-argument-type argument (argument-name argument))
                        #f)))
   ((self-argument? argument)
    (structured-argument (self-argument-argument argument)))
   ((tensor-options-arguments? argument)
    (error 'structured-tensor-options))
   (else (error 'structured-argument-unknown argument))))

(define (precompute-arguments precomputed)
  ;; Returns (values replace add) where replace is an alist name -> (argument ...)
  ;; and add is a list of arguments.
  (define (split-comma s)
    (map string-trim-both (string-split s #\,)))
  (let* ((raw (precompute-placeholder-raw precomputed))
         (items (map yaml-scalar-value (yaml-sequence-items raw))))
    (let* ((last (last items))
           (add-items (if (string-contains last " -> ")
                          '()
                          (split-comma last)))
           (replace-items (if (string-contains last " -> ")
                              items
                              (drop-right items 1))))
      (values
       (map (lambda (item)
              (let* ((arrow (string-contains item " -> "))
                     (arg (substring item 0 arrow))
                     (with-list (split-comma (substring item (+ arrow 4)))))
                (cons arg (map parse-argument with-list))))
            replace-items)
       (map parse-argument add-items)))))

(define (structured-impl-arguments group)
  (let* ((out (native-functions-group-out group))
         (out-func (native-function-func out))
         (precomputed (native-function-precomputed out)))
    (if precomputed
        (call-with-values
            (lambda () (precompute-arguments precomputed))
          (lambda (replace add)
            (let* ((non-out (arguments-non-out (function-schema-arguments out-func)))
                   (replaced
                    (append-map
                     (lambda (a)
                       (if (argument? a)
                           (let ((entry (assoc-ref replace (argument-name a))))
                             (if entry entry (list a)))
                           (list a)))
                     non-out))
                   (args (append replaced add
                                 (arguments-out (function-schema-arguments out-func)))))
              (append-map structured-argument args))))
        (let ((args (append (arguments-non-out (function-schema-arguments out-func))
                            (arguments-out (function-schema-arguments out-func)))))
          (append-map structured-argument args)))))

(define (structured-meta-arguments group)
  (let ((functional-func (native-function-func (native-functions-group-functional group))))
    (append-map structured-argument
                (arguments-non-out (function-schema-arguments functional-func)))))

(define (structured-out-arguments group)
  (let ((out-func (native-function-func (native-functions-group-out group))))
    (append-map structured-argument
                (arguments-out (function-schema-arguments out-func)))))

;; ---------------------------------------------------------------------------
;; dispatcher translation (torchgen api/dispatcher.py).
;; ---------------------------------------------------------------------------

(define (dispatcher-name func) (cpp-name func))

(define (dispatcher-jit-arguments func)
  ;; jit_arguments == Arguments.flat_all.
  (arguments-all (function-schema-arguments func)))

(define (dispatcher-argument argument remove-non-owning-ref-types symint
                             use-const-ref use-ilistref)
  (make-binding (argument-name argument)
                (cpp-argumenttype-type (argument-type argument)
                                       (argument-is-write? argument)
                                       (argument-name argument)
                                       remove-non-owning-ref-types symint
                                       use-const-ref use-ilistref)
                #f))

(define (dispatcher-arguments func symint use-const-ref use-ilistref)
  (map (lambda (a)
         (dispatcher-argument a #f symint use-const-ref use-ilistref))
       (dispatcher-jit-arguments func)))

(define (dispatcher-returns-type returns symint use-const-ref)
  (cpp-returns-type returns symint use-const-ref))

;; ---------------------------------------------------------------------------
;; Signatures (torchgen api/types/signatures.py).
;; ---------------------------------------------------------------------------

;; A CppSignature is (cpp-sig func faithful symint method fallback-binding
;; cpp-no-default-args).
(define (cpp-signature func faithful symint method fallback-binding cpp-no-default-args)
  (list 'cpp-sig func faithful symint method fallback-binding cpp-no-default-args))

(define (cpp-signature-arguments sig use-const-ref use-ilistref)
  (cpp-arguments (function-schema-arguments (cadr sig))
                 (caddr sig)        ; faithful
                 (cadddr sig)       ; symint
                 (list-ref sig 4)   ; method
                 (list-ref sig 6)   ; cpp-no-default-args
                 use-const-ref use-ilistref))

(define (cpp-signature-name sig suppress-symint-suffix)
  (let* ((func (cadr sig))
         (faithful (caddr sig))
         (symint (cadddr sig))
         (fallback-binding (list-ref sig 5))
         (n (cpp-name func
                      #:faithful-name-for-out-overloads faithful
                      #:symint-overload (if suppress-symint-suffix #f symint))))
    (if fallback-binding (string-append "__dispatch_" n) n)))

(define (cpp-signature-returns-type sig use-const-ref)
  (cpp-returns-type (function-schema-returns (cadr sig))
                    (cadddr sig) use-const-ref))

(define* (cpp-signature-decl sig use-const-ref use-ilistref
                             #:key (suppress-symint-suffix #f)
                             (is-redispatching-fn #f)
                             (prefix ""))
  (let* ((returns (ctype-cpp-type (cpp-signature-returns-type sig use-const-ref)))
         (args (map binding-decl (cpp-signature-arguments sig use-const-ref use-ilistref)))
         (name (string-append prefix (cpp-signature-name sig suppress-symint-suffix))))
    (string-append
     returns " " name "("
     (string-join
      (if is-redispatching-fn
          (cons "c10::DispatchKeySet dispatchKeySet" args)
          args)
      ", ")
     ")")))

(define* (cpp-signature-defn sig use-const-ref use-ilistref
                             #:key (prefix ""))
  (let* ((returns (ctype-cpp-type (cpp-signature-returns-type sig use-const-ref)))
         (args (map binding-defn (cpp-signature-arguments sig use-const-ref use-ilistref)))
         (name (string-append prefix (cpp-signature-name sig #f))))
    (string-append returns " " name "(" (string-join args ", ") ")")))

(define* (cpp-signature-group-signatures func symint include-symint?
                                         fallback-binding cpp-no-default-args
                                         #:key (method #f))
  ;; Returns the ordered list of CppSignatures (signature, faithful_signature,
  ;; symint_signature, symint_faithful_signature filtered by include-symint?).
  (define (make-sig faithful symint)
    (cpp-signature func faithful symint method fallback-binding cpp-no-default-args))
  (let* ((args (function-schema-arguments func))
         (has-faithful? (or (arguments-tensor-options args)
                            (not (null? (arguments-out args)))))
         (signature (make-sig #f #f))
         (faithful-signature (and has-faithful? (make-sig #t #f)))
         (symint-signature (and (function-schema-has-symint? func) (make-sig #f #t)))
         (symint-faithful-signature
          (and has-faithful?
               (function-schema-has-symint? func)
               (make-sig #t #t))))
    (append (list signature)
            (if faithful-signature (list faithful-signature) '())
            (if include-symint?
                (append (if symint-signature (list symint-signature) '())
                        (if symint-faithful-signature (list symint-faithful-signature) '()))
                '()))))

;; DispatcherSignature (func prefix symint).
(define (dispatcher-signature-arguments func symint use-const-ref use-ilistref)
  (dispatcher-arguments func symint use-const-ref use-ilistref))

(define (dispatcher-signature-returns-type func symint use-const-ref)
  (dispatcher-returns-type (function-schema-returns func) symint use-const-ref))

(define (dispatcher-signature-name func prefix)
  (string-append prefix (dispatcher-name func)))

(define (dispatcher-signature-decl func prefix symint use-const-ref use-ilistref)
  (let* ((returns (ctype-cpp-type (dispatcher-signature-returns-type func symint use-const-ref)))
         (args (map binding-decl (dispatcher-signature-arguments func symint use-const-ref use-ilistref)))
         (name (dispatcher-signature-name func prefix)))
    (string-append returns " " name "(" (string-join args ", ") ")")))

(define (dispatcher-signature-defn func prefix symint use-const-ref use-ilistref)
  (let* ((returns (ctype-cpp-type (dispatcher-signature-returns-type func symint use-const-ref)))
         (args (map binding-defn (dispatcher-signature-arguments func symint use-const-ref use-ilistref)))
         (name (dispatcher-signature-name func prefix)))
    (string-append returns " " name "(" (string-join args ", ") ")")))

(define (dispatcher-signature-exprs func symint use-const-ref use-ilistref)
  (map (lambda (a) (make-expr (binding-name a) (binding-nctype a)))
       (dispatcher-signature-arguments func symint use-const-ref use-ilistref)))

;; NativeSignature (func symint prefix).
(define (native-signature-arguments func symint use-const-ref use-ilistref)
  (native-arguments func symint use-const-ref use-ilistref))

(define (native-signature-returns-type func symint use-const-ref)
  (native-returns-type (function-schema-returns func) symint use-const-ref))

(define (native-signature-name func prefix)
  (string-append prefix (native-name func)))

(define (native-signature-decl func prefix symint use-const-ref use-ilistref)
  (let* ((returns (ctype-cpp-type (native-signature-returns-type func symint use-const-ref)))
         (args (map binding-decl (native-signature-arguments func symint use-const-ref use-ilistref)))
         (name (native-signature-name func prefix)))
    (string-append returns " " name "(" (string-join args ", ") ")")))

(define (native-signature-defn func prefix symint use-const-ref use-ilistref)
  (let* ((returns (ctype-cpp-type (native-signature-returns-type func symint use-const-ref)))
         (args (map binding-defn (native-signature-arguments func symint use-const-ref use-ilistref)))
         (name (native-signature-name func prefix)))
    (string-append returns " " name "(" (string-join args ", ") ")")))

;; kernel_signature(f, backend-index, prefix="").
;; backend-index :: (dispatch-key index device-guard) where index is an alist
;; operator-name -> backend-metadata.  external is always #f in-tree.
(define (kernel-signature f backend-index prefix use-const-ref use-ilistref)
  (let* ((meta (backend-index-get-kernel f backend-index))
         (symint (and meta (backend-metadata-supports-symint? meta))))
    (native-signature-defn (native-function-func f) prefix symint
                           use-const-ref use-ilistref)))

;; ---------------------------------------------------------------------------
;; BackendIndex helpers.  A backend index is represented as a list
;; (dispatch-key index device-guard), where index is an alist op-name ->
;; backend-metadata.
;; ---------------------------------------------------------------------------

(define (backend-index-dispatch-key bi) (car bi))
(define (backend-index-index bi) (cadr bi))
(define (backend-index-device-guard? bi) (caddr bi))

(define (backend-index-primary bi group)
  (native-functions-group-out group))  ; use_out_as_primary always True

(define (backend-index-get-kernel-entry bi f)
  (let* ((name (operator-name->string (function-schema-name (native-function-func f)))))
    (assoc-ref (backend-index-index bi) name)))

(define (backend-index-get-kernel f bi)
  (backend-index-get-kernel-entry bi f))

(define (backend-index-has-kernel f bi)
  (and (backend-index-get-kernel f bi) #t))

(define (backend-index-get-kernel-group group bi)
  (backend-index-get-kernel-entry bi (backend-index-primary bi group)))

(define (backend-index-has-kernel-group group bi)
  (and (backend-index-get-kernel-group group bi) #t))

;; ---------------------------------------------------------------------------
;; translate (torchgen api/translate.py), with method and
;; allow_expensive_conversions hard-wired to #f (G2 always uses the defaults).
;; ---------------------------------------------------------------------------

(define %options-nct (make-named-ctype "options"
                                       (make-const-ref-ctype (make-base-ctype %tensor-options-t))))
(define %out-tensor-nct (make-named-ctype "out"
                                          (make-const-ref-ctype (make-base-ctype %tensor-t))))
(define %long-vec-ctype (make-vector-ctype (make-base-ctype %long-t)))
(define %long-sym-vec-ctype (make-vector-ctype (make-base-ctype %sym-int-t)))
(define %optional-long-vec-ctype (make-optional-ctype %long-vec-ctype))
(define %optional-scalar-ctype (make-optional-ctype (make-base-ctype %scalar-t)))
(define %optional-tensor-ctype (make-optional-ctype (make-base-ctype %tensor-t)))

(define (alist-set alist key value)
  (cons (cons key value)
        (filter (lambda (e) (not (equal? (car e) key))) alist)))

(define* (translate bindings goals #:key (allow-expensive-conversions #f)
                    (method #f))
  ;; bindings :: (listof Binding-or-Expr); goals :: (listof Binding-or-NamedCType).
  (define (normalize-binding b)
    (if (binding? b)
        (make-expr (binding-name b) (binding-nctype b))
        b))
  (define (normalize-goal g)
    (if (binding? g) (binding-nctype g) g))

  (let* ((binding-exprs (map normalize-binding bindings))
         (goal-ctypes (map normalize-goal goals)))
    (define ctx (make-variable '()))
    (define (ctx-ref key) (assoc-ref (variable-ref ctx) key))
    (define (ctx-set! key value)
      (variable-set! ctx (alist-set (variable-ref ctx) key value)))

    ;; forward inference
    (for-each
     (lambda (b)
       (let* ((t (expr-type b))   ; NamedCType
              (expr (expr-expr b))
              (ct (named-ctype-type t))
              (name (named-ctype-name t)))
         (ctx-set! t expr)
         (cond
          ((equal? ct (make-const-ref-ctype (make-optional-ctype (make-base-ctype %tensor-t))))
           (ctx-set! (make-named-ctype name (make-base-ctype %optional-tensor-ref-t))
                     (string-append "((" expr ".has_value() && (*" expr
                                    ").defined()) ? at::OptionalTensorRef(*" expr
                                    ") : at::OptionalTensorRef())")))
          ((equal? ct (make-const-ref-ctype (make-base-ctype %scalar-t)))
           (ctx-set! (make-named-ctype name (make-base-ctype %opmath-t))
                     (string-append "(" expr ").to<opmath_t>()")))
          ((equal? ct (make-const-ref-ctype (make-optional-ctype (make-base-ctype %scalar-t))))
           (ctx-set! (make-named-ctype name (make-base-ctype %optional-scalar-ref-t))
                     (string-append "(" expr ".has_value() ? at::OptionalScalarRef(&("
                                    expr ".value())) : at::OptionalScalarRef())")))
          ;; BaseCType(scalar_t) -> opmath_t (torchgen api/translate.py):
          ;; plain scalar_t (NOT at::Scalar) widens via static_cast.
          ((equal? ct (make-base-ctype "scalar_t"))
           (ctx-set! (make-named-ctype name (make-base-ctype %opmath-t))
                     (string-append "static_cast<opmath_t>(" expr ")")))
          ((equal? ct (make-const-ref-ctype
                       (make-list-ctype (make-optional-ctype (make-base-ctype %tensor-t)))))
           (ctx-set! (make-named-ctype name (make-base-ctype %i-opt-tensor-list-ref-t))
                     (string-append "at::IOptTensorListRef(" expr ")"))))))
     binding-exprs)

    ;; Add implicit bindings if the generated code is inside a Tensor method
    (when method
      (ctx-set! (make-named-ctype "self" (make-mut-ref-ctype (make-base-ctype %tensor-t)))
                "const_cast<Tensor&>(*this)")
      (ctx-set! (make-named-ctype "self" (make-const-ref-ctype (make-base-ctype %tensor-t)))
                "const_cast<Tensor&>(*this)"))

    (define (solve goal direct)
      (define (direct-solve g) (solve g #t))
      (let ((ty (named-ctype-type goal))
            (name (named-ctype-name goal)))
        (or (ctx-ref goal)
            (and (const-ref-ctype? ty)
                 (solve (make-named-ctype name (make-mut-ref-ctype (const-ref-ctype-elem ty)))
                        direct))
            (and (mut-ref-ctype? ty)
                 (solve (make-named-ctype name (mut-ref-ctype-elem ty)) direct))
            (and (equal? ty (make-array-ref-ctype (make-base-ctype %long-t)))
                 (solve (make-named-ctype name (make-base-ctype %int-array-ref-t))
                        direct))
            (and (not direct)
                 (or
                  ;; memory_format
                  (and (equal? goal (make-named-ctype "memory_format"
                                                     (make-optional-ctype (make-base-ctype %memory-format-t))))
                       (let ((mf (direct-solve
                                  (make-named-ctype 'possibly-redundant-memory-format
                                                    (make-optional-ctype (make-base-ctype %memory-format-t))))))
                         (and mf
                              (if (member %options-nct goal-ctypes)
                                  mf
                                  (or (let ((options (direct-solve %options-nct)))
                                        (and options
                                             (string-append
                                              "c10::impl::check_tensor_options_and_extract_memory_format("
                                              options ", " mf ")")))
                                      mf)))))
                  ;; options
                  (and (equal? goal (make-named-ctype "options" (make-base-ctype %tensor-options-t)))
                       (let ((dtype (direct-solve (make-named-ctype "dtype"
                                                                  (make-optional-ctype (make-base-ctype %scalar-type-t)))))
                             (pin-memory (direct-solve (make-named-ctype "pin_memory"
                                                                        (make-optional-ctype (make-base-ctype %bool-t)))))
                             (device (direct-solve (make-named-ctype "device"
                                                                    (make-optional-ctype (make-base-ctype %device-t)))))
                             (layout (direct-solve (make-named-ctype "layout"
                                                                    (make-optional-ctype (make-base-ctype %layout-t))))))
                         (and dtype pin-memory device layout
                              (string-append "TensorOptions().dtype(" dtype
                                             ").layout(" layout ").device(" device
                                             ").pinned_memory(" pin-memory ")"))))
                  ;; dtype
                  (and (equal? goal (make-named-ctype "dtype"
                                                     (make-optional-ctype (make-base-ctype %scalar-type-t))))
                       (or (let ((options (direct-solve %options-nct)))
                             (and options (string-append "c10::optTypeMetaToScalarType(" options ".dtype_opt())")))
                           (let ((out (direct-solve %out-tensor-nct)))
                             (and out (string-append out ".scalar_type()")))))
                  ;; layout
                  (and (equal? goal (make-named-ctype "layout"
                                                     (make-optional-ctype (make-base-ctype %layout-t))))
                       (or (let ((options (direct-solve %options-nct)))
                             (and options (string-append options ".layout_opt()")))
                           (let ((out (direct-solve %out-tensor-nct)))
                             (and out (string-append out ".layout()")))))
                  ;; device
                  (and (equal? goal (make-named-ctype "device"
                                                     (make-optional-ctype (make-base-ctype %device-t))))
                       (or (let ((options (direct-solve %options-nct)))
                             (and options (string-append options ".device_opt()")))
                           (let ((out (direct-solve %out-tensor-nct)))
                             (and out (string-append out ".device()")))))
                  ;; pin_memory
                  (and (equal? goal (make-named-ctype "pin_memory"
                                                     (make-optional-ctype (make-base-ctype %bool-t))))
                       (or (let ((options (direct-solve %options-nct)))
                             (and options (string-append options ".pinned_memory_opt()")))
                           (let ((out (direct-solve %out-tensor-nct)))
                             (and out "::std::nullopt"))))
                  ;; intArrayRef
                  (and (equal? ty (make-base-ctype %int-array-ref-t))
                       (or (direct-solve (make-named-ctype name %long-vec-ctype))
                           (let ((r (direct-solve (make-named-ctype name (make-base-ctype %sym-int-array-ref-t)))))
                             (and r (string-append "C10_AS_INTARRAYREF_SLOW(" r ")")))))
                  ;; symIntArrayRef
                  (and (equal? ty (make-base-ctype %sym-int-array-ref-t))
                       (or (let ((r (direct-solve (make-named-ctype name (make-base-ctype %int-array-ref-t)))))
                             (and r (string-append "c10::fromIntArrayRefSlow(" r ")")))
                           (direct-solve (make-named-ctype name %long-sym-vec-ctype))))
                  ;; SymInt
                  (and (equal? ty (make-base-ctype %sym-int-t))
                       (direct-solve (make-named-ctype name (make-base-ctype %long-t))))
                  ;; Optional(SymInt)
                  (and (equal? ty (make-optional-ctype (make-base-ctype %sym-int-t)))
                       (let ((argname (direct-solve (make-named-ctype name
                                                                    (make-optional-ctype (make-base-ctype %long-t))))))
                         (and argname
                              (string-append argname ".has_value() ? ::std::make_optional(c10::SymInt(*"
                                             argname ")) : ::std::nullopt"))))
                  ;; long
                  (and (equal? ty (make-base-ctype %long-t))
                       (let ((s (direct-solve (make-named-ctype name (make-base-ctype %sym-int-t)))))
                         (and s (string-append s ".guard_int(__FILE__, __LINE__)"))))
                  ;; Optional(long)
                  (and (equal? ty (make-optional-ctype (make-base-ctype %long-t)))
                       (let ((argname (direct-solve (make-named-ctype name
                                                                    (make-optional-ctype (make-base-ctype %sym-int-t))))))
                         (and argname
                              (string-append argname ".has_value() ? ::std::make_optional("
                                             argname "->guard_int(__FILE__, __LINE__)) : ::std::nullopt"))))
                  ;; optionalIntArrayRef
                  (and (equal? ty (make-base-ctype %optional-int-array-ref-t))
                       (or (direct-solve (make-named-ctype name %optional-long-vec-ctype))
                           (let ((argname (direct-solve (make-named-ctype name (make-base-ctype %optional-sym-int-array-ref-t)))))
                             (and argname
                                  (string-append argname ".has_value() ? ::std::make_optional(C10_AS_INTARRAYREF_SLOW(*"
                                                 argname ")) : ::std::nullopt")))))
                  ;; optionalSymIntArrayRef
                  (and (equal? ty (make-base-ctype %optional-sym-int-array-ref-t))
                       (let ((argname (direct-solve (make-named-ctype name (make-base-ctype %optional-int-array-ref-t)))))
                         (and argname
                              (string-append argname ".has_value() ? ::std::make_optional(c10::fromIntArrayRefSlow(*"
                                             argname ")) : ::std::nullopt"))))
                  ;; optionalScalarRef
                  (and (equal? ty (make-base-ctype %optional-scalar-ref-t))
                       (direct-solve (make-named-ctype name %optional-scalar-ctype)))
                  ;; optionalTensorRef
                  (and (equal? ty (make-base-ctype %optional-tensor-ref-t))
                       (direct-solve (make-named-ctype name %optional-tensor-ctype)))
                  ;; allow_expensive_conversions (torchgen api/translate.py):
                  ;; owning <- non-owning conversions used by ViewMeta decl.
                  (and allow-expensive-conversions
                       (or
                        ;; VectorCType(longT) <- IntArrayRef
                        (and (equal? ty %long-vec-ctype)
                             (let ((r (direct-solve (make-named-ctype name (make-base-ctype %int-array-ref-t)))))
                               (and r (string-append r ".vec()"))))
                        ;; VectorCType(SymIntT) <- SymIntArrayRef
                        (and (equal? ty %long-sym-vec-ctype)
                             (let ((r (direct-solve (make-named-ctype name (make-base-ctype %sym-int-array-ref-t)))))
                               (and r (string-append r ".vec()"))))
                        ;; BaseCType(tensorT) <- const-ref tensor
                        (and (equal? ty (make-base-ctype %tensor-t))
                             (direct-solve (make-named-ctype name
                                                             (make-const-ref-ctype (make-base-ctype %tensor-t)))))
                        ;; OptionalCType(tensorT) <- const-ref optional tensor
                        (and (equal? ty (make-optional-ctype (make-base-ctype %tensor-t)))
                             (direct-solve (make-named-ctype name
                                                             (make-const-ref-ctype (make-optional-ctype (make-base-ctype %tensor-t))))))
                        ;; OptionalCType(VectorCType(longT)) <- OptionalIntArrayRef
                        (and (equal? ty (make-optional-ctype %long-vec-ctype))
                             (let ((r (direct-solve (make-named-ctype name (make-base-ctype %optional-int-array-ref-t)))))
                               (and r (string-append r ".has_value() ? ::std::make_optional(" r "->vec()) : ::std::nullopt"))))
                        ;; OptionalCType(scalarT) <- OptionalScalarRef
                        (and (equal? ty (make-optional-ctype (make-base-ctype %scalar-t)))
                             (let ((r (direct-solve (make-named-ctype name (make-base-ctype %optional-scalar-ref-t)))))
                               (and r (string-append r ".has_value() ? ::std::make_optional(" r ") : ::std::nullopt"))))
                        ;; OptionalCType(tensorT) <- OptionalTensorRef (unreachable:
                        ;; the const-ref optional-tensor rule above always matches).
                        (and (equal? ty (make-optional-ctype (make-base-ctype %tensor-t)))
                             (let ((r (direct-solve (make-named-ctype name (make-base-ctype %optional-tensor-ref-t)))))
                               (and r (string-append r ".has_value() ? ::std::make_optional(" r ") : ::std::nullopt"))))))
                  ;; const_cast on tensors
                  (and (equal? ty (make-mut-ref-ctype (make-base-ctype %tensor-t)))
                       (let ((argname (direct-solve (make-named-ctype name
                                                                    (make-const-ref-ctype (make-base-ctype %tensor-t))))))
                         (and argname (string-append "const_cast<Tensor&>(" argname ")")))))))))

    (map (lambda (g)
           (make-expr (or (solve g #f) (error 'translate-unsat g)) g))
         goal-ctypes)))

;; ---------------------------------------------------------------------------
;; precomputed parsing (torchgen model.Precompute.parse).
;; ---------------------------------------------------------------------------

(define (parse-precomputed precomputed)
  ;; precomputed is a <precompute-placeholder>.  Returns (values replace add).
  (precompute-arguments precomputed))
