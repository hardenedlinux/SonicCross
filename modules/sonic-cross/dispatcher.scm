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

;; The dispatcher API type translation (torchgen api/cpp.py + api/dispatcher.py
;; + api/types).  This maps a JIT function schema to the unboxed dispatcher
;; calling convention used by RegistrationDeclarations.h (G3): the C++ return
;; type, the dispatcher name (base name + "_out"), and the flattened argument
;; list with C++ argument types.  The dispatcher defaults symint=True and
;; remove_non_owning_ref_types=False, which are hard-wired here.

(define-module (sonic-cross dispatcher)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-13)
  #:use-module (sonic-cross core-ir)
  #:use-module (sonic-cross native-function)
  #:export (dispatcher-name
            registration-declaration
            dispatcher-signature-type
            dispatcher-signature-defn))

;; ---------------------------------------------------------------------------
;; Names
;; ---------------------------------------------------------------------------

(define (base-operator-spelling name)
  ;; Mirrors torchgen str(BaseOperatorName): the full base spelling carrying the
  ;; "_"/"_functional" suffix and dunder-method wrapping, without the overload.
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

(define (dispatcher-name func)
  ;; Mirrors torchgen cpp.name(func): str(BaseOperatorName), plus "_out" for
  ;; out functions.
  (string-append
   (base-operator-spelling (function-schema-name func))
   (if (function-schema-is-out-fn? func) "_out" "")))

;; ---------------------------------------------------------------------------
;; Value types (torchgen api/cpp.py valuetype_type with symint=True)
;; ---------------------------------------------------------------------------

(define (optional-element type)
  (car (type-arguments type)))

(define (base-type-cpp-string name)
  ;; torchgen api/types.py BaseTypeToCppMapping, restricted to the base-type
  ;; spellings that actually occur (Tensor/Scalar/int/float/bool/str/SymInt/
  ;; MemoryFormat are distinct type kinds, not base-type records).
  (cond
   ((string=? name "Generator") "at::Generator")
   ((string=? name "ScalarType") "at::ScalarType")
   ((string=? name "DimVector") "at::DimVector")
   ((string=? name "Layout") "at::Layout")
   ((string=? name "Device") "at::Device")
   ((string=? name "DeviceIndex") "at::DeviceIndex")
   ((string=? name "QScheme") "at::QScheme")
   ((string=? name "Storage") "at::Storage")
   ((string=? name "Stream") "at::Stream")
   ((string=? name "SymBool") "c10::SymBool")
   (else (error 'unknown-base-type name))))

(define (valuetype-type type)
  ;; Returns the C++ value type string, or #f when the type is not a value type
  ;; (Tensor, Scalar, tensor/list wrappers of them, custom classes, ...).
  (case (type-kind type)
    ((tensor scalar) #f)
    ((int) "int64_t")
    ((float) "double")
    ((bool) "bool")
    ((string) "c10::string_view")
    ((sym-int) "c10::SymInt")
    ((memory-format) "at::MemoryFormat")
    ((base-type) (base-type-cpp-string (base-type-name type)))
    ((optional)
     (let ((elem (valuetype-type (optional-element type))))
       (and elem (string-append "::std::optional<" elem ">"))))
    ((list)
     (if (string=? (type->string (list-type-element type)) "bool")
         (let ((size (list-type-size type)))
           (unless size (error 'bool-list-type-must-have-size))
           (string-append "::std::array<bool," (number->string size) ">"))
         #f))
    (else #f)))

;; ---------------------------------------------------------------------------
;; Argument types (torchgen api/cpp.py argumenttype_type)
;; ---------------------------------------------------------------------------

(define (argumenttype-type type mutable use-const-ref use-ilistref)
  (or (valuetype-type type)
      (case (type-kind type)
        ((tensor)
         (if (and mutable (not use-const-ref))
             "at::Tensor &"
             "const at::Tensor &"))
        ((scalar) "const at::Scalar &")
        ((optional)
         (let* ((elem (optional-element type))
                (elem-str (type->string elem)))
           (cond
            ((string=? elem-str "Tensor")
             (if (and mutable (not use-const-ref))
                 "at::Tensor &"
                 "const ::std::optional<at::Tensor> &"))
            ((string=? elem-str "Scalar")
             "const ::std::optional<at::Scalar> &")
            ((and (eq? (type-kind elem) 'list)
                  (string=? (type->string (list-type-element elem)) "int"))
             "at::OptionalIntArrayRef")
            ((and (eq? (type-kind elem) 'list)
                  (string=? (type->string (list-type-element elem)) "SymInt"))
             "at::OptionalSymIntArrayRef")
            (else
             (string-append
              "::std::optional<"
              (argumenttype-type elem mutable use-const-ref use-ilistref)
              ">")))))
        ((list)
         (let ((elem-str (type->string (list-type-element type))))
           (cond
            ((string=? elem-str "int") "at::IntArrayRef")
            ((string=? elem-str "SymInt") "c10::SymIntArrayRef")
            ((string=? elem-str "Tensor")
             (if use-ilistref "const at::ITensorListRef &" "at::TensorList"))
            ((string=? elem-str "Scalar") "at::ArrayRef<at::Scalar>")
            ((string=? elem-str "Tensor?")
             "const c10::List<::std::optional<at::Tensor>> &")
            (else
             (string-append
              "at::ArrayRef<"
              (argumenttype-type (list-type-element type)
                                 mutable use-const-ref use-ilistref)
              ">")))))
        (else (error 'unrecognized-argument-type type)))))

;; ---------------------------------------------------------------------------
;; Return types (torchgen api/cpp.py returntype_type / returns_type)
;; ---------------------------------------------------------------------------

(define (return-is-write? return)
  (and (return-annotation return)
       (annotation? (return-annotation return))
       (annotation-is-write? (return-annotation return))))

(define (returntype-type type mutable use-const-ref)
  ;; symint is always respected for return types (hard-wired True here).
  (or (valuetype-type type)
      (case (type-kind type)
        ((tensor)
         (if mutable
             (if use-const-ref "const at::Tensor &" "at::Tensor &")
             "at::Tensor"))
        ((scalar) "at::Scalar")
        ((list)
         (when mutable (error 'mutable-tensor-list-return))
         (when (list-type-size type) (error 'fixed-size-list-return))
         (string-append "::std::vector<"
                        (returntype-type (list-type-element type) #f use-const-ref)
                        ">"))
        ((optional)
         (let ((elem (optional-element type)))
           (if (string=? (type->string elem) "Tensor")
               (string-append "::std::optional<"
                              (returntype-type elem mutable use-const-ref)
                              ">")
               (error 'unrecognized-return-type type))))
        (else (error 'unrecognized-return-type type)))))

(define (returns-type returns use-const-ref)
  (cond
   ((null? returns) "void")
   ((null? (cdr returns))
    (returntype-type (return-type (car returns))
                     (return-is-write? (car returns))
                     use-const-ref))
   (else
    (string-append
     "::std::tuple<"
     (string-join
      (map (lambda (return)
             (returntype-type (return-type return)
                              (return-is-write? return)
                              use-const-ref))
           returns)
      ",")
     ">"))))

;; ---------------------------------------------------------------------------
;; Dispatcher signature (torchgen api/types/signatures.py DispatcherSignature)
;; ---------------------------------------------------------------------------

(define (dispatcher-signature-type func use-const-ref use-ilistref)
  ;; DispatcherSignature.type(): "{returns_type} ({arg_types})", argument C++
  ;; types only (no names).  The dispatcher is oblivious of defaults, so this
  ;; is the C++ function type used by MethodOperators.h `using schema = ...`.
  (string-append
   (returns-type (function-schema-returns func) use-const-ref)
   " ("
   (string-join
    (map (lambda (argument)
           (argumenttype-type (argument-type argument)
                              (argument-is-write? argument)
                              use-const-ref use-ilistref))
         (arguments-all (function-schema-arguments func)))
    ", ")
   ")"))

(define (dispatcher-signature-defn func use-const-ref use-ilistref name
                                   is-redispatching?)
  ;; DispatcherSignature.defn(name, is_redispatching_fn): "{returns_type}
  ;; {name}({args})", where each arg is "type name" (no defaults).  A
  ;; redispatch fn prepends "c10::DispatchKeySet dispatchKeySet".
  (let ((args
         (map (lambda (argument)
                (string-append
                 (argumenttype-type (argument-type argument)
                                    (argument-is-write? argument)
                                    use-const-ref use-ilistref)
                 " "
                 (argument-name argument)))
              (arguments-all (function-schema-arguments func)))))
    (string-append
     (returns-type (function-schema-returns func) use-const-ref)
     " " name "("
     (if is-redispatching?
         (string-append "c10::DispatchKeySet dispatchKeySet"
                        (if (null? args) "" ", "))
         "")
     (string-join args ", ")
     ")")))

;; ---------------------------------------------------------------------------
;; Registration declaration (torchgen gen.py compute_registration_declarations)
;; ---------------------------------------------------------------------------

(define (string-replace-all s target replacement)
  (let loop ((result "") (rest s))
    (let ((idx (string-contains rest target)))
      (if (not idx)
          (string-append result rest)
          (loop (string-append result (substring rest 0 idx) replacement)
                (substring rest (+ idx (string-length target))))))))

(define (json-dumps-string s)
  ;; Python json.dumps(s) with ensure_ascii=True.  The only characters needing
  ;; escaping in the canonical schema strings are '"' and '\\' (verified against
  ;; the frozen oracle); control characters and non-ASCII never occur.
  (let* ((s (string-replace-all s "\\" "\\\\"))
         (s (string-replace-all s "\"" "\\\"")))
    (string-append "\"" s "\"")))

(define (dispatch-field-string dispatch-keys)
  ;; Mirrors the "dispatch" comment field: True unless the set of dispatch keys
  ;; is exactly {CompositeImplicitAutograd} or exactly
  ;; {CompositeImplicitAutograd, CompositeImplicitAutogradNestedTensor}.
  (let ((sorted (sort (delete-duplicates dispatch-keys string=?) string<?)))
    (if (or (equal? sorted '("CompositeImplicitAutograd"))
            (equal? sorted '("CompositeImplicitAutograd"
                             "CompositeImplicitAutogradNestedTensor")))
        "False"
        "True")))

(define (default-field-string f)
  ;; Mirrors the "default" comment field: has_composite_kernel or
  ;; has_autogenerated_composite_kernel.
  (let* ((has-cia (native-function-has-composite-implicit-autograd-kernel? f))
         (has-cea (native-function-has-composite-explicit-autograd-kernel? f))
         (has-ceanf
          (native-function-has-composite-explicit-autograd-non-functional-kernel? f))
         (has-composite (or has-cia has-cea has-ceanf))
         (kind (function-schema-kind (native-function-func f)))
         (autogen-composite
          (and (or (native-function-structured? f)
                   (native-function-structured-delegate f))
               (or (eq? kind schema-kind-functional)
                   (eq? kind schema-kind-inplace)))))
    (if (or has-composite autogen-composite) "True" "False")))

(define (registration-declaration f)
  ;; One native function -> the RegistrationDeclarations.h declaration string
  ;; (terminated by a newline), matching compute_registration_declarations.
  (let* ((func (native-function-func f))
         (name (dispatcher-name func))
         (use-const-ref (native-function-use-const-ref-for-mutable-tensors? f))
         (use-ilistref (or (native-function-structured? f)
                           (native-function-structured-delegate f)))
         (returns-type (returns-type (function-schema-returns func)
                                     use-const-ref))
         (args-str
          (string-join
           (map (lambda (argument)
                  (string-append
                   (argumenttype-type (argument-type argument)
                                      (argument-is-write? argument)
                                      use-const-ref use-ilistref)
                   " "
                   (argument-name argument)))
                (arguments-all (function-schema-arguments func)))
           ", "))
         (schema (string-append "aten::" (function-schema->string func)))
         (comment
          (string-append
           "{\"schema\": " (json-dumps-string schema)
           ", \"dispatch\": "
           (json-dumps-string (dispatch-field-string
                               (native-function-dispatch-keys f)))
           ", \"default\": "
           (json-dumps-string (default-field-string f))
           "}")))
    (string-append returns-type " " name "(" args-str "); // " comment "\n")))
