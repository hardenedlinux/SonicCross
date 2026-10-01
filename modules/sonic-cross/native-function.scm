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

(define-module (sonic-cross native-function)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:use-module (srfi srfi-13)
  #:use-module (sonic-cross core-ir)
  #:use-module (sonic-cross schema-parser)
  #:use-module (sonic-cross yaml)
  #:export (make-namespace-helper
            namespace-helper?
            namespace-helper-namespace
            namespace-helper-entity-name
            namespace-helper-from-namespaced-entity
            namespace-helper-get-cpp-namespace
            dispatch-keys
            dispatch-key?
            dispatch-key-parse
            make-backend-metadata
            backend-metadata?
            backend-metadata-kernel
            backend-metadata-structured
            backend-metadata-cpp-namespace
            backend-metadata-supports-symint?
            make-precompute-placeholder
            precompute-placeholder?
            precompute-placeholder-raw
            make-ufunc-inner-loop-placeholder
            ufunc-inner-loop-placeholder?
            ufunc-inner-loop-placeholder-raw
            make-native-function
            native-function?
            native-function-func
            native-function-namespace
            native-function-use-const-ref-for-mutable-tensors?
            native-function-device-guard?
            native-function-device-check
            native-function-python-module
            native-function-category-override
            native-function-variants
            native-function-manual-kernel-registration?
            native-function-manual-cpp-binding?
            native-function-loc
            native-function-autogen
            native-function-ufunc-inner-loop
            native-function-structured?
            native-function-structured-delegate
            native-function-structured-inherits
            native-function-precomputed
            native-function-cpp-no-default-args
            native-function-is-abstract?
            native-function-has-composite-implicit-autograd-kernel?
            native-function-has-composite-implicit-autograd-nested-tensor-kernel?
            native-function-has-composite-explicit-autograd-kernel?
            native-function-has-composite-explicit-autograd-non-functional-kernel?
            native-function-tags
            native-function-is-view-op?
            native-function-from-yaml
            make-native-functions-group
            native-functions-group?
            native-functions-group-functional
            native-functions-group-inplace
            native-functions-group-mutable
            native-functions-group-out
            native-functions-group-from-dict
            native-functions-group-structured?
            native-functions-group-root-name
            native-functions-group-functions
            native-functions-group-signature))

(define-record-type <namespace-helper>
  (make-namespace-helper namespace entity-name)
  namespace-helper?
  (namespace namespace-helper-namespace)
  (entity-name namespace-helper-entity-name))

(define (namespace-helper-from-namespaced-entity value max-level)
  (let* ((names (string-split value #\:))
         (filtered (let loop ((xs names) (result '()))
                     (cond ((null? xs) (reverse result))
                           ((string-null? (car xs))
                            (loop (cdr xs) result))
                           (else (loop (cdr xs) (cons (car xs) result))))))
         (entity (if (null? filtered) "" (last filtered)))
         (namespace-parts (if (null? filtered) '() (drop-right filtered 1))))
    (when (> (length namespace-parts) max-level)
      (error 'namespace-too-deep value max-level))
    (make-namespace-helper (string-join namespace-parts "::") entity)))

(define (namespace-helper-get-cpp-namespace helper default)
  (if (string-null? (namespace-helper-namespace helper))
      default
      (namespace-helper-namespace helper)))

(define dispatch-keys
  '("CPU" "SparseCPU" "SparseCsrCPU" "MkldnnCPU" "CUDA" "MPS" "XPU"
    "SparseXPU" "SparseCsrXPU" "SparseCUDA" "SparseCsrCUDA" "SparseMPS"
    "SparseCsrMPS" "QuantizedCPU" "QuantizedCUDA"
    "CompositeImplicitAutograd" "CompositeImplicitAutogradNestedTensor"
    "CompositeExplicitAutograd" "CompositeExplicitAutogradNonFunctional"
    "NestedTensorCPU" "NestedTensorCUDA" "NestedTensorXPU" "NestedTensorHPU"
    "Meta" "SparseMeta" "SparseCsrMeta" "QuantizedMeta" "NestedTensorMeta"
    "ZeroTensor" "MTIA"))

(define (dispatch-key? value)
  (and (string? value) (member value dispatch-keys)))

(define (dispatch-key-parse value)
  (unless (dispatch-key? value)
    (error 'invalid-dispatch-key value))
  value)

(define-record-type <backend-metadata>
  (make-backend-metadata kernel structured cpp-namespace)
  backend-metadata?
  (kernel backend-metadata-kernel)
  (structured backend-metadata-structured)
  (cpp-namespace backend-metadata-cpp-namespace))

(define (backend-metadata-supports-symint? value)
  (and (string-contains (backend-metadata-kernel value) "_symint") #t))

(define-record-type <precompute-placeholder>
  (make-precompute-placeholder raw)
  precompute-placeholder?
  (raw precompute-placeholder-raw))

(define-record-type <ufunc-inner-loop-placeholder>
  (make-ufunc-inner-loop-placeholder raw)
  ufunc-inner-loop-placeholder?
  (raw ufunc-inner-loop-placeholder-raw))

(define-record-type <native-function>
  (make-native-function func namespace use-const-ref-for-mutable-tensors
                        device-guard device-check python-module category-override
                        variants manual-kernel-registration manual-cpp-binding loc
                        autogen ufunc-inner-loop structured structured-delegate
                        structured-inherits precomputed cpp-no-default-args
                        is-abstract has-composite-implicit-autograd-kernel
                        has-composite-implicit-autograd-nested-tensor-kernel
                        has-composite-explicit-autograd-kernel
                        has-composite-explicit-autograd-non-functional-kernel
                        tags)
  native-function?
  (func native-function-func)
  (namespace native-function-namespace)
  (use-const-ref-for-mutable-tensors
   native-function-use-const-ref-for-mutable-tensors?)
  (device-guard native-function-device-guard?)
  (device-check native-function-device-check)
  (python-module native-function-python-module)
  (category-override native-function-category-override)
  (variants native-function-variants)
  (manual-kernel-registration native-function-manual-kernel-registration?)
  (manual-cpp-binding native-function-manual-cpp-binding?)
  (loc native-function-loc)
  (autogen native-function-autogen)
  (ufunc-inner-loop native-function-ufunc-inner-loop)
  (structured native-function-structured?)
  (structured-delegate native-function-structured-delegate)
  (structured-inherits native-function-structured-inherits)
  (precomputed native-function-precomputed)
  (cpp-no-default-args native-function-cpp-no-default-args)
  (is-abstract native-function-is-abstract?)
  (has-composite-implicit-autograd-kernel
   native-function-has-composite-implicit-autograd-kernel?)
  (has-composite-implicit-autograd-nested-tensor-kernel
   native-function-has-composite-implicit-autograd-nested-tensor-kernel?)
  (has-composite-explicit-autograd-kernel
   native-function-has-composite-explicit-autograd-kernel?)
  (has-composite-explicit-autograd-non-functional-kernel
   native-function-has-composite-explicit-autograd-non-functional-kernel?)
  (tags native-function-tags))

(define (mapping-entry mapping key)
  (find (lambda (entry)
          (and (yaml-scalar? (car entry))
               (string=? (yaml-scalar-value (car entry)) key)))
        (yaml-mapping-entries mapping)))

(define (mapping-value mapping key default)
  (let ((entry (mapping-entry mapping key)))
    (if entry (cadr entry) default)))

(define (required-scalar mapping key)
  (let ((value (mapping-value mapping key #f)))
    (unless (and value (yaml-scalar? value))
      (error 'missing-native-function-field key))
    (yaml-scalar-value value)))

(define (scalar-value value default)
  (cond ((not value) default)
        ((yaml-null? value) #f)
        ((yaml-scalar? value) (yaml-scalar-value value))
        (else (error 'expected-yaml-scalar value))))

(define (boolean-value value default)
  (let ((text (scalar-value value #f)))
    (cond ((not text) default)
          ((string=? text "true") #t)
          ((string=? text "false") #f)
          (else (error 'expected-yaml-boolean text)))))

(define (string-list-value value default)
  (cond ((not value) default)
        ((yaml-null? value) '())
        ((yaml-scalar? value) (list (yaml-scalar-value value)))
        ((yaml-sequence? value)
         (map (lambda (item)
                (unless (yaml-scalar? item)
                  (error 'expected-yaml-string item))
                (yaml-scalar-value item))
              (yaml-sequence-items value)))
        (else (error 'expected-yaml-string-list value))))

(define (split-comma-space value)
  (if (string-null? value) '() (string-split value #\,)))

(define (trimmed-split value)
  (map string-trim-both (split-comma-space value)))

(define (parse-variants value)
  (let ((values (if (and value (not (yaml-null? value)))
                    (trimmed-split (scalar-value value ""))
                    '("function"))))
    (unless (and (pair? values) (every (lambda (x) (member x '("function" "method"))) values))
      (error 'invalid-variants values))
    (unless (every (lambda (x) (not (string-null? x))) values)
      (error 'invalid-variants values))
    (delete-duplicates values string=?)))

(define (parse-autogen value)
  (map parse-operator-name
       (if (and value (not (yaml-null? value)))
           (trimmed-split (scalar-value value ""))
           '())))

(define (cpp-name func)
  (let* ((name (function-schema-name func))
         (base (base-operator-name-base (operator-name-base name))))
    (string-append base (if (function-schema-is-out-fn? func) "_out" ""))))

(define (all-function-arguments func)
  (arguments-all (function-schema-arguments func)))

(define (has-tensor-arg? func)
  (any (lambda (argument)
         (type-is-tensor-like? (argument-type argument)))
       (all-function-arguments func)))

(define (has-generator-arg? func)
  (any (lambda (argument)
         (and (eq? (type-kind (argument-type argument)) 'base-type)
              (string=? (base-type-name (argument-type argument)) "Generator")))
       (all-function-arguments func)))

(define composite-only
  '("CompositeImplicitAutograd"))
(define composite-nested-only
  '("CompositeImplicitAutogradNestedTensor"))
(define composite-both
  '("CompositeImplicitAutograd" "CompositeImplicitAutogradNestedTensor"))

(define (same-keys? actual expected)
  (equal? (sort (delete-duplicates actual string=?) string<?)
          (sort (delete-duplicates expected string=?) string<?)))

(define (dispatch-map-keys dispatch)
  (map car dispatch))

(define (dispatch-assoc-set mapping key value)
  (cons (cons key value)
        (filter (lambda (entry) (not (string=? (car entry) key))) mapping)))

(define structured-dispatch-keys
  '("CPU" "CUDA" "MPS" "XPU" "MTIA"))

(define (parse-dispatch raw structured func)
  (if (not raw)
      (let* ((name (base-operator-name-base
                    (operator-name-base (function-schema-name func))))
             (factory? (or (string-prefix? "new_" name)
                           (string-suffix? "_like" name)
                           (and (arguments-tensor-options
                                 (function-schema-arguments func))
                                (not (has-tensor-arg? func)))))
             (kernel (cpp-name func)))
        (when factory? (error 'factory-without-dispatch name))
                    (list (cons "CompositeImplicitAutograd"
                    (list (cons (operator-name->string
                                 (function-schema-name func))
                                (make-backend-metadata kernel #f "at::native"))))))
      (begin
        (unless (yaml-mapping? raw) (error 'expected-dispatch-mapping raw))
        (fold (lambda (entry result)
                (let ((key-node (car entry)) (value-node (cadr entry)))
                  (unless (and (yaml-scalar? key-node)
                               (yaml-scalar? value-node))
                    (error 'invalid-dispatch-entry entry))
                  (if (string=? (yaml-scalar-value key-node) "__line__")
                      result
                      (let* ((kernel-helper
                              (namespace-helper-from-namespaced-entity
                               (yaml-scalar-value value-node) 3))
                             (kernel-namespace
                              (namespace-helper-get-cpp-namespace kernel-helper "at")))
                        (fold (lambda (key current)
                                (let ((dispatch-key (dispatch-key-parse
                                                     (string-trim-both key))))
                                  (dispatch-assoc-set
                                   current dispatch-key
                                   (list
                                    (cons (operator-name->string
                                           (function-schema-name func))
                                          (make-backend-metadata
                                           (namespace-helper-entity-name kernel-helper)
                                           (and structured
                                                (member dispatch-key
                                                        structured-dispatch-keys)
                                                #t)
                                           (string-append kernel-namespace
                                                          "::native")))))))
                              result
                              (string-split (yaml-scalar-value key-node) #\,))))))
              '()
              (yaml-mapping-entries raw)))))

(define valid-tags
  '("pt2_compliant_tag" "out" "inplace" "nondeterministic_seeded"
    "core" "inplace_view" "view_copy" "generated"))

(define (tag-set tags)
  (let ((result (delete-duplicates tags string=?)))
    (unless (every (lambda (tag) (member tag valid-tags)) result)
      (error 'invalid-native-function-tag result))
    result))

(define (annotation-write? annotation)
  (and (annotation? annotation)
       (annotation-is-write? annotation)))

(define (annotation-wildcard-after? annotation)
  (and (annotation? annotation)
       (let ((after (annotation-alias-set-after annotation)))
         (or (eq? after '*)
             (and (pair? after) (member "*" after))))))

(define (native-function-is-view-op? value)
  (let* ((func (native-function-func value))
         (returns (function-schema-returns func))
         (non-mutating-view?
          (and (pair? returns)
               (any (lambda (return)
                     (let ((annotation (return-annotation return)))
                       (and (annotation? annotation)
                            (not (annotation-is-write? annotation)))))
                   returns)))
         (operator (operator-name->string (function-schema-name func)))
         (inplace-view?
          (and (member "inplace_view" (native-function-tags value))
               (not (string=? operator "resize_"))
               (not (string=? operator "resize_as_"))))
         (wildcard-view?
          (any (lambda (argument)
                (annotation-wildcard-after?
                 (argument-annotation argument)))
              (arguments-all (function-schema-arguments func)))))
    (or non-mutating-view? inplace-view? wildcard-view?)))

(define (native-function-from-yaml mapping)
  (unless (yaml-mapping? mapping) (error 'expected-native-function-mapping mapping))
  (let* ((entries (yaml-mapping-entries mapping))
         (known '("func" "namespace" "cpp_no_default_args" "use_const_ref_for_mutable_tensors"
                  "variants" "manual_kernel_registration" "manual_cpp_binding"
                  "device_guard" "device_check" "structured" "structured_delegate"
                  "structured_inherits" "python_module" "category_override"
                  "precomputed" "tags" "dispatch" "autogen" "ufunc_inner_loop"
                  "__line__"))
         (func (parse-function-schema (required-scalar mapping "func")))
         (namespace (let ((entry (mapping-entry mapping "namespace")))
                      (if entry (scalar-value (cadr entry) "aten") "aten")))
         (structured (boolean-value (mapping-value mapping "structured" #f) #f))
         (structured-delegate (scalar-value (mapping-value mapping "structured_delegate" #f) #f))
         (device-guard (boolean-value (mapping-value mapping "device_guard" #f) #t))
         (device-check (scalar-value (mapping-value mapping "device_check" #f) "ExactCheck"))
         (variants (parse-variants (mapping-value mapping "variants" #f)))
         (out? (function-schema-is-out-fn? func))
         (dispatch (parse-dispatch (mapping-value mapping "dispatch" #f)
                                   structured func))
         (dispatch-keys (dispatch-map-keys dispatch))
         (raw-tags (string-list-value (mapping-value mapping "tags" #f) '()))
         (base (operator-name-base (function-schema-name func)))
         (name (base-operator-name-base base))
         (tags (tag-set
                (append raw-tags
                        (if (and (string=? namespace "aten")
                                 (member "pt2_compliant_tag" valid-tags))
                            '("pt2_compliant_tag") '())
                        (if (and out? (member "out" valid-tags)) '("out") '())
                        (if (and (base-operator-name-inplace? base)
                                 (member "inplace" valid-tags)) '("inplace") '())
                        (if (or (and (string-contains (operator-name->string
                                                       (function-schema-name func)) "rand")
                                      (not (string-contains
                                            (operator-name->string
                                             (function-schema-name func)) "_philox_")))
                                  (string-contains (operator-name->string
                                                    (function-schema-name func)) "dropout")
                                  (has-generator-arg? func))
                            '("nondeterministic_seeded") '()))))
         (cpp-no-default-args
          (string-list-value (mapping-value mapping "cpp_no_default_args" #f) '()))
         (argument-names-with-defaults
          (map argument-name
               (filter argument-default
                       (all-function-arguments func))))
         (precomputed-raw (mapping-value mapping "precomputed" #f))
         (ufunc-raw (mapping-value mapping "ufunc_inner_loop" #f)))
    (unless (every (lambda (entry)
                    (and (yaml-scalar? (car entry))
                         (member (yaml-scalar-value (car entry)) known)))
                  entries)
      (let ((unknown
             (find (lambda (entry)
                     (or (not (yaml-scalar? (car entry)))
                         (not (member (yaml-scalar-value (car entry)) known))))
                   entries)))
        (error 'unknown-native-function-field unknown)))
    (when (and out? (not (equal? variants '("function"))))
      (error 'out-function-variants variants))
    (when (and (boolean-value (mapping-value mapping "use_const_ref_for_mutable_tensors" #f) #f)
               (pair? (arguments-out (function-schema-arguments func))))
      (error 'const-ref-out-function))
    (when structured
      (unless (eq? (function-schema-kind func) schema-kind-out)
        (error 'structured-non-out-function))
      (unless device-guard (error 'structured-device-guard)))
    (when structured-delegate
      (when out? (error 'delegate-out-function))
      (unless device-guard (error 'delegate-device-guard))
      (when (string-contains structured-delegate "::")
        (error 'namespaced-structured-delegate structured-delegate)))
    (when (and structured structured-delegate)
      (error 'structured-and-delegate))
    (let ((inherits (scalar-value (mapping-value mapping "structured_inherits" #f) #f)))
      (when (and inherits (or (not structured) (string-contains inherits "::")))
        (error 'invalid-structured-inherits inherits)))
    (when (and (mapping-value mapping "python_module" #f)
               (not (yaml-null? (mapping-value mapping "python_module" #f)))
               (member "method" variants))
      (error 'python-module-method-variant))
    (when (and precomputed-raw (not (yaml-null? precomputed-raw))
               (not structured))
      (error 'precomputed-non-structured))
    (unless (every (lambda (name) (member name argument-names-with-defaults))
                   cpp-no-default-args)
      (error 'invalid-cpp-no-default-args cpp-no-default-args))
    (when (and (string-prefix? "_foreach"
                              name)
               (not (string=? device-check "NoCheck")))
      (error 'foreach-device-check device-check))
    (let* ((composite? (same-keys? dispatch-keys composite-only))
           (composite-nested? (same-keys? dispatch-keys composite-nested-only))
           (composite-both? (same-keys? dispatch-keys composite-both))
           (is-abstract (or structured-delegate
                            (not (or composite? composite-nested? composite-both?)))))
      (values
       (make-native-function
        func namespace
        (boolean-value (mapping-value mapping "use_const_ref_for_mutable_tensors" #f) #f)
        device-guard device-check
        (scalar-value (mapping-value mapping "python_module" #f) #f)
        (scalar-value (mapping-value mapping "category_override" #f) #f)
        variants
        (boolean-value (mapping-value mapping "manual_kernel_registration" #f) #f)
        (boolean-value (mapping-value mapping "manual_cpp_binding" #f) #f)
        (scalar-value (mapping-value mapping "__line__" #f) #f)
        (parse-autogen (mapping-value mapping "autogen" #f))
        (and ufunc-raw
             (not (yaml-null? ufunc-raw))
             (make-ufunc-inner-loop-placeholder ufunc-raw))
        structured structured-delegate
        (scalar-value (mapping-value mapping "structured_inherits" #f) #f)
        (and precomputed-raw
             (not (yaml-null? precomputed-raw))
             (make-precompute-placeholder precomputed-raw))
        cpp-no-default-args is-abstract
        (and (member "CompositeImplicitAutograd" dispatch-keys) #t)
        (and (member "CompositeImplicitAutogradNestedTensor" dispatch-keys) #t)
        (and (member "CompositeExplicitAutograd" dispatch-keys) #t)
        (and (member "CompositeExplicitAutogradNonFunctional" dispatch-keys) #t)
        tags)
       dispatch))))

(define-record-type <native-functions-group>
  (make-native-functions-group functional inplace mutable out)
  native-functions-group?
  (functional native-functions-group-functional)
  (inplace native-functions-group-inplace)
  (mutable native-functions-group-mutable)
  (out native-functions-group-out))

(define (group-members group)
  (filter identity (list (native-functions-group-functional group)
                         (native-functions-group-inplace group)
                         (native-functions-group-mutable group)
                         (native-functions-group-out group))))

(define (native-functions-group-from-dict entries)
  (when (null? entries) (error 'empty-native-functions-group))
  (let ((keys (map car entries)))
    (unless (every (lambda (key) (member key '("functional" "inplace" "mutable" "out"))) keys)
      (error 'invalid-native-functions-group-key keys))
    (let* ((functional (assoc-ref entries "functional"))
           (inplace (assoc-ref entries "inplace"))
           (mutable (assoc-ref entries "mutable"))
           (out (assoc-ref entries "out"))
           (members (filter identity (list functional inplace mutable out))))
      (cond
       ((= (length members) 1) #f)
       ((not out) #f)
       (else
        (unless functional (error 'group-missing-functional))
        (unless (eq? (function-schema-kind (native-function-func functional)) schema-kind-functional)
          (error 'group-functional-kind))
        (for-each
       (lambda (function)
         (unless (equal? (function-schema-signature
                         (native-function-func function))
                        (function-schema-signature
                         (native-function-func functional)))
           (error 'group-signature-mismatch))
         (unless (string=? (native-function-namespace function)
                           (native-function-namespace functional))
           (error 'group-namespace-mismatch)))
         members)
        (when inplace
        (unless (eq? (function-schema-kind (native-function-func inplace)) schema-kind-inplace)
          (error 'group-inplace-kind)))
        (when mutable
        (unless (eq? (function-schema-kind (native-function-func mutable)) schema-kind-mutable)
          (error 'group-mutable-kind))
        (unless (operator-name-functional-overload?
                 (function-schema-name (native-function-func mutable)))
          (error 'group-mutable-functional-overload)))
        (unless (eq? (function-schema-kind (native-function-func out)) schema-kind-out)
        (error 'group-out-kind))
        (let ((structured (native-function-structured? functional)))
        (unless (every (lambda (function)
                        (eq? structured (native-function-structured? function)))
                       members)
          (error 'group-structured-mismatch)))
        (when (native-function-structured-delegate functional)
          (unless (string=?
                   (native-function-structured-delegate functional)
                   (operator-name->string
                    (function-schema-name (native-function-func out))))
            (error 'group-structured-delegate-mismatch)))
        (unless (every (lambda (function)
                      (equal? (native-function-autogen function)
                              (native-function-autogen functional)))
                     members)
        (error 'group-autogen-mismatch))
        (make-native-functions-group functional inplace mutable out))))))

(define (native-functions-group-structured? group)
  (native-function-structured? (native-functions-group-functional group)))

(define (native-functions-group-root-name group)
  (base-operator-name-base
   (operator-name-base
    (function-schema-name
     (native-function-func (native-functions-group-functional group))))))

(define (native-functions-group-functions group)
  (group-members group))

(define (native-functions-group-signature group)
  (function-schema-signature
   (native-function-func (native-functions-group-functional group))))
