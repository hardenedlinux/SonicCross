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

(define-module (sonic-cross core-ir)
  #:use-module (ice-9 regex)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:export
  (make-base-operator-name
   base-operator-name?
   base-operator-name-base
   base-operator-name-inplace?
   base-operator-name-dunder-method?
   base-operator-name-functional-overload?
   base-operator-name-hash
   make-operator-name
   operator-name?
   operator-name-base
   operator-name-overload-name
   operator-name-inplace?
   operator-name-dunder-method?
   operator-name-functional-overload?
   operator-name-hash
   operator-name->string
   parse-operator-name
   schema-kind-functional
   schema-kind-inplace
   schema-kind-mutable
   schema-kind-out
   schema-kind?
   view-schema-kind-aliasing
   view-schema-kind-aliasing-inplace
   view-schema-kind-non-aliasing
   view-schema-kind?
   make-type
   type?
   type-kind
   type-arguments
   make-tensor-type
   make-scalar-type
   make-any-type
   make-number-type
   make-int-type
   make-float-type
   make-bool-type
   make-string-type
   make-sym-int-type
   make-memory-format-type
   make-dimname-type
   make-const-type
   make-base-type
   base-type-name
   make-optional-type
   make-list-type
   list-type-element
   list-type-size
   tensor-type
   scalar-type
   any-type
   number-type
   int-type
   float-type
   bool-type
   string-type
   sym-int-type
   memory-format-type
   dimname-type
   const-type
   base-type
   type-is-tensor-like?
   tensor-like?
   has-symint?
   type-has-symint?
   type-is-symint-like?
   arguments-flat-non-out
   function-schema-has-symint?
   type->string
   type-hash
   make-argument
   argument?
   argument-name
   argument-type
   argument-default
   argument-annotation
   argument-is-write?
   argument-hash
   argument->string
   make-return
   return?
   return-name
   return-type
   return-annotation
   return-hash
   return->string
   make-annotation
   annotation?
   annotation-alias-set
   annotation-is-write?
   annotation-alias-set-after
   annotation->string
   make-self-argument
   self-argument?
   self-argument-argument
   make-tensor-options-arguments
   tensor-options-arguments?
   tensor-options-arguments-dtype
   tensor-options-arguments-layout
   tensor-options-arguments-device
   tensor-options-arguments-pin-memory
   tensor-options-arguments-all
   make-arguments
   arguments?
   arguments-pre-self-positional
   arguments-self-arg
   arguments-post-self-positional
   arguments-pre-tensor-options-kwarg-only
   arguments-tensor-options
   arguments-post-tensor-options-kwarg-only
   arguments-out
   arguments-all
   arguments-hash
   make-function-schema
   function-schema?
   function-schema-name
   function-schema-arguments
   function-schema-returns
   function-schema-kind
   function-schema-is-out-fn?
   is-out-fn?
   function-schema->string
   function-schema-signature
   signature
   function-schema-hash))

;; Records have no exported mutators: semantic values are constructed once
;; and compared structurally by Guile's equal? implementation.
(define-record-type <base-operator-name>
  (make-base-operator-name base inplace dunder-method functional-overload)
  base-operator-name?
  (base base-operator-name-base)
  (inplace base-operator-name-inplace?)
  (dunder-method base-operator-name-dunder-method?)
  (functional-overload base-operator-name-functional-overload?))

(define (base-operator-name-hash value)
  (hash value 31))

(define-record-type <operator-name>
  (make-operator-name base overload-name)
  operator-name?
  (base operator-name-base)
  (overload-name operator-name-overload-name))

(define (operator-name-inplace? value)
  (base-operator-name-inplace? (operator-name-base value)))

(define (operator-name-dunder-method? value)
  (base-operator-name-dunder-method? (operator-name-base value)))

(define (operator-name-functional-overload? value)
  (base-operator-name-functional-overload? (operator-name-base value)))

(define (operator-name-hash value)
  (hash value 31))

(define %augmented-assignment-names
  '("add" "sub" "mul" "div" "mod" "pow" "lshift" "rshift"
    "and" "xor" "or"))

(define (augmented-assignment-name? name)
  (member name %augmented-assignment-names))

(define (strip-dunder-namespace name)
  (let ((separator (string-contains name "::")))
    (if separator
        (substring name (+ separator 2))
        name)))

(define (dunder-base name)
  (let* ((candidate (strip-dunder-namespace name))
         (match (string-match "^__([^_]+)__$" candidate)))
    (and match (match:substring match 1))))

(define (make-parsed-base base)
  (let ((dunder (dunder-base base)))
    (if dunder
        (begin
          (when (string-suffix? "_functional" dunder)
            (error 'invalid-functional-overload base))
          (if (string-prefix? "i" dunder)
              (if (augmented-assignment-name? (substring dunder 1))
                  (make-base-operator-name
                   (substring dunder 1) #t #t #f)
                  (error 'invalid-dunder-name base))
              (make-base-operator-name dunder #f #t #f)))
        (let* ((functional-suffix "_functional")
               (functional-overload
                (and (string-suffix? functional-suffix base)
                     (> (string-length base)
                        (string-length functional-suffix))))
               (without-functional
                (if functional-overload
                    (substring base 0
                               (- (string-length base)
                                  (string-length functional-suffix)))
                    base))
               (inplace (and (positive? (string-length without-functional))
                             (char=? (string-ref without-functional
                                                  (1- (string-length without-functional)))
                                     #\_))))
          (when (and functional-overload inplace)
            (error 'invalid-functional-overload base))
          (make-base-operator-name
           (if inplace
               (substring without-functional 0
                          (1- (string-length without-functional)))
               without-functional)
           inplace #f functional-overload)))))

(define (parse-operator-name value)
  (let ((dot (string-index value #\.)))
    (if dot
        (make-operator-name
         (make-parsed-base (substring value 0 dot))
         (substring value (1+ dot)))
        (make-operator-name (make-parsed-base value) ""))))

(define (operator-name->string value)
  (let* ((base (operator-name-base value))
         (base-name (base-operator-name-base base))
         (spelling (if (base-operator-name-dunder-method? base)
                       (string-append "__"
                                      (if (base-operator-name-inplace? base)
                                          (string-append "i" base-name)
                                          base-name)
                                      "__")
                       (if (base-operator-name-inplace? base)
                           (string-append base-name "_")
                           (if (base-operator-name-functional-overload? base)
                               (string-append base-name "_functional")
                               base-name))))
         (overload (operator-name-overload-name value)))
    (if (string-null? overload)
        spelling
        (string-append spelling "." overload))))

(define schema-kind-functional 'functional)
(define schema-kind-inplace 'inplace)
(define schema-kind-mutable 'mutable)
(define schema-kind-out 'out)

(define (schema-kind? value)
  (memq value (list schema-kind-functional schema-kind-inplace
                    schema-kind-mutable schema-kind-out)))

(define view-schema-kind-aliasing 'aliasing)
(define view-schema-kind-aliasing-inplace 'aliasing_inplace)
(define view-schema-kind-non-aliasing 'non_aliasing)

(define (view-schema-kind? value)
  (memq value (list view-schema-kind-aliasing
                    view-schema-kind-aliasing-inplace
                    view-schema-kind-non-aliasing)))

(define-record-type <type>
  (make-type kind arguments)
  type?
  (kind type-kind)
  (arguments type-arguments))

(define (type-hash value)
  (hash value 31))

(define (simple-type kind)
  (make-type kind '()))

(define tensor-type (simple-type 'tensor))
(define scalar-type (simple-type 'scalar))
(define any-type (simple-type 'any))
(define number-type (simple-type 'number))
(define int-type (simple-type 'int))
(define float-type (simple-type 'float))
(define bool-type (simple-type 'bool))
(define string-type (simple-type 'string))
(define sym-int-type (simple-type 'sym-int))
(define memory-format-type (simple-type 'memory-format))
(define dimname-type (simple-type 'dimname))
(define const-type (simple-type 'const))
(define base-type (simple-type 'base-type))

(define (make-tensor-type) tensor-type)
(define (make-scalar-type) scalar-type)
(define (make-any-type) any-type)
(define (make-number-type) number-type)
(define (make-int-type) int-type)
(define (make-float-type) float-type)
(define (make-bool-type) bool-type)
(define (make-string-type) string-type)
(define (make-sym-int-type) sym-int-type)
(define (make-memory-format-type) memory-format-type)
(define (make-dimname-type) dimname-type)
(define (make-const-type) const-type)
(define* (make-base-type #:optional (name #f))
  (if name
      (make-type 'base-type (list name))
      base-type))

(define (make-optional-type element-type)
  (make-type 'optional (list element-type)))

(define* (make-list-type element-type #:optional (size #f))
  (make-type 'list (list element-type size)))

(define (list-type-element value)
  (car (type-arguments value)))

(define (list-type-size value)
  (cadr (type-arguments value)))

(define (base-type-name value)
  (and (eq? (type-kind value) 'base-type)
       (pair? (type-arguments value))
       (car (type-arguments value))))

(define (type-is-tensor-like? value)
  (or (eq? (type-kind value) 'tensor)
      (and (pair? (type-arguments value))
           (or (eq? (type-kind value) 'optional)
               (eq? (type-kind value) 'list))
           (type-is-tensor-like? (car (type-arguments value))))))

(define tensor-like? type-is-tensor-like?)

(define (has-symint? value)
  (or (eq? (type-kind value) 'sym-int)
      (any has-symint? (type-arguments value))))

(define type-has-symint? has-symint?)

(define (type-is-symint-like? value)
  (or (eq? (type-kind value) 'sym-int)
      (and (pair? (type-arguments value))
           (or (eq? (type-kind value) 'optional)
               (eq? (type-kind value) 'list))
           (type-is-symint-like? (car (type-arguments value))))))

(define-record-type <argument>
  (make-argument name type default annotation is-write)
  argument?
  (name argument-name)
  (type argument-type)
  (default argument-default)
  (annotation argument-annotation)
  (is-write argument-is-write?))

(define (argument-hash value)
  (hash value 31))

(define-record-type <return>
  (make-return name type annotation)
  return?
  (name return-name)
  (type return-type)
  (annotation return-annotation))

(define (return-hash value)
  (hash value 31))

(define-record-type <annotation>
  (make-annotation alias-set is-write alias-set-after)
  annotation?
  (alias-set annotation-alias-set)
  (is-write annotation-is-write?)
  (alias-set-after annotation-alias-set-after))

(define (annotation->string value)
  (let ((before (string-join (annotation-alias-set value) "|"))
        (after (annotation-alias-set-after value)))
    (string-append before
                   (if (annotation-is-write? value) "!" "")
                   (if after
                       (string-append " -> "
                                      (if (eq? after '*)
                                          "*"
                                          (string-join after "|")))
                       ""))))

(define-record-type <self-argument>
  (make-self-argument argument)
  self-argument?
  (argument self-argument-argument))

(define-record-type <tensor-options-arguments>
  (make-tensor-options-arguments dtype layout device pin-memory)
  tensor-options-arguments?
  (dtype tensor-options-arguments-dtype)
  (layout tensor-options-arguments-layout)
  (device tensor-options-arguments-device)
  (pin-memory tensor-options-arguments-pin-memory))

(define (tensor-options-arguments-all value)
  (list (tensor-options-arguments-dtype value)
        (tensor-options-arguments-layout value)
        (tensor-options-arguments-device value)
        (tensor-options-arguments-pin-memory value)))

(define (self-argument-value value)
  (if (self-argument? value)
      (self-argument-argument value)
      value))

(define (tensor-options-values value)
  (if (tensor-options-arguments? value)
      (tensor-options-arguments-all value)
      (list value)))

(define-record-type <arguments>
  (make-arguments pre-self-positional self-arg post-self-positional
                  pre-tensor-options-kwarg-only tensor-options
                  post-tensor-options-kwarg-only out)
  arguments?
  (pre-self-positional arguments-pre-self-positional)
  (self-arg arguments-self-arg)
  (post-self-positional arguments-post-self-positional)
  (pre-tensor-options-kwarg-only
   arguments-pre-tensor-options-kwarg-only)
  (tensor-options arguments-tensor-options)
  (post-tensor-options-kwarg-only
   arguments-post-tensor-options-kwarg-only)
  (out arguments-out))

(define (arguments-all value)
  (append (arguments-pre-self-positional value)
          (if (arguments-self-arg value)
              (list (self-argument-value (arguments-self-arg value)))
              '())
          (arguments-post-self-positional value)
          (arguments-pre-tensor-options-kwarg-only value)
          (if (arguments-tensor-options value)
              (tensor-options-values (arguments-tensor-options value))
              '())
          (arguments-post-tensor-options-kwarg-only value)
          (arguments-out value)))

(define (arguments-flat-non-out value)
  (append (arguments-pre-self-positional value)
          (if (arguments-self-arg value)
              (list (self-argument-value (arguments-self-arg value)))
              '())
          (arguments-post-self-positional value)
          (arguments-pre-tensor-options-kwarg-only value)
          (if (arguments-tensor-options value)
              (tensor-options-values (arguments-tensor-options value))
              '())
          (arguments-post-tensor-options-kwarg-only value)))

(define (arguments-hash value)
  (hash value 31))

(define-record-type <function-schema>
  (make-function-schema name arguments returns)
  function-schema?
  (name function-schema-name)
  (arguments function-schema-arguments)
  (returns function-schema-returns))

(define (function-schema-is-out-fn? value)
  (not (null? (arguments-out (function-schema-arguments value)))))

(define is-out-fn? function-schema-is-out-fn?)

(define (function-schema-has-symint? value)
  (any (lambda (argument)
         (type-is-symint-like? (argument-type argument)))
       (arguments-flat-non-out (function-schema-arguments value))))

(define (function-schema-kind value)
  (let ((arguments (function-schema-arguments value)))
    (cond
     ((function-schema-is-out-fn? value) schema-kind-out)
     ((and (arguments-self-arg arguments)
           (argument-is-write?
            (self-argument-value (arguments-self-arg arguments))))
      schema-kind-inplace)
     ((any argument-is-write?
           (append (arguments-pre-self-positional arguments)
                   (arguments-post-self-positional arguments)
                   (arguments-pre-tensor-options-kwarg-only arguments)
                   (if (arguments-tensor-options arguments)
                       (tensor-options-values
                        (arguments-tensor-options arguments))
                       '())
                   (arguments-post-tensor-options-kwarg-only arguments)))
      schema-kind-mutable)
     (else schema-kind-functional))))

(define (without-annotation argument strip-default?)
  (make-argument (argument-name argument)
                 (argument-type argument)
                 (and (not strip-default?) (argument-default argument))
                 #f
                 #f))

(define (without-return-annotation return keep-name?)
  (make-return (and keep-name? (return-name return))
               (return-type return)
               #f))

(define (signature-mutable-arguments value)
  (filter argument-is-write?
          (append (if (arguments-self-arg value)
                      (list (self-argument-value (arguments-self-arg value)))
                      '())
                  (arguments-post-self-positional value)
                  (arguments-out value))))

(define (same-annotation? left right)
  (and (annotation? left)
       (annotation? right)
       (equal? left right)))

(define (synthetic-returns arguments original-returns keep-name?)
  (filter-map
   (lambda (argument)
     (if (any (lambda (return)
               (same-annotation? (argument-annotation argument)
                                 (return-annotation return)))
             original-returns)
         #f
         (make-return
          (and keep-name?
               (string-append (argument-name argument) "_out"))
          (argument-type argument)
          #f)))
   (signature-mutable-arguments arguments)))

(define* (function-schema-signature value
                                    #:key
                                    (strip-default #f)
                                    (strip-view-copy-name #f)
                                    (keep-return-names #f))
  (let* ((arguments (function-schema-arguments value))
         (original-returns (function-schema-returns value))
         (normalized-returns
          (append
           (map (lambda (return)
                  (without-return-annotation return keep-return-names))
                original-returns)
           (synthetic-returns arguments original-returns keep-return-names)))
         (normalized-arguments
          (make-arguments
           (map (lambda (argument)
                  (without-annotation argument strip-default))
                (arguments-pre-self-positional arguments))
           (and (arguments-self-arg arguments)
                (make-self-argument
                 (without-annotation
                  (self-argument-value (arguments-self-arg arguments))
                  strip-default)))
           (map (lambda (argument)
                  (without-annotation argument strip-default))
                (arguments-post-self-positional arguments))
           (append
            (map (lambda (argument)
                   (without-annotation argument strip-default))
                 (arguments-pre-tensor-options-kwarg-only arguments))
            (map (lambda (argument)
                   (without-annotation argument strip-default))
                 (arguments-post-tensor-options-kwarg-only arguments)))
           #f
           '()
           '())))
    (make-function-schema
     (make-operator-name
      (make-base-operator-name
       (base-operator-name-base (operator-name-base
                                 (function-schema-name value)))
       #f
       (base-operator-name-dunder-method?
        (operator-name-base (function-schema-name value)))
       #f)
      "")
     normalized-arguments
     normalized-returns)))

(define signature function-schema-signature)

(define (type->string value)
  (case (type-kind value)
    ((tensor) "Tensor")
    ((scalar) "Scalar")
    ((any) "Any")
    ((number) "Number")
    ((int) "int")
    ((float) "float")
    ((bool) "bool")
    ((string) "str")
    ((sym-int) "SymInt")
    ((memory-format) "MemoryFormat")
    ((dimname) "Dimname")
    ((const) "const")
    ((base-type) (or (base-type-name value) "BaseType"))
    ((custom-class)
     (string-append "__torch__.torch.classes."
                    (car (type-arguments value))))
    ((optional)
     (string-append (type->string (car (type-arguments value))) "?"))
    ((list)
     (let ((size (list-type-size value)))
       (string-append (type->string (list-type-element value))
                      "[" (if size (number->string size) "") "]")))
    (else (error 'unknown-type-kind (type-kind value)))))

(define (argument->string value)
  (string-append
   (type->string (argument-type value))
   (if (argument-annotation value)
       (string-append "(" (annotation->string (argument-annotation value)) ")")
       "")
   " " (argument-name value)
   (if (argument-default value)
       (string-append "=" (argument-default value))
       "")))

(define (return->string value)
  (string-append
   (type->string (return-type value))
   (if (return-annotation value)
       (string-append "(" (annotation->string (return-annotation value)) ")")
       "")
   (if (return-name value)
       (string-append " " (return-name value))
       "")))

(define (function-schema->string value)
  (let* ((arguments (function-schema-arguments value))
         (positional (arguments-pre-self-positional arguments))
         (self (if (arguments-self-arg arguments)
                   (list (self-argument-value
                          (arguments-self-arg arguments)))
                   '()))
         (post (arguments-post-self-positional arguments))
         (kw-before (arguments-pre-tensor-options-kwarg-only arguments))
         (options (if (arguments-tensor-options arguments)
                      (tensor-options-values
                       (arguments-tensor-options arguments))
                      '()))
         (kw-after (arguments-post-tensor-options-kwarg-only arguments))
         (out (arguments-out arguments))
         (pos (append positional self post))
         (kw (append kw-before options kw-after))
         (all (append pos kw out))
         (argument-texts (append (map argument->string pos)
                                (if (and (null? kw) (null? out))
                                    '()
                                    (cons "*" (map argument->string kw)))
                                (map argument->string out)))
         (returns (function-schema-returns value)))
    (string-append
     (operator-name->string (function-schema-name value))
     "(" (string-join argument-texts ", ") ") -> "
     (cond ((null? returns) "")
           ((null? (cdr returns)) (return->string (car returns)))
           (else (string-append "(" (string-join (map return->string returns) ", ") ")"))))))

(define (function-schema-hash value)
  (hash value 31))
