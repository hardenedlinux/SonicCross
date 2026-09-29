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
   make-optional-type
   make-list-type
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
   type-hash
   make-argument
   argument?
   argument-name
   argument-type
   argument-default
   argument-annotation
   argument-is-write?
   argument-hash
   make-return
   return?
   return-name
   return-type
   return-annotation
   return-hash
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
        (if (string-prefix? "i" dunder)
            (if (augmented-assignment-name? (substring dunder 1))
                (make-base-operator-name
                 (substring dunder 1) #t #t #f)
                (error 'invalid-dunder-name base))
            (make-base-operator-name dunder #f #t #f))
        (let ((inplace (and (positive? (string-length base))
                            (char=? (string-ref base
                                                 (1- (string-length base)))
                                    #\_))))
          (make-base-operator-name
           (if inplace
               (substring base 0 (1- (string-length base)))
               base)
           inplace #f #f)))))

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
                           base-name)))
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
(define (make-base-type) base-type)

(define (make-optional-type element-type)
  (make-type 'optional (list element-type)))

(define (make-list-type element-type)
  (make-type 'list (list element-type)))

(define (type-is-tensor-like? value)
  (eq? (type-kind value) 'tensor))

(define tensor-like? type-is-tensor-like?)

(define (has-symint? value)
  (or (eq? (type-kind value) 'sym-int)
      (any has-symint? (type-arguments value))))

(define type-has-symint? has-symint?)

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
              (list (arguments-self-arg value))
              '())
          (arguments-post-self-positional value)
          (arguments-pre-tensor-options-kwarg-only value)
          (if (arguments-tensor-options value)
              (list (arguments-tensor-options value))
              '())
          (arguments-post-tensor-options-kwarg-only value)
          (arguments-out value)))

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

(define (function-schema-kind value)
  (let ((arguments (function-schema-arguments value)))
    (cond
     ((function-schema-is-out-fn? value) schema-kind-out)
     ((and (arguments-self-arg arguments)
           (argument-is-write? (arguments-self-arg arguments)))
      schema-kind-inplace)
     ((any argument-is-write?
           (append (arguments-pre-self-positional arguments)
                   (arguments-post-self-positional arguments)
                   (arguments-pre-tensor-options-kwarg-only arguments)
                   (if (arguments-tensor-options arguments)
                       (list (arguments-tensor-options arguments))
                       '())
                   (arguments-post-tensor-options-kwarg-only arguments)))
      schema-kind-mutable)
     (else schema-kind-functional))))

(define (without-annotation argument)
  (make-argument (argument-name argument)
                 (argument-type argument)
                 (argument-default argument)
                 #f
                 (argument-is-write? argument)))

(define (without-return-annotation return keep-name?)
  (make-return (and keep-name? (return-name return))
               (return-type return)
               #f))

(define (return-represents-argument? return argument)
  (and (return-name return)
       (string=? (return-name return) (argument-name argument))))

(define (writable-arguments value)
  (filter argument-is-write?
          (append (arguments-pre-self-positional value)
                  (if (arguments-self-arg value)
                      (list (arguments-self-arg value))
                      '())
                  (arguments-post-self-positional value)
                  (arguments-pre-tensor-options-kwarg-only value)
                  (if (arguments-tensor-options value)
                      (list (arguments-tensor-options value))
                      '()))))

(define (synthetic-returns arguments original-returns keep-name?)
  (filter-map
   (lambda (argument)
     (if (any (lambda (return)
               (return-represents-argument? return argument))
             original-returns)
         #f
         (make-return
          (and keep-name?
               (string-append (argument-name argument) "_out"))
          (argument-type argument)
          #f)))
   (writable-arguments arguments)))

(define* (function-schema-signature value #:key (keep-return-names #f))
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
           (map without-annotation
                (arguments-pre-self-positional arguments))
           (and (arguments-self-arg arguments)
                (without-annotation (arguments-self-arg arguments)))
           (map without-annotation
                (arguments-post-self-positional arguments))
           (append
            (map without-annotation
                 (arguments-pre-tensor-options-kwarg-only arguments))
            (map without-annotation
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

(define (function-schema-hash value)
  (hash value 31))
