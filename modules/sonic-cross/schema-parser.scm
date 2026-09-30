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

(define-module (sonic-cross schema-parser)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-13)
  #:use-module (sonic-cross core-ir)
  #:use-module (sonic-cross regex)
  #:export (parse-annotation
            parse-type
            parse-argument
            parse-return
            parse-returns
            arguments-preparse
            parse-arguments
            parse-function-schema
            function-schema-parse))

(define (fail who value)
  (error who value))

(define (split-exact text separator)
  (let ((separator-length (string-length separator)))
    (let loop ((start 0) (result '()))
      (let find ((index start))
        (cond
         ((> (+ index separator-length) (string-length text))
          (reverse (cons (substring text start) result)))
         ((string=? (substring text index (+ index separator-length))
                    separator)
          (loop (+ index separator-length)
                (cons (substring text start index) result)))
         (else (find (1+ index))))))))

(define (rsplit-space text)
  (let loop ((index (1- (string-length text))))
    (cond
     ((< index 0) (fail 'invalid-schema-token text))
     ((char=? (string-ref text index) #\space)
      (list (substring text 0 index)
            (substring text (1+ index))))
     (else (loop (1- index))))))

(define annotation-re
  (regex-compile "^([a-z])(\\|[a-z])*(!?)( -> (\\*|[a-z](\\|[a-z])*))?$"))

(define (parse-alias-set text)
  (map (lambda (alias) alias) (split-exact text "|")))

(define (parse-annotation text)
  (let ((match (regex-match annotation-re text)))
    (if (not match)
        (fail 'invalid-annotation text)
        (let* ((arrow (string-contains text " -> "))
               (before-text (if arrow (substring text 0 arrow) text))
               (after-text (and arrow
                                (substring text (+ arrow 4))))
               (write? (and (positive? (string-length before-text))
                            (char=? (string-ref before-text
                                                 (1- (string-length before-text)))
                                    #\!)))
               (aliases-text (if write?
                                 (substring before-text 0
                                            (1- (string-length before-text)))
                                 before-text))
               (before (parse-alias-set aliases-text))
               (after (and after-text
                           (if (string=? after-text "*")
                               '*
                               (parse-alias-set after-text)))))
          (when (and write? (> (length before) 1))
            (fail 'invalid-annotation text))
          (when (and (pair? after) (not (eq? after '*))
                     (> (length before) 1) (> (length after) 1))
            (fail 'invalid-annotation text))
          (let ((annotation (make-annotation before write? after)))
            (unless (string=? (annotation->string annotation) text)
              (fail 'annotation-round-trip text))
            annotation)))))

(define tensor-annotation-re
  (regex-compile "^Tensor\\((.+)\\)(.*)$"))

(define (parse-type-and-annotation text)
  (let ((match (regex-match tensor-annotation-re text)))
    (if match
        (let ((annotation-text (regex-capture match 1))
              (suffix (regex-capture match 2)))
          (unless (member suffix '("" "?" "[]"))
            (fail 'invalid-tensor-annotation-suffix text))
          (cons (parse-type (string-append "Tensor" suffix))
                (parse-annotation annotation-text)))
        (cons (parse-type text) #f))))

(define type-optional-re (regex-compile "^(.+)\\?$"))
(define type-list-re (regex-compile "^(.+)\\[([0-9]+)?\\]$"))
(define custom-type-re
  (regex-compile "^__torch__\\.torch\\.classes\\.([a-zA-Z0-9_.]+)$"))

(define base-types
  '("Generator" "ScalarType" "Tensor" "int" "DimVector" "float"
    "str" "bool" "Layout" "Device" "DeviceIndex" "Scalar"
    "MemoryFormat" "QScheme" "Storage" "Stream" "SymInt" "SymBool"
    "GraphModule"))

(define (base-type-for-name name)
  (unless (member name base-types)
    (fail 'invalid-type name))
  (cond ((string=? name "Tensor") (make-tensor-type))
        ((string=? name "Scalar") (make-scalar-type))
        ((string=? name "int") (make-int-type))
        ((string=? name "float") (make-float-type))
        ((string=? name "str") (make-string-type))
        ((string=? name "bool") (make-bool-type))
        ((string=? name "SymInt") (make-sym-int-type))
        ((string=? name "MemoryFormat") (make-memory-format-type))
        ((string=? name "Dimname") (make-dimname-type))
        (else (make-base-type name))))

(define (parse-type text)
  (let ((optional-match (regex-match type-optional-re text)))
    (if optional-match
        (let ((type (make-optional-type
                     (parse-type (regex-capture optional-match 1)))))
          (unless (string=? (type->string type) text)
            (fail 'type-round-trip text))
          type)
        (let ((list-match (regex-match type-list-re text)))
          (if list-match
              (let* ((element (parse-type (regex-capture list-match 1)))
                     (size-text (regex-capture list-match 2))
                     (size (and size-text (string->number size-text)))
                     (type (make-list-type element size)))
                (unless (string=? (type->string type) text)
                  (fail 'type-round-trip text))
                type)
              (let ((custom-match (regex-match custom-type-re text)))
                (if custom-match
                    (let ((type (make-type 'custom-class
                                           (list (regex-capture custom-match 1)))))
                      (unless (string=? (type->string type) text)
                        (fail 'type-round-trip text))
                      type)
                    (let ((type (base-type-for-name text)))
                      (unless (string=? (type->string type) text)
                        (fail 'type-round-trip text))
                      type))))))))

(define (parse-argument text)
  (unless (string-contains text " ")
    (fail 'invalid-argument text))
  (let* ((equals (string-count text #\=))
         (parts (if (positive? equals)
                    (begin
                      (unless (= equals 1) (fail 'invalid-default text))
                      (split-exact text "="))
                    (list text #f)))
         (declaration (car parts))
         (default (cadr parts))
         (type-name (rsplit-space declaration))
         (parsed (parse-type-and-annotation (car type-name)))
         (argument (make-argument (cadr type-name)
                                  (car parsed)
                                  default
                                  (cdr parsed)
                                  (and (cdr parsed)
                                       (annotation-is-write? (cdr parsed))))))
    (unless (string=? (argument->string argument) text)
      (fail 'argument-round-trip text))
    argument))

(define (parse-return text)
  (let* ((has-space (string-contains text " "))
         (type-name (if has-space (rsplit-space text) (list text #f)))
         (parsed (parse-type-and-annotation (car type-name)))
         (result (make-return (cadr type-name) (car parsed) (cdr parsed))))
    (unless (string=? (return->string result) text)
      (fail 'return-round-trip text))
    result))

(define (parse-returns text)
  (cond ((string-null? text) '())
        ((and (char=? (string-ref text 0) #\()
              (char=? (string-ref text (1- (string-length text))) #\)))
         (map parse-return
              (split-exact (substring text 1 (1- (string-length text))) ", ")))
        (else (list (parse-return text)))))

(define (arguments-preparse text)
  (let loop ((tokens (split-exact text ", "))
             (mode 'positional)
             (positional '())
             (kwarg-only '())
             (out '()))
    (if (null? tokens)
        (list (reverse positional) (reverse kwarg-only) (reverse out))
        (let ((token (car tokens)))
          (cond
           ((string-null? token)
            (loop (cdr tokens) mode positional kwarg-only out))
           ((string=? token "*")
            (when (eq? mode 'kwarg-only) (fail 'duplicate-star text))
            (when (eq? mode 'out) (fail 'duplicate-star text))
            (loop (cdr tokens) 'kwarg-only positional kwarg-only out))
           (else
            (let* ((argument (parse-argument token))
                   (mutable? (and (argument-annotation argument)
                                  (annotation-is-write?
                                   (argument-annotation argument)))))
              (cond
               ((eq? mode 'positional)
                (loop (cdr tokens) mode (cons argument positional)
                      kwarg-only out))
               ((eq? mode 'kwarg-only)
                (if mutable?
                    (loop (cdr tokens) 'out positional kwarg-only
                          (cons argument out))
                    (loop (cdr tokens) mode positional
                          (cons argument kwarg-only) out)))
               (else
                (unless mutable? (fail 'argument-after-out token))
                (loop (cdr tokens) mode positional kwarg-only
                      (cons argument out)))))))))))

(define (matches-type? actual expected)
  (or (equal? actual expected)
      (equal? actual (make-optional-type expected))))

(define (tensor-options-window? values)
  (and (= (length values) 4)
       (string=? (argument-name (list-ref values 0)) "dtype")
       (matches-type? (argument-type (list-ref values 0))
                      (make-base-type "ScalarType"))
       (string=? (argument-name (list-ref values 1)) "layout")
       (matches-type? (argument-type (list-ref values 1))
                      (make-base-type "Layout"))
       (string=? (argument-name (list-ref values 2)) "device")
       (matches-type? (argument-type (list-ref values 2))
                      (make-base-type "Device"))
       (string=? (argument-name (list-ref values 3)) "pin_memory")
       (matches-type? (argument-type (list-ref values 3))
                      (make-bool-type))))

(define (parse-arguments text)
  (let* ((preparsed (arguments-preparse text))
         (positional (car preparsed))
         (kwarg-only (cadr preparsed))
         (out (caddr preparsed))
         (self-index
          (let loop ((values positional) (index 0))
            (cond ((null? values) #f)
                  ((string=? (argument-name (car values)) "self") index)
                  (else (loop (cdr values) (1+ index))))))
         (pre-self (if self-index (take positional self-index) '()))
         (self (and self-index
                    (make-self-argument (list-ref positional self-index))))
         (post-self (if self-index
                        (drop positional (1+ self-index))
                        positional)))
    (when (any (lambda (argument)
                 (and (argument-annotation argument)
                      (annotation-is-write?
                       (argument-annotation argument))))
               pre-self)
      (fail 'mutable-pre-self text))
    (let scan ((remaining kwarg-only) (before '()) (options #f) (after '()))
      (cond
       ((null? remaining)
        (let ((arguments (make-arguments pre-self self post-self
                                         (reverse before) options
                                         (reverse after) out)))
          (unless self
            (unless (null? pre-self) (fail 'self-invariant text)))
          (unless options
            (unless (null? after) (fail 'tensor-options-invariant text)))
          arguments))
       ((and (not options) (>= (length remaining) 4)
             (tensor-options-window? (take remaining 4)))
        (scan (drop remaining 4) before
              (apply make-tensor-options-arguments (take remaining 4)) after))
       ((and options (>= (length remaining) 4)
             (tensor-options-window? (take remaining 4)))
        (fail 'duplicate-tensor-options text))
       (options (scan (cdr remaining) before options
                      (cons (car remaining) after)))
       (else (scan (cdr remaining) (cons (car remaining) before)
                   options after))))))

(define function-schema-re
  (regex-compile "([^\\(]+)\\((.*)\\) -> (.*)"))

(define (parse-function-schema text)
  (let ((match (regex-match function-schema-re text)))
    (unless match (fail 'malformed-function-schema text))
    (let* ((name (regex-capture match 1))
           (args (regex-capture match 2))
           (returns (regex-capture match 3))
           (schema (make-function-schema
                    (parse-operator-name name)
                    (parse-arguments args)
                    (parse-returns returns))))
      (unless (string=? (function-schema->string schema) text)
        (fail 'function-schema-round-trip text))
      schema)))

(define function-schema-parse parse-function-schema)
