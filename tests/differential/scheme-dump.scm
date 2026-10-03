#!/usr/bin/guile --no-auto-compile
!#

(use-modules (srfi srfi-1)
             (srfi srfi-13)
             (sonic-cross yaml)
             (sonic-cross core-ir)
             (sonic-cross native-function)
             (sonic-cross generated-functions))

(define (field name value)
  (format #t "~a: ~a~%" name (if value value "-")))

(define (bool value) (if value "true" "false"))

(define (sorted-tags function)
  (sort (native-function-tags function) string<?))

(define (dispatch-rows dispatch)
  (sort
   (apply append
          (map (lambda (entry)
                 (let ((key (car entry))
                       (values (cdr entry)))
                   (map (lambda (value)
                          (let ((metadata (cdr value)))
                            (list key
                                  (backend-metadata-kernel metadata)
                                  (backend-metadata-structured metadata)
                                  (backend-metadata-cpp-namespace metadata)
                                  (backend-metadata-supports-symint? metadata))))
                        values)))
               dispatch))
   (lambda (a b)
     (or (string<? (car a) (car b))
         (and (string=? (car a) (car b))
              (string<? (cadr a) (cadr b)))))))

(define (emit index function dispatch generated?)
  (let* ((schema (native-function-func function))
         (operator (function-schema-name schema))
         (base (operator-name-base operator))
         (tags (sorted-tags function)))
    (display "record-begin\n")
    (field "index" (format #f "~6,'0d" index))
    (field "namespace" (native-function-namespace function))
    (field "operator" (operator-name->string operator))
    (field "base" (base-operator-name-base base))
    (field "overload" (operator-name-overload-name operator))
    (field "schema" (function-schema->string schema))
    (field "schema-kind" (symbol->string (function-schema-kind schema)))
    (field "generated" (bool generated?))
    (field "tags" (if (null? tags) "-" (string-join tags ",")))
    (display "dispatch-begin\n")
    (for-each
     (lambda (row)
       (format #t "dispatch: ~a|~a|~a|~a|~a~%"
               (list-ref row 0) (list-ref row 1) (bool (list-ref row 2))
               (list-ref row 3) (bool (list-ref row 4))))
     (dispatch-rows dispatch))
    (display "dispatch-end\nrecord-end\n")))

(define (emit-view-group index group)
  (let ((view (native-functions-view-group-view group))
        (view-copy (native-functions-view-group-view-copy group))
        (view-inplace (native-functions-view-group-view-inplace group)))
    (display "view-group-begin\n")
    (field "index" (format #f "~6,'0d" index))
    (field "root" (native-functions-view-group-root-name group))
    (field "view" (operator-name->string
                   (function-schema-name (native-function-func view))))
    (field "view-schema-kind"
           (symbol->string (native-function-view-schema-kind view)))
    (field "view-copy"
           (if view-copy
               (operator-name->string
                (function-schema-name (native-function-func view-copy)))
               "-"))
    (field "view-inplace"
           (if view-inplace
               (operator-name->string
                (function-schema-name (native-function-func view-inplace)))
               "-"))
    (field "composite" (bool (native-functions-view-group-composite? group)))
    (display "view-group-end\n")))

(unless (= (length (command-line)) 2)
  (error "usage: scheme-dump.scm FIXTURE"))
(let* ((root (yaml-load (cadr (command-line))))
       (entries (if (yaml-sequence? root) (yaml-sequence-items root)
                    (error 'fixture-root-must-be-sequence)))
       (parsed
        (map (lambda (mapping)
               (call-with-values
                   (lambda () (native-function-from-yaml mapping))
                 (lambda (function dispatch) (cons function dispatch))))
             entries))
       (original (map car parsed))
       (original-dispatch (map cdr parsed)))
  (call-with-values
      (lambda () (add-generated-native-functions! original '()))
    (lambda (functions indices)
      (let ((original-count (length original)))
        (for-each
         (lambda (index function)
           (if (< index original-count)
               (emit index function (list-ref original-dispatch index) #f)
               (let* ((operator (operator-name->string
                                 (function-schema-name
                                  (native-function-func function))))
                      (entry (assoc "CompositeExplicitAutograd" indices))
                      (dispatch
                       (if entry
                           (list (cons "CompositeExplicitAutograd"
                                       (filter (lambda (item)
                                                 (string=? (car item) operator))
                                               (cdr entry))))
                           '())))
                 (emit index function dispatch #t))))
         (iota (length functions)) functions)
      (let ((groups (native-functions-view-groups functions)))
        (for-each
         (lambda (index group) (emit-view-group index group))
         (iota (length groups)) groups))))))
