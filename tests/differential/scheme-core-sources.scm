#!/usr/bin/guile --no-auto-compile
!#
;; Emit SonicCross's G16-G20 core-source / operators artifacts from the frozen
;; Semantic IR into OUTDIR.  These are the exact C++ texts, byte-compared
;; against the frozen torchgen oracle:
;;
;;   G16  Operators.h, OperatorsEverything.cpp, Operators_0.._4.cpp
;;   G17  TensorBody.h
;;   G18  aten_interned_strings.h, enum_tag.h
;;   G19  TensorMethods.cpp, ATenOpList.cpp
;;   G20  RegisterBackendSelect.cpp

(use-modules (srfi srfi-1)
             (sonic-cross yaml)
             (sonic-cross core-ir)
             (sonic-cross native-function)
             (sonic-cross generated-functions)
             (sonic-cross operators)
             (sonic-cross tensor-body)
             (sonic-cross core-sources)
             (sonic-cross backend-select))

(define (write-file outdir name content)
  (call-with-output-file (string-append outdir "/" name)
    (lambda (port) (display content port))))

(unless (= (length (command-line)) 3)
  (error "usage: scheme-core-sources.scm NATIVE_FUNCTIONS_YAML OUTDIR"))
(let* ((native-yaml (cadr (command-line)))
       (outdir (caddr (command-line)))
       (root (yaml-load native-yaml))
       (entries (if (yaml-sequence? root) (yaml-sequence-items root)
                    (error 'fixture-root-must-be-sequence)))
       (original
        (map (lambda (mapping)
               (call-with-values
                   (lambda () (native-function-from-yaml mapping))
                 (lambda (function dispatch) function)))
             entries))
       (functions
        (call-with-values
            (lambda () (add-generated-native-functions! original '()))
          (lambda (functions indices) functions))))
  ;; G16 Operators.h + Operators*.cpp (sharded x5).
  (write-file outdir "Operators.h" (render-operators-h functions))
  (for-each (lambda (entry)
              (write-file outdir (car entry) (cdr entry)))
            (render-operators-cpp-files functions))
  ;; G17 TensorBody.h
  (write-file outdir "TensorBody.h" (render-tensor-body-h functions))
  ;; G18 aten_interned_strings.h + enum_tag.h
  (write-file outdir "aten_interned_strings.h"
              (render-aten-interned-strings-h functions))
  (write-file outdir "enum_tag.h" (render-enum-tag-h))
  ;; G19 TensorMethods.cpp + ATenOpList.cpp
  (write-file outdir "TensorMethods.cpp" (render-tensor-methods-cpp))
  (write-file outdir "ATenOpList.cpp" (render-aten-op-list-cpp functions))
  ;; G20 RegisterBackendSelect.cpp
  (write-file outdir "RegisterBackendSelect.cpp"
              (render-backend-select-cpp functions)))
