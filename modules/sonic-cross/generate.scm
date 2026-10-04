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

;; The production generation driver: the one entry point that turns
;; native_functions.yaml into the full set of C++ artifacts written to disk,
;; byte-identical to frozen torchgen `python3 torchgen/gen.py`.  This is the
;; "guild torchgen" backend — no Python is involved anywhere in this path: the
;; YAML is read through the libyaml FFI, tags come from the frozen `valid-tags`
;; constant in (sonic-cross native-function), and every artifact is rendered by
;; the existing G1-G20 emitter modules.
;;
;; The per-group render calls and the index-seeding helpers are the same ones
;; exercised by tests/differential/scheme-*.scm (byte-verified against the
;; frozen oracle).  The install-dir subdirectory layout mirrors frozen
;; torchgen/gen.py: cpu/cpu_vec/cuda file managers all write into <install-dir>,
;; core writes into <install-dir>/core, and headeronly defaults to
;; <install-dir>/core.

(define-module (sonic-cross generate)
  #:use-module (ice-9 optargs)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-13)
  #:use-module (sonic-cross yaml)
  #:use-module (sonic-cross core-ir)
  #:use-module (sonic-cross native-function)
  #:use-module (sonic-cross generated-functions)
  #:use-module (sonic-cross orchestration)
  #:use-module (sonic-cross emitter)
  #:use-module (sonic-cross register-dispatch-key)
  #:use-module (sonic-cross per-operator-headers)
  #:use-module (sonic-cross functionalization)
  #:use-module (sonic-cross operators)
  #:use-module (sonic-cross tensor-body)
  #:use-module (sonic-cross core-sources)
  #:use-module (sonic-cross backend-select)
  #:use-module (sonic-cross ufunc)
  #:use-module (sonic-cross composite-view-copy-kernels)
  #:export (load-corpus
            index-entry
            index-add
            merge-dispatch
            backend-index-for
            generate-all))

;; ---------------------------------------------------------------------------
;; Backend-index seeding helpers.  Verbatim from the differential scripts: the
;; YAML per-entry `dispatch` field is merged into an association list of
;; (key (operator . backend-metadata) ...), which add-generated-native-functions!
;; then augments with the generated out= / functional / inplace variants.
;; ---------------------------------------------------------------------------

(define (index-entry indices key)
  (find (lambda (entry) (string=? (car entry) key)) indices))

(define (index-add indices key operator metadata)
  (let ((entry (index-entry indices key)))
    (if entry
        (cons (cons key
                    (cons (cons operator metadata)
                          (filter (lambda (item)
                                    (not (string=? (car item) operator)))
                                  (cdr entry))))
              (filter (lambda (item) (not (string=? (car item) key))) indices))
        (cons (cons key (list (cons operator metadata))) indices))))

(define (merge-dispatch indices dispatch)
  ;; dispatch :: ((key (op . backend-metadata) ...) ...)
  (fold (lambda (key-entry acc)
          (let ((key (car key-entry)))
            (fold (lambda (op-meta inner)
                    (index-add inner key (car op-meta) (cdr op-meta)))
                  acc
                  (cdr key-entry))))
        indices
        dispatch))

(define (backend-index-for indices key)
  ;; The lightweight BackendIndex the emitter modules consume for a single
  ;; dispatch key: (effective-key kernels device-guard?).  device-guard? is
  ;; always #f here — this matches the byte-verified differential scripts (the
  ;; emitters that take a backend-index never consult device-guard for these
  ;; render paths).
  (list key (or (assoc-ref indices key) '()) #f))

;; ---------------------------------------------------------------------------
;; Corpus loading
;; ---------------------------------------------------------------------------

(define (load-corpus source-path)
  ;; Load <source-path>/native/native_functions.yaml, parse each entry, seed the
  ;; backend indices from the per-entry dispatch block, then run the generation
  ;; closure.  Returns (values functions indices).
  (let* ((yaml-path (string-append source-path "/native/native_functions.yaml"))
         (root (yaml-load yaml-path))
         (entries (if (yaml-sequence? root)
                      (yaml-sequence-items root)
                      (error 'native-functions-yaml-must-be-sequence yaml-path)))
         (parsed
          (map (lambda (mapping)
                 (call-with-values
                     (lambda () (native-function-from-yaml mapping))
                   (lambda (function dispatch) (cons function dispatch))))
               entries))
         (original (map car parsed))
         (original-dispatch (map cdr parsed))
         (seeded (fold merge-dispatch '() original-dispatch)))
    (add-generated-native-functions! original seeded)))

;; ---------------------------------------------------------------------------
;; File-system helpers
;; ---------------------------------------------------------------------------

(define (ensure-directory! dir)
  ;; mkdir -p equivalent, for both absolute and relative paths.
  (unless (file-exists? dir)
    (let* ((absolute? (string-prefix? "/" dir))
           (parts (remove string-null? (string-split dir #\/)))
           (prefix (if absolute? "/" "")))
      (let loop ((cur prefix) (rest parts))
        (unless (null? rest)
          (let ((next (if (string-null? cur)
                          (car rest)
                          (string-append cur "/" (car rest)))))
            (unless (or (string-null? next) (file-exists? next))
              (mkdir next))
            (loop next (cdr rest))))))))

;; ---------------------------------------------------------------------------
;; File-manager label recovery (for --output-dependencies)
;; ---------------------------------------------------------------------------

;; torchgen splits the artifacts across file managers cpu / cpu_vec / core /
;; ops / cuda / aoti / headeronly.  The install-dir grouping alone cannot
;; recover cpu_vec (UfuncCPUKernel_*.cpp, same dir as cpu) or the cpu-vs-cuda
;; split of Register{key}.cpp and {key}Functions.h.  These helpers recover the
;; label from the filename using the frozen dispatch-key sets.

(define (register-dispatch-key->label filename)
  ;; filename = "Register<key><suffix>.cpp" where suffix is "Everything" or
  ;; "_0".."_n".  Classify by dispatch key: the CUDA-family keys land in the
  ;; cuda file manager, everything else in cpu.  (The old suffix-stripping
  ;; substring approach broke once "Everything" shards got their real name —
  ;; "RegisterCUDA_0.cpp" was misread as key "CUDA_0".)
  (if (find (lambda (key)
              (string-prefix? (string-append "Register" key) filename))
            cuda-dispatch-keys)
      'cuda 'cpu))

(define (dispatch-key-functions->label filename)
  ;; filename = "<key>Functions.h" or "<key>Functions_inl.h"
  (let ((idx (string-contains filename "Functions")))
    (if (member (substring filename 0 idx) cuda-dispatch-keys) 'cuda 'cpu)))

;; ---------------------------------------------------------------------------
;; The driver
;; ---------------------------------------------------------------------------

(define* (generate-all source-path install-dir
                       #:key
                       (aoti-install-dir #f)
                       (headeronly-install-dir #f)
                       (dry-run? #f)
                       (per-operator-headers? #f)
                       (generate '(headers sources declarations_yaml)))
  ;; Render the full in-scope G1-G20 closure and write it under install-dir.
  ;; `generate` mirrors torchgen's --generate subset: sources / headers /
  ;; declarations_yaml.  Out-of-scope artifacts (AOTI c_shim_*.cpp,
  ;; VmapGeneratedPlumbing.h, Declarations.yaml) are never emitted.
  ;;
  ;; Returns a list of (<label> <dir> <filename>) triples, in generation order,
  ;; where <label> is one of cpu | cpu-vec | core | cuda | headeronly.
  (let* ((want-sources (memq 'sources generate))
         (want-headers (memq 'headers generate))
         (core-dir (string-append install-dir "/core"))
         (headeronly-dir (or headeronly-install-dir core-dir))
         (written '()))
    (define (write! label dir filename content)
      (set! written (cons (list label dir filename) written))
      (unless dry-run?
        (ensure-directory! dir)
        (call-with-output-file (string-append dir "/" filename)
          (lambda (port) (display content port)))))
    (call-with-values (lambda () (load-corpus source-path))
      (lambda (functions indices)
        (let* ((grouped (get-grouped-native-functions functions))
               (structured (get-structured-native-functions grouped))
               (view-groups (native-functions-view-groups functions))
               (all-groups (get-all-groups functions structured view-groups)))
          ;; ---- gen_source_files ----
          (when want-sources
            ;; G2 Register{key}.cpp (sharded); cpu vs cuda by dispatch key.
            (for-each
             (lambda (f)
               (write! (register-dispatch-key->label (car f))
                       install-dir (car f) (cdr f)))
             (render-register-dispatch-key-files grouped indices
                                                 #:per-operator?
                                                 per-operator-headers?))
            ;; G10 structured ufunc kernels.
            (for-each
             (lambda (g)
               (let ((name (base-operator-spelling
                            (native-function-func
                             (native-functions-group-functional g)))))
                 (write! 'cpu install-dir (string-append "UfuncCPU_" name ".cpp")
                         (render-ufunc-cpu-cpp
                          g (backend-index-for indices "CPU")))
                 (write! 'cpu-vec install-dir
                         (string-append "UfuncCPUKernel_" name ".cpp")
                         (render-ufunc-cpu-kernel-cpp g))
                 (write! 'cuda install-dir (string-append "UfuncCUDA_" name ".cu")
                         (render-ufunc-cuda-cu
                          g (backend-index-for indices "CUDA")))))
             (filter (lambda (g)
                       (native-function-ufunc-inner-loop
                        (native-functions-group-out g)))
                     (filter native-functions-group? grouped)))
            ;; G20 RegisterBackendSelect.cpp.
            (write! 'cpu install-dir "RegisterBackendSelect.cpp"
                    (render-backend-select-cpp functions))
            ;; G1 RegisterSchema.cpp.
            (write! 'cpu install-dir "RegisterSchema.cpp"
                    (render-register-schema-cpp functions))
            ;; G16 Operators*.cpp (sharded x5).
            (for-each (lambda (f) (write! 'cpu install-dir (car f) (cdr f)))
                      (render-operators-cpp-files functions))
            ;; G5 Functions.cpp (verbatim template).
            (write! 'cpu install-dir "Functions.cpp" (render-functions-cpp))
            ;; G19 TensorMethods.cpp + ATenOpList.cpp (core).
            (write! 'core core-dir "TensorMethods.cpp" (render-tensor-methods-cpp))
            (write! 'core core-dir "ATenOpList.cpp"
                    (render-aten-op-list-cpp functions))
            ;; G13 RegisterFunctionalization*.cpp (sharded x4).
            (for-each
             (lambda (shard)
               (let ((suffix (car shard)))
                 (write! 'cpu install-dir
                         (string-append "RegisterFunctionalization"
                                        suffix ".cpp")
                         (cdr shard))))
             (render-register-functionalization-files
              all-groups (backend-index-for indices "CompositeImplicitAutograd")))
            ;; G14/G15 + G11/G12 functionalization / view / composite kernels.
            (write! 'cpu install-dir "FunctionalInverses.h"
                    (render-functional-inverses view-groups))
            (write! 'cpu install-dir "ViewMetaClasses.h"
                    (render-view-meta-classes-h view-groups))
            (write! 'cpu install-dir "ViewMetaClasses.cpp"
                    (render-view-meta-classes-cpp view-groups))
            (write! 'cpu install-dir "CompositeViewCopyKernels.cpp"
                    (render-composite-view-copy-kernels-cpp
                     structured view-groups
                     (backend-index-for
                      indices "CompositeExplicitAutogradNonFunctional"))))
          ;; ---- gen_headers ----
          (when want-headers
            (if per-operator-headers?
                ;; Per-operator mode: ops/*.h + include-only aggregate shims +
                ;; per-dispatch-key {key}Functions*.h.
                (for-each
                 (lambda (entry)
                   (let ((label (car entry))
                         (filename (cadr entry))
                         (content (caddr entry)))
                     (write! label
                             (if (eq? label 'ops)
                                 (string-append install-dir "/ops")
                                 install-dir)
                             filename content)))
                 (render-per-operator-headers functions grouped indices))
                (begin
                  ;; G9 NativeMetaFunctions.h.
                  (write! 'cpu install-dir "NativeMetaFunctions.h"
                          (render-native-meta-functions-h structured))
                  ;; G4 MethodOperators.h.
                  (write! 'cpu install-dir "MethodOperators.h"
                          (render-method-operators-h functions))
                  ;; G16 Operators.h.
                  (write! 'cpu install-dir "Operators.h" (render-operators-h functions))
                  ;; G5 Functions.h.
                  (write! 'cpu install-dir "Functions.h" (render-functions-h functions))
                  ;; G7 NativeFunctions.h.
                  (write! 'cpu install-dir "NativeFunctions.h"
                          (render-native-functions-h grouped indices))
                  ;; G8 {key}Functions.h / {key}Functions_inl.h; cpu vs cuda by key.
                  (for-each
                   (lambda (f)
                     (write! (dispatch-key-functions->label (car f))
                             install-dir (car f) (cdr f)))
                   (render-dispatch-key-functions grouped indices))))
            ;; G17 TensorBody.h (core).
            (write! 'core core-dir "TensorBody.h" (render-tensor-body-h functions))
            ;; G6 RedispatchFunctions.h.
            (write! 'cpu install-dir "RedispatchFunctions.h"
                    (render-redispatch-functions-h functions))
            ;; G3 RegistrationDeclarations.h.
            (write! 'cpu install-dir "RegistrationDeclarations.h"
                    (render-registration-declarations-h functions))
            ;; G18 aten_interned_strings.h (core) + enum_tag.h (headeronly).
            (write! 'core core-dir "aten_interned_strings.h"
                    (render-aten-interned-strings-h functions))
            (write! 'headeronly headeronly-dir "enum_tag.h" (render-enum-tag-h)))
          ;; ---- gen_declarations_yaml: out of scope (G21) ----
          #t)))
    (reverse written)))
