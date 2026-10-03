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

;; Mirrors the frozen `gen.py main()` orchestration (release/2.14 @
;; 41ffbc4a994e058af9fe00ed5caba73fc1033359) with the default flag set (no
;; --mps/--xpu/--mtia/--per-operator-headers/--rocm).  This module recovers the
;; *observable* artifact-output semantics only:
;;
;;   - the dispatch-key filter (MPS/XPU/MTIA removed),
;;   - the grouping (get_grouped_native_functions) and structured/view splits,
;;   - the write call graph: which FileManager label, which filename, and for
;;     sharded writes which items + num_shards,
;;   - the sharded membership (string_stable_hash(root_name) % num_shards).
;;
;; No template content is rendered; this produces the Tier 0 artifact manifest
;; that a later emitter phase will turn into C++.

(define-module (sonic-cross orchestration)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-13)
  #:use-module (sonic-cross core-ir)
  #:use-module (sonic-cross native-function)
  #:use-module (sonic-cross generated-functions)
  #:use-module (sonic-cross file-manager)
  #:export (native-function-root-name
            item-root-name
            filtered-dispatch-keys
            functions-keys
            cuda-dispatch-keys
            is-cuda-dispatch-key?
            get-grouped-native-functions
            get-structured-native-functions
            get-all-groups
            generate-artifact-manifest))

;; ---------------------------------------------------------------------------
;; root-name recovery
;; ---------------------------------------------------------------------------

(define (native-function-root-name nf)
  ;; Mirrors NativeFunction.root_name == func.name.name.base.
  (base-operator-name-base
   (operator-name-base (function-schema-name (native-function-func nf)))))

(define (item-root-name item)
  ;; Mirrors the shared `x.root_name` key_fn used by the sharded writes.
  (cond
   ((native-functions-group? item) (native-functions-group-root-name item))
   ((native-functions-view-group? item)
    (native-functions-view-group-root-name item))
   ((native-function? item) (native-function-root-name item))
   (else (error 'unknown-grouped-item item))))

;; ---------------------------------------------------------------------------
;; dispatch-key recovery (default flags)
;; ---------------------------------------------------------------------------

(define mps-dispatch-keys '("MPS" "SparseMPS" "SparseCsrMPS"))
(define xpu-dispatch-keys
  '("XPU" "SparseXPU" "SparseCsrXPU" "NestedTensorXPU"))

(define (filtered-dispatch-keys)
  ;; gen.main(): MPS_KEYS / XPU_KEYS dropped, then MTIA deleted.
  (filter (lambda (key)
            (and (not (member key mps-dispatch-keys))
                 (not (member key xpu-dispatch-keys))
                 (not (string=? key "MTIA"))))
          dispatch-keys))

;; gen.main(): the set of dispatch keys that receive `{Key}Functions.h` headers.
(define functions-keys
  '("CPU" "CUDA" "CompositeImplicitAutograd"
    "CompositeImplicitAutogradNestedTensor"
    "CompositeExplicitAutograd" "CompositeExplicitAutogradNonFunctional"
    "Meta"))

(define cuda-dispatch-keys
  '("CUDA" "QuantizedCUDA" "SparseCUDA" "SparseCsrCUDA" "NestedTensorCUDA"))

(define (is-cuda-dispatch-key? key)
  (member key cuda-dispatch-keys))

(define ufunc-dispatch-keys '("CUDA" "CPU"))

;; ---------------------------------------------------------------------------
;; grouping recovery
;; ---------------------------------------------------------------------------

;; Faithful mirror of torchgen NativeFunctionsGroup.from_dict: no autogen /
;; namespace / signature consistency checks, which the existing
;; native-functions-group-from-dict adds (and which fail on the real corpus).
(define (faithful-group-from-pairs pairs)
  ;; pairs :: ((<schema-kind> . <native-function>) ...)
  (let* ((functional (assq-ref pairs schema-kind-functional))
         (inplace (assq-ref pairs schema-kind-inplace))
         (mutable (assq-ref pairs schema-kind-mutable))
         (out (assq-ref pairs schema-kind-out)))
    (cond
     ((null? pairs) (error 'from-dict-empty))
     ((= (length pairs) 1) #f)
     ((not functional) (error 'group-missing-functional))
     ((not out) #f)
     (else (make-native-functions-group functional inplace mutable out)))))

(define (flatten-pre-group entry)
  ;; entry :: (<signature> (<kind> . <native-function>) ...)
  (let* ((pairs (cdr entry))
         (group (faithful-group-from-pairs pairs)))
    (if group
        (list group)
        (begin
          (when (any (lambda (p)
                       (member "generated" (native-function-tags (cdr p))))
                     pairs)
            (error 'ungrouped-generated-function
                   (map (lambda (p)
                          (operator-name->string
                           (function-schema-name
                            (native-function-func (cdr p)))))
                        pairs)))
          (map cdr pairs)))))

(define (get-grouped-native-functions native-functions)
  ;; Mirrors get_grouped_native_functions: pre-group then flatten.
  (append-map flatten-pre-group
              (pre-group-native-functions native-functions)))

(define (get-structured-native-functions grouped)
  (filter native-functions-group? grouped))

;; ---------------------------------------------------------------------------
;; all-groups (RegisterFunctionalization.cpp)
;; ---------------------------------------------------------------------------

(define (get-all-groups native-functions structured view-groups)
  ;; Mirrors the all_groups assembly: structured groups, then view groups, then
  ;; every native function whose full operator name appears in neither map.
  (let* ((structured-functions
          (append-map native-functions-group-functions structured))
         (view-functions
          (append-map native-functions-view-group-functions view-groups))
         (structured-names
          (map (lambda (f)
                 (operator-name->string
                  (function-schema-name (native-function-func f))))
               structured-functions))
         (view-names
          (map (lambda (f)
                 (operator-name->string
                  (function-schema-name (native-function-func f))))
               view-functions))
         (remaining
          (filter (lambda (f)
                    (let ((name
                           (operator-name->string
                            (function-schema-name (native-function-func f)))))
                      (and (not (member name structured-names))
                           (not (member name view-names)))))
                  native-functions)))
    (append structured view-groups remaining)))

;; ---------------------------------------------------------------------------
;; base spelling (for the ufunc filenames, mirrors str(BaseOperatorName))
;; ---------------------------------------------------------------------------

(define (base-operator-spelling func)
  (let* ((base-name (operator-name-base (function-schema-name func)))
         (base (base-operator-name-base base-name)))
    (cond
     ((base-operator-name-dunder-method? base-name)
      (string-append "__"
                     (if (base-operator-name-inplace? base-name)
                         (string-append "i" base)
                         base)
                     "__"))
     ((base-operator-name-inplace? base-name) (string-append base "_"))
     ((base-operator-name-functional-overload? base-name)
      (string-append base "_functional"))
     (else base))))

;; ---------------------------------------------------------------------------
;; orchestration
;; ---------------------------------------------------------------------------

(define (make-fms)
  ;; Creation order fixes the labels: core, cpu, cpu_vec, cuda, ops, aoti,
  ;; headeronly.  install dirs are irrelevant to the manifest; only the label
  ;; matters.
  (list (make-file-manager "core" "core")
        (make-file-manager "cpu" "cpu")
        (make-file-manager "cpu_vec" "cpu_vec")
        (make-file-manager "cuda" "cuda")
        (make-file-manager "ops" "ops")
        (make-file-manager "aoti" "aoti")
        (make-file-manager "headeronly" "headeronly")))

(define (fm-ref fms label)
  (find (lambda (fm) (string=? (file-manager-label fm) label)) fms))

(define (file-manager-for-key key cpu-fm cuda-fm)
  (if (is-cuda-dispatch-key? key) cuda-fm cpu-fm))

(define (generate-artifact-manifest native-functions)
  ;; native-functions: the full corpus (already augmented with generated out=
  ;; and functional variants).  Returns the sorted manifest: a list of
  ;; (<label> <filename> <ops>) where <ops> is "-" or a space-joined sorted
  ;; multiset of root names.
  (let* ((dispatch-keys (filtered-dispatch-keys))
         (grouped (get-grouped-native-functions native-functions))
         (structured (get-structured-native-functions grouped))
         (view-groups (native-functions-view-groups native-functions))
         (all-groups (get-all-groups native-functions structured view-groups))
         (fms (make-fms))
         (core-fm (fm-ref fms "core"))
         (cpu-fm (fm-ref fms "cpu"))
         (cpu-vec-fm (fm-ref fms "cpu_vec"))
         (cuda-fm (fm-ref fms "cuda"))
         (aoti-fm (fm-ref fms "aoti"))
         (headeronly-fm (fm-ref fms "headeronly")))

    ;; ---- gen_source_files ----

    ;; Register{key}.cpp (sharded) + ufunc files.
    (for-each
     (lambda (key)
       (let ((fm (file-manager-for-key key cpu-fm cuda-fm)))
         (file-manager-write-sharded!
          fm (string-append "Register" key ".cpp")
          (if (string=? key "CPU") 4 1)
          grouped
          item-root-name)
         ;; ufunc inner loop: only structured ops with a ufunc_inner_loop block
         ;; on CPU/CUDA produce Ufunc* files.
         (for-each
          (lambda (g)
            (when (and (native-function-ufunc-inner-loop
                        (native-functions-group-out g))
                       (member key ufunc-dispatch-keys))
              (let ((name
                     (base-operator-spelling
                      (native-function-func
                       (native-functions-group-functional g)))))
                (cond
                 ((string=? key "CPU")
                  (file-manager-write!
                   fm (string-append "UfuncCPU_" name ".cpp"))
                  (file-manager-write!
                   cpu-vec-fm (string-append "UfuncCPUKernel_" name ".cpp")))
                 ((string=? key "CUDA")
                  (file-manager-write!
                   fm (string-append "UfuncCUDA_" name ".cu")))))))
          structured)))
     dispatch-keys)

    ;; gen_aoti_c_shim_files (default aoti_backends = {CPU, CUDA, None}).
    (for-each
     (lambda (name)
       (file-manager-write! aoti-fm (string-append "c_shim_" name ".cpp")))
     '("aten" "cpu" "cuda"))

    ;; BackendSelect / Schema (unsharded).
    (file-manager-write! cpu-fm "RegisterBackendSelect.cpp")
    (file-manager-write! cpu-fm "RegisterSchema.cpp")

    ;; Operators.cpp (sharded over the full corpus).
    (file-manager-write-sharded!
     cpu-fm "Operators.cpp" 5 native-functions native-function-root-name)

    (file-manager-write! cpu-fm "Functions.cpp")
    (file-manager-write! core-fm "TensorMethods.cpp")
    (file-manager-write! core-fm "ATenOpList.cpp")

    ;; RegisterFunctionalization.cpp (sharded over all_groups).
    (file-manager-write-sharded!
     cpu-fm "RegisterFunctionalization.cpp" 4 all-groups item-root-name)

    (file-manager-write! cpu-fm "FunctionalInverses.h")
    (file-manager-write! cpu-fm "ViewMetaClasses.h")
    (file-manager-write! cpu-fm "ViewMetaClasses.cpp")
    (file-manager-write! cpu-fm "CompositeViewCopyKernels.cpp")

    ;; ---- gen_headers (gen_aggregated_headers) ----

    (file-manager-write! cpu-fm "NativeMetaFunctions.h")
    (file-manager-write! cpu-fm "MethodOperators.h")
    (file-manager-write! cpu-fm "Operators.h")
    (file-manager-write! cpu-fm "Functions.h")
    (file-manager-write! cpu-fm "NativeFunctions.h")
    (for-each
     (lambda (key)
       (when (member key functions-keys)
         (let ((fm (file-manager-for-key key cpu-fm cuda-fm)))
           (file-manager-write!
            fm (string-append key "Functions.h"))
           (file-manager-write!
            fm (string-append key "Functions_inl.h")))))
     dispatch-keys)

    (file-manager-write! core-fm "TensorBody.h")
    (file-manager-write! cpu-fm "RedispatchFunctions.h")
    (file-manager-write! cpu-fm "RegistrationDeclarations.h")
    (file-manager-write! cpu-fm "VmapGeneratedPlumbing.h")
    (file-manager-write! core-fm "aten_interned_strings.h")
    (file-manager-write! headeronly-fm "enum_tag.h")

    ;; ---- gen_declarations_yaml ----

    (file-manager-write! cpu-fm "Declarations.yaml")

    ;; ---- render ----

    (render-manifest fms)))

(define (render-manifest fms)
  (let ((entries
         (append-map
          (lambda (fm)
            (map
             (lambda (artifact)
               (list (artifact-fm artifact)
                     (artifact-file artifact)
                     (if (artifact-shard artifact)
                         (string-join (sort (artifact-ops artifact) string<?)
                                      " ")
                         "-")))
             (file-manager-artifacts fm)))
          fms)))
    (sort entries
          (lambda (a b)
            (let ((la (car a)) (lb (car b)))
              (or (string<? la lb)
                  (and (string=? la lb)
                       (let ((fa (cadr a)) (fb (cadr b)))
                         (or (string<? fa fb)
                             (and (string=? fa fb)
                                  (string<? (caddr a) (caddr b))))))))))))
