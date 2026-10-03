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

(define-module (sonic-cross generated-functions)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-13)
  #:use-module (sonic-cross core-ir)
  #:use-module (sonic-cross native-function)
  #:export (pre-group-native-functions
            generate-out-args-from-schema
            self-to-out-schema
            mutable-to-out-schema
            functional-to-out-schema
            add-generated-native-functions!
            add-generated-native-functions
            unambiguous-name
            base-operator-spelling))

(define out-ops-that-dont-get-grouped-properly
  '("adaptive_avg_pool3d_backward.grad_input"
    "_slow_conv2d_backward.grad_input"))

(define mutable-ops-that-cannot-get-an-out-variant
  '("_cummax_helper" "_cummin_helper"))

(define functional-ops-that-cannot-get-an-out-variant
  '("_assert_async"
    "_assert_async.msg"
    "_assert_tensor_metadata"
    "_cslt_sparse_mm_search"
    "_assert_scalar"
    "_dimI"
    "_dimV"
    "_has_same_storage_numel"
    "_linalg_check_errors"
    "_local_scalar_dense"
    "_nested_tensor_from_mask_left_aligned"
    "_nnz"
    "_use_cudnn_ctc_loss"
    "_use_cudnn_ctc_loss.Tensor"
    "_use_miopen_ctc_loss"
    "_use_miopen_ctc_loss.Tensor"
    "_validate_compressed_sparse_indices"
    "allclose"
    "dense_dim"
    "equal"
    "is_coalesced"
    "is_pinned"
    "is_same_size"
    "is_set_to"
    "q_per_channel_axis"
    "q_scale"
    "q_zero_point"
    "qscheme"
    "record_stream"
    "sparse_dim"
    "sym_constrain_range"
    "sym_constrain_range_for_size"
    "_nested_tensor_storage_offsets"
    "_chunk_grad_outputs_efficient_attention"
    "_fused_sdp_choice"
    "_print"
    "_sink_tokens"
    "_nested_get_ragged_idx"))

(define (operator-string func)
  (operator-name->string (function-schema-name func)))

(define (base-operator-spelling func)
  ;; Reproduces torchgen's str(BaseOperatorName): the full base spelling
  ;; without the overload name, carrying the "_"/"_functional" suffix and
  ;; dunder-method wrapping.
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

(define (schema-kind-of native-function)
  (function-schema-kind (native-function-func native-function)))

(define (group-find groups signature)
  (find (lambda (group) (equal? (car group) signature)) groups))

(define (pre-group-native-functions native-functions)
  (fold (lambda (native-function groups)
          (let* ((func (native-function-func native-function))
                 (signature (function-schema-signature func))
                 (kind (schema-kind-of native-function))
                 (group (group-find groups signature)))
            (if group
                (begin
                  (when (assq kind (cdr group))
                    (error 'duplicate-schema-kind kind))
                  (map (lambda (candidate)
                         (if (eq? candidate group)
                             (cons (car candidate)
                                   (append (cdr candidate)
                                           (list (cons kind native-function))))
                             candidate))
                       groups))
                (append groups
                        (list (cons signature
                                    (list (cons kind native-function))))))))
        '()
        native-functions))

(define (group-ref group kind)
  (let ((entry (assq kind (cdr group))))
    (and entry (cdr entry))))

(define (group-functions group)
  (map cdr (cdr group)))

(define (group-is-core? group)
  (any (lambda (function)
         (member "core" (native-function-tags function)))
       (group-functions group)))

(define (group-all-composite-implicit? group)
  (every native-function-has-composite-implicit-autograd-kernel?
         (group-functions group)))

(define (expected-out-overload overload)
  (if (string-null? overload) "out"
      (string-append overload "_out")))

(define (schema-name-with-out-overload func remove-inplace?)
  (let* ((operator (function-schema-name func))
         (old-base (operator-name-base operator))
         (base (make-base-operator-name
                (base-operator-name-base old-base)
                (if remove-inplace? #f (base-operator-name-inplace? old-base))
                (base-operator-name-dunder-method? old-base)
                (base-operator-name-functional-overload? old-base)))
         (name (make-operator-name
                base
                (expected-out-overload
                 (operator-name-overload-name operator)))))
    name))

(define (copy-argument-without-annotation argument)
  (make-argument (argument-name argument)
                 (argument-type argument)
                 (argument-default argument)
                 #f
                 #f))

(define (append-out-arguments arguments out-arguments)
  (make-arguments
   (arguments-pre-self-positional arguments)
   (arguments-self-arg arguments)
   (arguments-post-self-positional arguments)
   (arguments-pre-tensor-options-kwarg-only arguments)
   (arguments-tensor-options arguments)
   (arguments-post-tensor-options-kwarg-only arguments)
   (append (arguments-out arguments) out-arguments)))

(define (used-argument-aliases arguments)
  (apply append
         (map (lambda (argument)
                (let ((annotation (argument-annotation argument)))
                  (if (and (annotation? annotation)
                           (pair? (annotation-alias-set annotation)))
                      (annotation-alias-set annotation)
                      '())))
              (arguments-all arguments))))

(define (fresh-aliases used count)
  (let ((available
         (filter (lambda (alias) (not (member alias used)))
                 (map string (string->list "abcdefghijklmnopqrstuvwxyz")))))
    (if (< (length available) count)
        (error 'not-enough-fresh-aliases count)
        (take available count))))

(define (return-has-write-annotation? return)
  (let ((annotation (return-annotation return)))
    (and (annotation? annotation)
         (annotation-is-write? annotation))))

(define (generate-out-args-from-schema func)
  (let* ((returns (function-schema-returns func))
         (tensor-returns
          (filter (lambda (return)
                    (type-is-tensor-like? (return-type return)))
                  returns))
         (used (used-argument-aliases
                (function-schema-arguments func)))
         (aliases (fresh-aliases used (length tensor-returns)))
         (all-plain-tensor?
          (every (lambda (return)
                   (equal? (return-type return) (make-tensor-type)))
                 returns)))
    (when (any return-has-write-annotation? returns)
      (error 'mutable-return-in-out-generation))
    (unless (pair? tensor-returns)
      (error 'no-tensor-like-return))
    (let loop ((remaining returns)
               (tensor-index 0)
               (out-arguments '())
               (new-returns '()))
      (if (null? remaining)
          (values (reverse new-returns) (reverse out-arguments))
          (let* ((return (car remaining))
                 (type (return-type return)))
            (if (type-is-tensor-like? type)
                (let* ((name (if (= (length returns) 1)
                                 "out"
                                 (string-append "out"
                                                (number->string tensor-index))))
                       (annotation
                        (make-annotation
                         (list (list-ref aliases tensor-index)) #t #f))
                       (argument (make-argument name type #f annotation #t))
                       (new-return
                        (make-return #f type annotation)))
                  (loop (cdr remaining)
                        (1+ tensor-index)
                        (cons argument out-arguments)
                        (if all-plain-tensor?
                            (cons new-return new-returns)
                            new-returns)))
                (loop (cdr remaining)
                      tensor-index
                      out-arguments
                      (cons return new-returns))))))))

(define (self-to-out-schema func)
  (unless (eq? (function-schema-kind func) schema-kind-inplace)
    (error 'expected-inplace-schema))
  (let ((self (arguments-self-arg (function-schema-arguments func))))
    (unless self (error 'missing-self-argument))
    (let* ((self-argument
            (if (self-argument? self)
                (self-argument-argument self)
                self))
           (arguments (function-schema-arguments func))
           (new-arguments
            (make-arguments
             (arguments-pre-self-positional arguments)
             (make-self-argument
              (copy-argument-without-annotation self-argument))
             (arguments-post-self-positional arguments)
             (arguments-pre-tensor-options-kwarg-only arguments)
             (arguments-tensor-options arguments)
             (arguments-post-tensor-options-kwarg-only arguments)
             (append (arguments-out arguments)
                     (list (make-argument
                            "out"
                            (argument-type self-argument)
                            #f
                            (argument-annotation self-argument)
                            #t)))))
           (new-func
            (make-function-schema
             (schema-name-with-out-overload func #t)
             new-arguments
             (function-schema-returns func))))
      new-func)))

(define (mutable-to-out-schema func)
  (unless (eq? (function-schema-kind func) schema-kind-mutable)
    (error 'expected-mutable-schema))
  (call-with-values
      (lambda () (generate-out-args-from-schema func))
    (lambda (returns out-arguments)
      (make-function-schema
       (schema-name-with-out-overload func #t)
       (append-out-arguments (function-schema-arguments func)
                             out-arguments)
       returns))))

(define (functional-to-out-schema func)
  (unless (eq? (function-schema-kind func) schema-kind-functional)
    (error 'expected-functional-schema))
  (call-with-values
      (lambda () (generate-out-args-from-schema func))
    (lambda (returns out-arguments)
      (let ((signature (function-schema-signature
                        func #:keep-return-names #t)))
        (make-function-schema
         (schema-name-with-out-overload func #f)
         (append-out-arguments (function-schema-arguments signature)
                               out-arguments)
         returns)))))

(define (unambiguous-name func)
  ;; Mirrors torchgen OperatorName.unambiguous_name(): str(BaseOperatorName),
  ;; plus "_<overload>" when an overload is present.
  (let* ((operator (function-schema-name func))
         (base (base-operator-spelling func))
         (overload (operator-name-overload-name operator)))
    (if (string-null? overload)
        base
        (string-append base "_" overload))))

(define (cpp-name func)
  (string-append
   (base-operator-spelling func)
   (if (function-schema-is-out-fn? func) "_out" "")))

(define (generated-tags source func)
  (let ((inherited
         (filter (lambda (tag)
                   (member tag '("nondeterministic_seeded"
                                 "view_copy"
                                 "pt2_compliant_tag")))
                 (native-function-tags source))))
    (delete-duplicates
     (append '("generated")
             inherited
             (if (function-schema-is-out-fn? func) '("out") '())
             (if (base-operator-name-inplace?
                  (operator-name-base (function-schema-name func)))
                 '("inplace")
                 '()))
     string=?)))

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

(define (generate-function source target)
  (let* ((source-func (native-function-func source))
         (source-kind (function-schema-kind source-func))
         (func
          (cond
           ((eq? target schema-kind-functional)
            (when (eq? source-kind schema-kind-functional)
              (error 'cannot-generate-functional-from-functional))
            (let* ((signature (function-schema-signature
                               source-func #:keep-return-names #t))
                   (old-name (function-schema-name source-func))
                   (old-base (operator-name-base old-name))
                   (new-name
                    (make-operator-name
                     (make-base-operator-name
                      (base-operator-name-base old-base)
                      #f
                      (base-operator-name-dunder-method? old-base)
                      (eq? source-kind schema-kind-mutable))
                     (operator-name-overload-name old-name))))
              (make-function-schema
               new-name
               (function-schema-arguments signature)
               (function-schema-returns signature))))
           ((eq? target schema-kind-out)
            (cond
             ((eq? source-kind schema-kind-inplace)
              (self-to-out-schema source-func))
             ((eq? source-kind schema-kind-mutable)
              (mutable-to-out-schema source-func))
             ((eq? source-kind schema-kind-functional)
              (functional-to-out-schema source-func))
             (else (error 'unsupported-out-source source-kind))))
           (else (error 'unsupported-generated-schema-kind target)))))
    (let* ((kernel-base (if (eq? target schema-kind-out)
                            (unambiguous-name func)
                            (cpp-name func)))
           (kernel (if (function-schema-has-symint? source-func)
                       (string-append kernel-base "_symint")
                       kernel-base))
           (metadata (make-backend-metadata kernel #f "at::native"))
           (tags (generated-tags source func))
           (generated
            (make-native-function
             func
             (native-function-namespace source)
             (native-function-use-const-ref-for-mutable-tensors? source)
             #f "NoCheck" #f #f '("function") #f #f
             (native-function-loc source) '() #f #f #f #f #f '()
             (native-function-is-abstract? source)
             #f #f #t #f tags '("CompositeExplicitAutograd")))
           (operator (operator-name->string (function-schema-name func))))
      (values generated
              (list (cons "CompositeExplicitAutograd"
                          (list (cons operator metadata))))))))

(define (invalid-base-exception? name)
  (or (member name mutable-ops-that-cannot-get-an-out-variant)
      (member name functional-ops-that-cannot-get-an-out-variant)))

(define (append-generated result indices generated metadata)
  (let* ((dispatch-entry (car metadata))
         (dispatch-key (car dispatch-entry))
         (function-entry (car (cdr dispatch-entry))))
    (values
     (append result (list generated))
     (index-add indices dispatch-key
                (car function-entry)
                (cdr function-entry)))))

(define (process-generation-group group result indices)
  (let* ((functional (group-ref group schema-kind-functional))
         (inplace (group-ref group schema-kind-inplace))
         (mutable (group-ref group schema-kind-mutable))
         (out (group-ref group schema-kind-out))
         (has-functional (and functional #t))
         (has-inplace (and inplace #t))
         (has-mutable (and mutable #t))
         (has-out (and out #t))
         (functions (group-functions group)))
    (if (not (or has-functional has-inplace has-mutable has-out))
        (values result indices)
        (cond
         ((every native-function-manual-cpp-binding? functions)
          (values result indices))
         ((any (lambda (function)
                 (and (native-function-is-view-op? function)
                      (not (string=?
                            (base-operator-spelling
                             (native-function-func function))
                            "set_"))))
               functions)
          (values result indices))
         ((and (group-all-composite-implicit? group)
               (not (group-is-core? group)))
          (values result indices))
         ((and has-out (not has-functional)
               (not has-inplace) (not has-mutable))
          (if (member (operator-string (native-function-func out))
                      out-ops-that-dont-get-grouped-properly)
              (values result indices)
              (error 'unpaired-out-schema
                     (operator-string (native-function-func out)))))
         ((and has-inplace
               (string=? (operator-string (native-function-func inplace))
                         "polygamma_"))
          (values result indices))
         (else
          (let* ((base (cond (mutable mutable)
                             (inplace inplace)
                             (out out)
                             (else functional)))
                 (base-func (native-function-func base))
                 (base-valid
                  (or (eq? (function-schema-kind base-func)
                           schema-kind-inplace)
                      (any (lambda (return)
                             (type-is-tensor-like? (return-type return)))
                           (function-schema-returns base-func))))
                 (needs-out
                  (any (lambda (operator)
                         (string-contains
                          (operator-name->string operator) "out"))
                       (native-function-autogen base)))
                 (gets-out (and (not has-out) base-valid needs-out)))
            (unless (or has-out base-valid
                        (invalid-base-exception? (operator-string base-func)))
              (error 'invalid-base-function (operator-string base-func)))
            (call-with-values
                (lambda ()
                  (if gets-out
                      (generate-function base schema-kind-out)
                      (values #f #f)))
              (lambda (generated metadata)
                (if generated
                    (call-with-values
                        (lambda ()
                          (append-generated result indices generated metadata))
                      (lambda (new-result new-indices)
                        (process-functional-generation
                         group base has-functional has-out gets-out
                         new-result new-indices)))
                    (process-functional-generation
                     group base has-functional has-out gets-out
                     result indices))))))))))

(define (process-functional-generation group base has-functional has-out gets-out
                                       result indices)
  (if (and (not has-functional) (or has-out gets-out))
      (call-with-values
          (lambda () (generate-function base schema-kind-functional))
        (lambda (generated metadata)
          (append-generated result indices generated metadata)))
      (values result indices)))

(define (add-generated-native-functions! native-functions indices)
  (let loop ((groups (pre-group-native-functions native-functions))
             (result native-functions)
             (updated-indices indices))
    (if (null? groups)
        (values result updated-indices)
        (call-with-values
            (lambda ()
              (process-generation-group (car groups) result updated-indices))
          (lambda (new-result new-indices)
            (loop (cdr groups) new-result new-indices))))))

(define add-generated-native-functions add-generated-native-functions!)
