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

;; Thin FileManager equivalent, mirroring only the observable artifact-output
;; semantics of torchgen's torchgen.utils.FileManager:
;;
;;   - artifact identity (label + relative filename)
;;   - duplicate-write detection
;;   - sharded writes: `sid = string_stable_hash(key_fn(item)) % num_shards`,
;;     producing `{stem}_0` .. `{stem}_{num_shards-1}` plus a discarded
;;     `{stem}Everything` artifact.
;;
;; No templates are rendered; this records the artifact manifest that a later
;; emitter phase will turn into C++ content.

(define-module (sonic-cross file-manager)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:use-module (srfi srfi-13)
  #:use-module (sonic-cross sha1)
  #:export (make-file-manager
            file-manager?
            file-manager-label
            file-manager-install-dir
            file-manager-records
            file-manager-write!
            file-manager-write-sharded!
            file-manager-files
            file-manager-artifacts
            shard-index
            split-extension
            artifact?
            artifact-fm
            artifact-file
            artifact-shard
            artifact-ops))

(define (shard-index key num-shards)
  ;; torchgen utils.write_sharded: string_stable_hash(key_fn(item)) % num_shards
  (modulo (string-stable-hash key) num-shards))

(define (split-extension filename)
  ;; Returns (values stem extension), where extension includes the leading dot.
  (let ((idx (string-rindex filename #\.)))
    (if idx
        (values (substring filename 0 idx) (substring filename idx))
        (values filename ""))))

;; An artifact is one observable output file.  `shard` is either #f (unsharded)
;; or (index . num-shards).  `ops` is either #f (no operator membership) or a
;; multiset of root-name strings assigned to this artifact (sharded files).
(define-record-type <artifact>
  (make-artifact fm file shard ops)
  artifact?
  (fm artifact-fm)
  (file artifact-file)
  (shard artifact-shard)
  (ops artifact-ops))

;; A file manager owns a label (core/cpu/cpu_vec/cuda/ops/aoti/headeronly), an
;; install directory, and a mutable list of recorded artifacts.
(define-record-type <file-manager>
  (raw-make-file-manager label install-dir records)
  file-manager?
  (label file-manager-label)
  (install-dir file-manager-install-dir)
  (records file-manager-records))

(define (make-file-manager label install-dir)
  (raw-make-file-manager label install-dir (make-variable '())))

(define (file-manager-add! fm artifact)
  (variable-set! (file-manager-records fm)
                 (cons artifact (variable-ref (file-manager-records fm)))))

(define (file-manager-files fm)
  ;; Files in write order (reverse of the accumulated cons list).
  (map artifact-file (reverse (variable-ref (file-manager-records fm)))))

(define (file-manager-artifacts fm)
  ;; Artifacts in write order.
  (reverse (variable-ref (file-manager-records fm))))

(define (file-manager-contains? fm filename)
  (member filename (file-manager-files fm)))

(define (file-manager-write! fm filename)
  ;; Non-sharded write: records the artifact; duplicate writes are rejected
  ;; exactly as torchgen FileManager.write_with_template asserts.
  (when (file-manager-contains? fm filename)
    (error 'duplicate-file-write (file-manager-label fm) filename))
  (file-manager-add! fm
                     (make-artifact (file-manager-label fm) filename #f #f)))

(define (file-manager-write-sharded! fm filename num-shards items root-name)
  ;; Sharded write.  `root-name` maps an item to its stable-hash key (a
  ;; root-name string).  Produces num_shards artifacts; the "Everything" shard
  ;; is computed but discarded, matching torchgen.
  (call-with-values
      (lambda () (split-extension filename))
    (lambda (stem extension)
      (let ((membership (make-vector num-shards '())))
        (for-each
         (lambda (item)
           (let* ((key (root-name item))
                  (sid (shard-index key num-shards)))
             (vector-set! membership sid
                          (cons key (vector-ref membership sid)))))
         items)
        (do ((i 0 (1+ i)))
            ((>= i num-shards))
          (let* ((file (string-append stem "_" (number->string i) extension))
                 (ops (sort (vector-ref membership i) string<?)))
            (when (file-manager-contains? fm file)
              (error 'duplicate-file-write (file-manager-label fm) file))
            (file-manager-add!
             fm
             (make-artifact (file-manager-label fm) file
                            (cons i num-shards) ops))))))))
