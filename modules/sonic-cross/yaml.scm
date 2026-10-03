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

(define-module (sonic-cross yaml)
  #:use-module (ice-9 textual-ports)
  #:use-module (srfi srfi-9)
  #:use-module (sonic-cross yaml ffi)
  #:export (make-yaml-mapping
            yaml-mapping?
            yaml-mapping-entries
            yaml-mapping-anchor
            yaml-mapping-tag
            make-yaml-sequence
            yaml-sequence?
            yaml-sequence-items
            yaml-sequence-anchor
            yaml-sequence-tag
            make-yaml-scalar
            yaml-scalar?
            yaml-scalar-value
            yaml-scalar-anchor
            yaml-scalar-tag
            yaml-scalar-style
            make-yaml-null
            yaml-null
            yaml-null?
            yaml-null-anchor
            yaml-null-tag
            make-yaml-alias
            yaml-alias?
            yaml-alias-anchor
            yaml-load
            yaml-load-port
            yaml-load-string))

(define-record-type <yaml-mapping>
  (make-yaml-mapping entries anchor tag)
  yaml-mapping?
  (entries yaml-mapping-entries)
  (anchor yaml-mapping-anchor)
  (tag yaml-mapping-tag))

(define-record-type <yaml-sequence>
  (make-yaml-sequence items anchor tag)
  yaml-sequence?
  (items yaml-sequence-items)
  (anchor yaml-sequence-anchor)
  (tag yaml-sequence-tag))

(define-record-type <yaml-scalar>
  (make-yaml-scalar value anchor tag style)
  yaml-scalar?
  (value yaml-scalar-value)
  (anchor yaml-scalar-anchor)
  (tag yaml-scalar-tag)
  (style yaml-scalar-style))

(define-record-type <yaml-null>
  (make-yaml-null anchor tag)
  yaml-null?
  (anchor yaml-null-anchor)
  (tag yaml-null-tag))

(define yaml-null (make-yaml-null #f #f))

(define-record-type <yaml-alias>
  (make-yaml-alias anchor)
  yaml-alias?
  (anchor yaml-alias-anchor))

(define (plain-null? value style)
  (and (eq? style 'plain)
       (member value '("" "~" "null" "Null" "NULL"))))

(define (event-node events)
  (let ((event (car events)))
    (case (car event)
      ((scalar)
       (let ((value (cadr event))
             (anchor (caddr event))
             (tag (cadddr event))
             (style (car (cddddr event))))
         (values (if (plain-null? value style)
                     (make-yaml-null anchor tag)
                     (make-yaml-scalar value anchor tag style))
                 (cdr events))))
      ((alias)
       (values (make-yaml-alias (cadr event)) (cdr events)))
      ((sequence-start)
       (let loop ((rest (cdr events)) (items '()))
         (let ((next (car rest)))
           (if (eq? (car next) 'sequence-end)
               (values (make-yaml-sequence (reverse items)
                                           (cadr event)
                                           (caddr event))
                       (cdr rest))
               (call-with-values
                   (lambda () (event-node rest))
                 (lambda (item remaining)
                   (loop remaining (cons item items))))))))
      ((mapping-start)
       (let loop ((rest (cdr events)) (entries '()))
         (let ((next (car rest)))
           (if (eq? (car next) 'mapping-end)
               (values (make-yaml-mapping (reverse entries)
                                          (cadr event)
                                          (caddr event))
                       (cdr rest))
               (call-with-values
                   (lambda () (event-node rest))
                 (lambda (key after-key)
                   (call-with-values
                       (lambda () (event-node after-key))
                     (lambda (value remaining)
                       (loop remaining
                             (cons (list key value) entries))))))))))
      (else (error 'yaml-error "unexpected libyaml event")))))

(define (yaml-load-string input)
  (let* ((events (yaml-parse-events input))
         (after-stream-start (cdr events)))
    (unless (and (pair? after-stream-start)
                 (eq? (car (car after-stream-start)) 'document-start))
      (error 'yaml-error "YAML stream has no document"))
    (call-with-values
        (lambda () (event-node (cdr after-stream-start)))
      (lambda (root remaining)
        (unless (and (pair? remaining)
                     (eq? (car (car remaining)) 'document-end))
          (error 'yaml-error "YAML document did not end correctly"))
        (let ((after-document (cdr remaining)))
          (when (and (pair? after-document)
                     (eq? (car (car after-document)) 'document-start))
            (error 'yaml-error "multiple YAML documents are not supported"))
          root)))))

(define (yaml-load-port port)
  (yaml-load-string (get-string-all port)))

(define (yaml-load filename)
  (catch #t
    (lambda ()
      (call-with-input-file filename yaml-load-port))
    (lambda (key . args)
      (cond
       ((eq? key 'yaml-error)
        (apply error key args))
       ;; system-error args are (who format-string format-args errno); the
       ;; first format-arg is the OS message ("No such file or directory"),
       ;; which is what we want instead of the bare key "system-error".
       ((eq? key 'system-error)
        (error 'yaml-error
               (format #f "unable to load YAML file ~a: ~a"
                       filename (car (caddr args)))))
       (else
        (error 'yaml-error
               (format #f "unable to load YAML file ~a: ~a"
                       filename key)))))))
