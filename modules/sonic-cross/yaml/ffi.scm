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

(define-module (sonic-cross yaml ffi)
  #:use-module (ice-9 format)
  #:use-module (rnrs bytevectors)
  #:use-module (system base compile)
  #:use-module (system foreign)
  #:export (yaml-parse-events))

;; libyaml does not expose heap constructors for yaml_parser_t or
;; yaml_event_t.  Keep the supported ABI layouts explicit and select one by
;; Guile's configured host triplet.  The AArch64 profile is the LP64 layout
;; used by the arm64 server target and should be checked with an on-host ABI
;; probe when that target becomes available.
(define %abi-profiles
  `((x86_64
     (parser-size . 480)
     (event-size . 104)
     (scalar-anchor-offset . 8)
     (scalar-tag-offset . 16)
     (scalar-value-offset . 24)
     (scalar-length-offset . 32)
     (scalar-style-offset . 48))
    (aarch64
     (parser-size . 480)
     (event-size . 104)
     (scalar-anchor-offset . 8)
     (scalar-tag-offset . 16)
     (scalar-value-offset . 24)
     (scalar-length-offset . 32)
     (scalar-style-offset . 48))))

(define (host-abi-name)
  (cond
   ((string-prefix? "x86_64" %host-type) 'x86_64)
   ((or (string-prefix? "aarch64" %host-type)
        (string-prefix? "arm64" %host-type))
    'aarch64)
   (else
    (error 'yaml-error
           (format #f "unsupported libyaml ABI host type: ~a" %host-type)))))

(define %abi
  (or (cdr (assq (host-abi-name) %abi-profiles))
      (error 'yaml-error "missing libyaml ABI profile")))

(define %parser-size (assq-ref %abi 'parser-size))
(define %event-size (assq-ref %abi 'event-size))
(define %pointer-size (sizeof '*))

(define %libyaml (dynamic-link "libyaml"))

(define %parser-initialize
  (pointer->procedure int
                      (dynamic-func "yaml_parser_initialize" %libyaml)
                      (list '*)))
(define %parser-delete
  (pointer->procedure void
                      (dynamic-func "yaml_parser_delete" %libyaml)
                      (list '*)))
(define %parser-set-input-string
  (pointer->procedure void
                      (dynamic-func "yaml_parser_set_input_string" %libyaml)
                      (list '* '* size_t)))
(define %parser-parse
  (pointer->procedure int
                      (dynamic-func "yaml_parser_parse" %libyaml)
                      (list '* '*)))
(define %event-delete
  (pointer->procedure void
                      (dynamic-func "yaml_event_delete" %libyaml)
                      (list '*)))

(define (allocated-buffer size)
  (let ((bytes (make-bytevector size 0)))
    (values bytes (bytevector->pointer bytes))))

(define (pointer-at pointer offset)
  (make-pointer (+ (pointer-address pointer) offset)))

(define (read-pointer pointer offset)
  (let ((address (bytevector-u64-native-ref
                  (pointer->bytevector (pointer-at pointer offset)
                                       %pointer-size)
                  0)))
    (make-pointer address)))

(define (read-u32 pointer offset)
  (bytevector-u32-native-ref
   (pointer->bytevector (pointer-at pointer offset) 4)
   0))

(define (copy-c-string pointer)
  (if (null-pointer? pointer)
      #f
      (pointer->string pointer)))

(define (copy-c-string/length pointer length)
  (utf8->string (pointer->bytevector pointer length)))

(define (scalar-style value)
  (case value
    ((1) 'plain)
    ((2) 'single-quoted)
    ((3) 'double-quoted)
    ((4) 'literal)
    ((5) 'folded)
    (else 'any)))

(define (decode-event event)
  (case (read-u32 event 0)
    ((1) '(stream-start))
    ((2) '(stream-end))
    ((3) '(document-start))
    ((4) '(document-end))
    ((5) (list 'alias
               (copy-c-string
                (read-pointer event (assq-ref %abi 'scalar-anchor-offset)))))
    ((6)
     (let ((anchor (copy-c-string
                    (read-pointer event
                                  (assq-ref %abi 'scalar-anchor-offset))))
           (tag (copy-c-string
                 (read-pointer event (assq-ref %abi 'scalar-tag-offset))))
           (value-pointer (read-pointer event
                                        (assq-ref %abi 'scalar-value-offset)))
           (length (bytevector-u64-native-ref
                    (pointer->bytevector
                     (pointer-at event
                                 (assq-ref %abi 'scalar-length-offset))
                     8)
                    0))
           (style (read-u32 event (assq-ref %abi 'scalar-style-offset))))
       (list 'scalar
             (copy-c-string/length value-pointer length)
             anchor
             tag
             (scalar-style style))))
    ((7)
     (list 'sequence-start
           (copy-c-string
            (read-pointer event (assq-ref %abi 'scalar-anchor-offset)))
           (copy-c-string
            (read-pointer event (assq-ref %abi 'scalar-tag-offset)))))
    ((8) '(sequence-end))
    ((9)
     (list 'mapping-start
           (copy-c-string
            (read-pointer event (assq-ref %abi 'scalar-anchor-offset)))
           (copy-c-string
            (read-pointer event (assq-ref %abi 'scalar-tag-offset)))))
    ((10) '(mapping-end))
    (else (error 'yaml-error "unknown libyaml event type"))))

(define (yaml-parse-events input)
  (let* ((input-bytes (string->utf8 input))
         (parser-storage #f)
         (parser #f)
         (event-storage #f)
         (event #f))
    (call-with-values
        (lambda () (allocated-buffer %parser-size))
      (lambda (bytes pointer)
        (set! parser-storage bytes)
        (set! parser pointer)))
    (call-with-values
        (lambda () (allocated-buffer %event-size))
      (lambda (bytes pointer)
        (set! event-storage bytes)
        (set! event pointer)))
    (unless (= 1 (%parser-initialize parser))
      (error 'yaml-error "libyaml parser initialization failed"))
    (dynamic-wind
      (lambda ()
        (%parser-set-input-string parser
                                  (bytevector->pointer input-bytes)
                                  (bytevector-length input-bytes)))
      (lambda ()
        (let loop ((events '()))
          (if (= 0 (%parser-parse parser event))
              (error 'yaml-error "libyaml rejected YAML input")
              (let ((decoded
                     (dynamic-wind
                       (lambda () #t)
                       (lambda () (decode-event event))
                       (lambda () (%event-delete event)))))
                (if (eq? (car decoded) 'stream-end)
                    (reverse (cons decoded events))
                    (loop (cons decoded events)))))))
      (lambda ()
        (%parser-delete parser)
        (set! event-storage #f)
        (set! parser-storage #f)))))
