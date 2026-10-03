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

;; RFC 3174 SHA-1, reproduced exactly so that sharding hashes match
;; torchgen's utils.string_stable_hash (hashlib.sha1 over latin1 bytes,
;; reinterpreted as a little-endian integer).

(define-module (sonic-cross sha1)
  #:use-module (rnrs bytevectors)
  #:use-module (ice-9 iconv)
  #:export (sha1-bytevector
            string-stable-hash))

;; NOTE: rotate-left is expressed as a rotate-right (left shift by (32-n) bits)
;; so the operand of the large left shift is always a *small* value (the low
;; (32-n) bits, at most 2^31).  Guile 3.0.9 mis-evaluates `(ash word 30)` for a
;; full 32-bit word (negates the ~bit-58 result, which then trips the u32
;; store's range check).  Keeping the shifted operand small sidesteps the bug.

(define (rotl32 x n)
  (let* ((x (logand x #xffffffff))
         (m (- 32 n))
         (mask (- (ash 1 m) 1))
         (hi (logand (ash (logand x mask) n) #xffffffff))
         (lo (ash x (- m))))
    (logior hi lo)))

(define (u32 x) (logand x #xffffffff))

(define (sha1-round f k a b c d e w i)
  (let* ((wr (bytevector-u32-ref w (* i 4) (endianness big)))
         (rl (rotl32 a 5))
         (temp (u32 (+ rl f e k wr))))
    (values temp a (rotl32 b 30) c d)))

(define (sha1-round-const i)
  (cond ((< i 20) #x5A827999)
        ((< i 40) #x6ED9EBA1)
        ((< i 60) #x8F1BBCDC)
        (else #xCA62C1D6)))

(define (sha1-round-f b c d i)
  (cond ((< i 20) (logior (logand b c) (logand (lognot b) d)))
        ((< i 40) (logxor b c d))
        ((< i 60) (logior (logand b c) (logand b d) (logand c d)))
        (else (logxor b c d))))

(define (sha1-process-block padded off h0 h1 h2 h3 h4)
  (let ((w (make-bytevector 320)))
    (do ((i 0 (1+ i))) ((>= i 16))
      (bytevector-u32-set! w (* i 4)
                           (bytevector-u32-ref padded (+ off (* i 4))
                                               (endianness big))
                           (endianness big)))
    (do ((i 16 (1+ i))) ((>= i 80))
      (let* ((w3  (bytevector-u32-ref w (* (- i 3) 4)  (endianness big)))
             (w8  (bytevector-u32-ref w (* (- i 8) 4)  (endianness big)))
             (w14 (bytevector-u32-ref w (* (- i 14) 4) (endianness big)))
             (w16 (bytevector-u32-ref w (* (- i 16) 4) (endianness big)))
             (xor (logxor w3 w8 w14 w16))
             (v   (rotl32 xor 1)))
        (bytevector-u32-set! w (* i 4) v (endianness big))))
    (let loop ((i 0) (a h0) (b h1) (c h2) (d h3) (e h4))
      (if (>= i 80)
          (values (u32 (+ a h0)) (u32 (+ b h1)) (u32 (+ c h2))
                  (u32 (+ d h3)) (u32 (+ e h4)))
          (call-with-values
              (lambda ()
                (sha1-round (sha1-round-f b c d i) (sha1-round-const i)
                            a b c d e w i))
            (lambda (a b c d e) (loop (1+ i) a b c d e)))))))

(define (sha1-bytevector msg)
  (let* ((ml (bytevector-length msg))
         (bit-len (* ml 8))
         (pad-len (modulo (- 56 (modulo (1+ ml) 64)) 64))
         (total (+ ml 1 pad-len 8))
         (padded (make-bytevector total 0)))
    (bytevector-copy! msg 0 padded 0 ml)
    (bytevector-u8-set! padded ml #x80)
    (bytevector-u32-set! padded (- total 8) (ash bit-len -32) (endianness big))
    (bytevector-u32-set! padded (- total 4) (logand bit-len #xffffffff)
                         (endianness big))
    (let loop ((off 0)
               (h0 #x67452301) (h1 #xEFCDAB89) (h2 #x98BADCFE)
               (h3 #x10325476) (h4 #xC3D2E1F0))
      (if (>= off total)
          (let ((out (make-bytevector 20)))
            (bytevector-u32-set! out 0 h0 (endianness big))
            (bytevector-u32-set! out 4 h1 (endianness big))
            (bytevector-u32-set! out 8 h2 (endianness big))
            (bytevector-u32-set! out 12 h3 (endianness big))
            (bytevector-u32-set! out 16 h4 (endianness big))
            out)
          (call-with-values
              (lambda () (sha1-process-block padded off h0 h1 h2 h3 h4))
            (lambda (h0 h1 h2 h3 h4)
              (loop (+ off 64) h0 h1 h2 h3 h4)))))))

(define (string-stable-hash string)
  ;; Equivalent to torchgen utils.string_stable_hash: sha1 of the latin1
  ;; encoding, read back as a little-endian integer.
  (let ((digest (sha1-bytevector (string->bytevector string "ISO-8859-1"))))
    (let loop ((i 19) (acc 0))
      (if (< i 0)
          acc
          (loop (1- i) (+ (ash acc 8) (bytevector-u8-ref digest i)))))))
