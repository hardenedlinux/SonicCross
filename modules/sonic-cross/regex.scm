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

(define-module (sonic-cross regex)
  #:use-module (sonic-cross irregex)
  #:export (regex-compile regex-match regex-capture))

;; The parser uses numbered captures because the bundled irregex API does
;; not accept Python's (?P<name>...) spelling.  This is intentionally a
;; small adapter, rather than a Python regular-expression compatibility
;; layer.
(define (regex-compile pattern)
  (string->irregex pattern))

(define (regex-match pattern text)
  (irregex-match pattern text))

(define (regex-capture match index)
  (and match (irregex-match-substring match index)))
